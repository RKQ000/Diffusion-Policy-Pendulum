# train_diffusion_policy.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt

from diffusion_model import DiffusionPolicy
from pendulum_dataset import get_dataloader
from pendulum_env import collect_demonstrations


def train_diffusion_policy(config):
    # 1. 收集或加载演示数据
    print("=" * 50)
    print("收集演示数据...")
    if os.path.exists(config.demo_data_path):
        print(f"加载已有演示数据: {config.demo_data_path}")
        with open(config.demo_data_path, "rb") as f:
            episodes = pickle.load(f)
        # 兼容旧格式：如果是字典（旧格式），转换为列表
        if isinstance(episodes, dict):
            print("检测到旧格式数据，请删除 demonstrations.pkl 后重新收集！")
            raise ValueError("旧格式数据包含跨 episode 拼接，请删除文件并重新收集")
    else:
        episodes = collect_demonstrations(config)
        with open(config.demo_data_path, "wb") as f:
            pickle.dump(episodes, f)
    
    # 2. 创建数据加载器
    dataloader = get_dataloader(episodes, config)
    
    # 3. 创建模型
    model = DiffusionPolicy(config).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()
    
    # 可选：学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # 4. 训练循环
    print("=" * 50)
    print(f"开始训练 (设备: {config.device})")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Epochs: {config.epochs}")
    print(f"   扩散步数: {config.num_diffusion_steps}")
    print(f"   Horizon: {config.horizon}")
    print("=" * 50)
    
    losses = []
    
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)
        
        for batch in progress_bar:
            action_seq = batch["action_seq"].to(config.device)
            current_obs = batch["current_obs"].to(config.device)
            
            B = action_seq.shape[0]
            t = torch.randint(0, config.num_diffusion_steps, (B,), device=config.device)
            noise = torch.randn_like(action_seq)
            x_t = model.q_sample(action_seq, t, noise)
            pred_noise = model.denoise_net(x_t, current_obs, t)
            loss = loss_fn(pred_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.6f}")
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/diffusion_policy_epoch_{epoch+1}.pt")
    
    # 5. 保存最终模型
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/diffusion_policy_final.pt")
    print(f"训练完成！最终模型已保存至 checkpoints/diffusion_policy_final.pt")
    
    # 6. 绘制训练损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Diffusion Policy Training Loss")
    plt.grid(True)
    plt.savefig("training_loss.png", dpi=150)
    plt.show()
    
    return model, losses