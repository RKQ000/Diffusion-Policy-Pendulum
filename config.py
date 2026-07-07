# config.py
import torch

class Config:
    # 环境参数
    env_name = "Pendulum-v1"
    obs_dim = 3          # [cosθ, sinθ, θ̇]
    action_dim = 1         # 力矩，范围[-2, 2]
    action_low = -2.0
    action_high = 2.0
    
    
    # Diffusion Policy 参数
    horizon = 16           # 动作序列长度（预测未来16步）
    n_obs_steps = 2        # 观测历史步数（输入最近2步状态）
    n_action_steps = 8     # 执行步数（每次执行预测结果的前8步）
    
    # 扩散过程参数
    num_diffusion_steps = 200  # 扩散步数 T
    beta_start = 1e-4
    beta_end = 0.02
    
    # 模型架构
    model_type = "unet"    # "unet" 或 "transformer"
    d_model = 128          # 隐藏维度
    n_head = 4             # Transformer多头数（仅当model_type="transformer"时生效）
    n_layers = 4           # Transformer层数
    
    # 训练参数
    batch_size = 512
    epochs = 400
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据收集参数
    num_episodes = 1024     # 收集演示数据的回合数
    max_episode_len = 200  # 每回合最大步数
    demo_data_path = "demonstrations.pkl"
    
    # 测试参数
    test_episodes = 10     # 测试回合数
    
config = Config()