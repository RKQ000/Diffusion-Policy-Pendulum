import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import os

from diffusion_model import DiffusionPolicy
from pendulum_env import PendulumEnvWrapper, RobustSwingUpExpert
import config


def test_diffusion_policy(model_path="checkpoints/diffusion_policy_final.pt", 
                          num_episodes=10, render=True,
                          init_angle=None, init_ang_vel=0.0,
                          compare_with_expert=True):
    """
    测试扩散策略模型，并可选择与专家策略对比（仅在第一个 episode 的相同初值下）
    
    Args:
        model_path: 扩散模型路径
        num_episodes: 测试回合数
        render: 是否渲染
        init_angle: 初始角度（弧度），None 表示随机
        init_ang_vel: 初始角速度（rad/s）
        compare_with_expert: 是否与专家策略对比并绘制曲线（需要 init_angle 不为 None）
    """
    
    # 如果要求对比但未指定初始角度，则自动使用默认角度并提示
    if compare_with_expert and init_angle is None:
        init_angle = np.random.uniform(-np.pi, np.pi)
    
    # 加载扩散模型
    cfg = config.Config()
    model = DiffusionPolicy(cfg).to(cfg.device)
    model.load_state_dict(torch.load(model_path, map_location=cfg.device))
    model.eval()
    
    # 创建环境
    env = PendulumEnvWrapper(render=render)
    expert = RobustSwingUpExpert()   # 专家策略
    
    episode_rewards = []
    
    # 存储扩散策略的数据（仅第一个 episode）
    diffusion_data = None
    
    print("=" * 50)
    print(f"开始测试 Diffusion Policy: {num_episodes} 个回合")
    print("=" * 50)
    
    for episode in range(num_episodes):
        # 设置初始状态
        if init_angle is not None:
            initial_state = (init_angle, init_ang_vel)
            obs = env.reset(initial_state=initial_state)
        else:
            obs = env.reset()
        
        episode_reward = 0.0
        done = False
        step = 0
        episode_theta = []
        episode_theta_dot = []
        episode_action = []
        
        while not done:
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(cfg.device)
            with torch.no_grad():
                action_seq = model.generate_action(obs_tensor)
            
            for i in range(cfg.n_action_steps):
                # 记录当前状态（执行动作前）
                cos_theta, sin_theta, theta_dot = obs
                theta = math.atan2(sin_theta, cos_theta)
                episode_theta.append(theta)
                episode_theta_dot.append(theta_dot)
                
                action = action_seq[0, i, 0].cpu().numpy()
                episode_action.append(action)
                
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                step += 1
                
                if done or step >= cfg.max_episode_len:
                    break
            if done or step >= cfg.max_episode_len:
                break
        
        episode_rewards.append(episode_reward)
        print(f"Diffusion Policy Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {step}")
        
        # 保存第一个 episode 的扩散策略数据
        if episode == 0:
            diffusion_data = {
                "time": list(range(len(episode_theta))),
                "theta": episode_theta,
                "theta_dot": episode_theta_dot,
                "action": episode_action
            }
    
    # 如果启用对比，运行专家策略（相同初始状态）
    expert_data = None
    print("\n" + "=" * 50)
    print("运行专家策略 (RobustSwingUpExpert) 进行对比...")
    print("=" * 50)
    
    # 重新创建环境（避免干扰）
    env_expert = PendulumEnvWrapper(render=False)
    obs = env_expert.reset(initial_state=(init_angle, init_ang_vel))
    
    expert_theta = []
    expert_theta_dot = []
    expert_action = []
    step = 0
    done = False
    total_reward = 0.0
    
    while not done and step < cfg.max_episode_len:
        cos_theta, sin_theta, theta_dot = obs
        theta = math.atan2(sin_theta, cos_theta)
        expert_theta.append(theta)
        expert_theta_dot.append(theta_dot)
        
        action = expert.get_action(obs)
        expert_action.append(action)
        
        obs, reward, done, _ = env_expert.step(action)
        total_reward += reward
        step += 1
    
    env_expert.close()
    print(f"Expert Episode: Reward = {total_reward:.2f}, Steps = {step}")
    
    expert_data = {
        "time": list(range(len(expert_theta))),
        "theta": expert_theta,
        "theta_dot": expert_theta_dot,
        "action": expert_action
    }
    
    # 绘制对比曲线
    plt.figure(figsize=(14, 12))
    
    # 角度对比
    plt.subplot(3, 1, 1)
    plt.plot(diffusion_data["time"], diffusion_data["theta"], 'b-', linewidth=1.5, label='Diffusion Policy')
    plt.plot(expert_data["time"], expert_data["theta"], 'r--', linewidth=1.5, label='Expert (SwingUp+PD)')
    plt.axhline(y=0, color='k', linestyle=':', linewidth=0.8)
    plt.ylabel('Angle (rad)')
    plt.title(f'Pendulum Control Comparison (Initial Angle = {init_angle:.2f} rad)')
    plt.legend()
    plt.grid(True)
    
    # 角速度对比
    plt.subplot(3, 1, 2)
    plt.plot(diffusion_data["time"], diffusion_data["theta_dot"], 'g-', linewidth=1.5, label='Diffusion Policy')
    plt.plot(expert_data["time"], expert_data["theta_dot"], 'm--', linewidth=1.5, label='Expert')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.grid(True)
    
    # 控制力矩对比
    plt.subplot(3, 1, 3)
    plt.plot(diffusion_data["time"], diffusion_data["action"], 'c-', linewidth=1.5, label='Diffusion Policy')
    plt.plot(expert_data["time"], expert_data["action"], 'orange', linestyle='--', linewidth=1.5, label='Expert')
    plt.axhline(y=2.0, color='k', linestyle=':', linewidth=0.8)
    plt.axhline(y=-2.0, color='k', linestyle=':', linewidth=0.8)
    plt.xlabel('Time step')
    plt.ylabel('Torque (u)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("comparison_diffusion_vs_expert.png", dpi=150)
    plt.show()
    print("对比曲线已保存为 comparison_diffusion_vs_expert.png")
    
    env.close()
    
    # 统计扩散策略的奖励
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print("\n" + "=" * 50)
    print(f"Diffusion Policy 测试完成!")
    print(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"奖励范围: [{np.min(episode_rewards):.2f}, {np.max(episode_rewards):.2f}]")
    
    return episode_rewards