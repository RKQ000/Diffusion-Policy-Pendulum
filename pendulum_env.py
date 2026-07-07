import gymnasium as gym
import numpy as np
import math

class PendulumEnvWrapper:
    def __init__(self, render=False):
        self.env = gym.make("Pendulum-v1", render_mode="human" if render else None)
        self.state_dim = 3
        self.action_dim = 1
        self.action_low = -2.0
        self.action_high = 2.0
        
    def reset(self, initial_state=None):
        # 先正常 reset，确保包装器内部标志正确
        obs, info = self.env.reset()
        
        if initial_state is not None:
            theta, theta_dot = initial_state
            theta = np.clip(theta, -np.pi, np.pi)
            # 覆盖底层环境的状态
            self.env.unwrapped.state = np.array([theta, theta_dot], dtype=np.float32)
            # 重新计算观测
            obs = np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
        return obs
    
    def step(self, action):
        # 强制转换为形状 (1,) 的数组
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, self.action_low, self.action_high)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
    
    def render(self):
        self.env.render()
        
    def close(self):
        self.env.close()


class RobustSwingUpExpert:
    """
    倒立摆起摆与镇定控制器（参数已调优）
    使用能量控制起摆 + 高增益PD镇定
    """
    def __init__(self):
        # 镇定参数（高增益，确保捕获）
        self.kp_stabilize = 8.0
        self.kd_stabilize = 1.5
        
        # 起摆参数（能量控制）
        self.energy_gain = 0.8      # 能量误差增益
        self.target_energy = 2.0    # 直立时的归一化能量（cosθ=-1 => 1-(-1)=2）
        
        # 切换角度（弧度）
        self.switch_angle = 0.4     # 约23°，更早进入镇定区
        
    def get_action(self, state):
        cos_theta, sin_theta, theta_dot = state
        theta = math.atan2(sin_theta, cos_theta)
        abs_theta = abs(theta)
        
        # 镇定区域：角度小于切换阈值
        if abs_theta < self.switch_angle:
            # 高增益PD控制
            u = -self.kp_stabilize * theta - self.kd_stabilize * theta_dot
        else:
            # 能量起摆
            # 归一化当前能量：势能 (1 - cosθ) + 动能 0.5*θ̇²
            current_energy = (1 - cos_theta) + 0.5 * theta_dot * theta_dot
            energy_error = self.target_energy - current_energy
            
            # 能量控制律：u = k * energy_error * sign(θ̇ * cosθ)
            # 利用 cosθ 符号决定方向，使摆向直立方向运动
            direction = 1.0 if theta_dot * cos_theta > 0 else -1.0
            u = self.energy_gain * energy_error * direction
            
            # 限制最大力矩，避免失控
            u = np.clip(u, -1.5, 1.5)
        
        # 最终限幅
        return np.clip(u, -2.0, 2.0)


def collect_demonstrations(config, expert_policy=None, render=False):
    """
    收集演示数据
    返回: list of dict, 每个dict对应一个episode:
        {
            "observations": np.array [T, state_dim],
            "actions": np.array [T, action_dim],
            "rewards": np.array [T]
        }
    """
    if expert_policy is None:
        expert_policy = RobustSwingUpExpert()
    
    env = PendulumEnvWrapper(render=render)
    episodes = []
    
    for episode in range(config.num_episodes):
        obs = env.reset()
        episode_obs, episode_actions, episode_rewards = [], [], []
        
        for step in range(config.max_episode_len):
            action = expert_policy.get_action(obs)
            # 添加探索噪声
            if episode < config.num_episodes * 0.8:
                noise = np.random.normal(0, 0.1)
                action = np.clip(action + noise, -2.0, 2.0)
            
            next_obs, reward, done, _ = env.step(action)
            
            episode_obs.append(obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            obs = next_obs
            if done:
                break
        
        episodes.append({
            "observations": np.array(episode_obs, dtype=np.float32),
            "actions": np.array(episode_actions, dtype=np.float32).reshape(-1, 1),
            "rewards": np.array(episode_rewards, dtype=np.float32)
        })
        
        if (episode + 1) % 50 == 0:
            print(f"收集演示数据: Episode {episode + 1}/{config.num_episodes}")
    
    env.close()
    print(f"演示数据收集完成: {len(episodes)} 个回合, 总步数 {sum(len(e['observations']) for e in episodes)}")
    return episodes