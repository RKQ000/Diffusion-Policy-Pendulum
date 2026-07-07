# pendulum_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TrajectoryDataset(Dataset):
    def __init__(self, episodes, config):
        self.config = config
        self.horizon = config.horizon
        self.n_obs_steps = config.n_obs_steps
        self.samples = []
        
        for ep in episodes:
            observations = ep["observations"]  # [T, obs_dim]
            actions = ep["actions"]            # [T, action_dim]
            T = len(observations)
            
            # 对当前 episode 内部滑动窗口
            for start_idx in range(T - self.horizon + 1):
                # 观测窗口: [start_idx - n_obs_steps + 1, start_idx]
                obs_start = max(0, start_idx - self.n_obs_steps + 1)
                obs_window = observations[obs_start : start_idx + 1]
                # 填充不足部分（用第一个观测重复）
                if len(obs_window) < self.n_obs_steps:
                    pad_len = self.n_obs_steps - len(obs_window)
                    obs_window = np.concatenate([np.tile(obs_window[0:1], (pad_len, 1)), obs_window], axis=0)
                obs_window = obs_window[-self.n_obs_steps:]  # 取最近 n_obs_steps 步
                
                # 动作窗口: [start_idx, start_idx + horizon)
                action_window = actions[start_idx : start_idx + self.horizon]
                # 当前观测（作为条件）
                current_obs = observations[start_idx]
                
                self.samples.append({
                    "obs_seq": obs_window.astype(np.float32),
                    "action_seq": action_window.astype(np.float32),
                    "current_obs": current_obs.astype(np.float32),
                })
        
        print(f"数据集构建完成: {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "obs_seq": torch.from_numpy(sample["obs_seq"]),
            "action_seq": torch.from_numpy(sample["action_seq"]),
            "current_obs": torch.from_numpy(sample["current_obs"]),
        }


def get_dataloader(episodes, config, shuffle=True):
    dataset = TrajectoryDataset(episodes, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )
    return dataloader