# diffusion_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SinusoidalPositionalEmbedding(nn.Module):
    """时间步t的正弦位置编码"""
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, t):
        device = t.device
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb


class ConditionalUNet1D(nn.Module):
    """
    条件UNet1D：扩散策略的去噪网络核心
    输入: [B, T, D] 加噪后的动作序列
    条件: [B, obs_dim] 观测（状态）
    时间步: [B] 扩散步数
    输出: [B, T, D] 预测的噪声
    """
    
    def __init__(self, action_dim, horizon, obs_dim, d_model=128):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.d_model = d_model
        
        # 时间步编码
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        # 观测条件编码（Film Conditioning）
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        # 输入投影：[action_dim] -> [d_model]
        self.input_proj = nn.Linear(action_dim, d_model)
        
        # 下采样层（编码器）
        self.down1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.down2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2)
        self.down3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2)
        
        # 中间层
        self.mid = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        
        # 上采样层（解码器）
        self.up1 = nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        
        # 输出投影
        self.output_proj = nn.Linear(d_model, action_dim)
        
    def forward(self, noisy_action, obs, timestep):
        """
        noisy_action: [B, horizon, action_dim] 加噪后的动作序列
        obs: [B, obs_dim] 当前观测
        timestep: [B] 扩散步数
        """
        B, T, _ = noisy_action.shape
        
        # 时间编码
        t_emb = self.time_mlp(timestep)  # [B, d_model]
        
        # 观测编码
        obs_emb = self.obs_encoder(obs)  # [B, d_model]
        
        # 组合条件（Film Conditioning: 条件特征加到特征图上）
        cond_emb = t_emb + obs_emb  # [B, d_model]
        
        # 输入投影
        x = self.input_proj(noisy_action)  # [B, T, d_model]
        x = x.permute(0, 2, 1)  # [B, d_model, T] (Conv1D需要通道维度在前)
        
        # 扩展条件到空间维度
        cond_emb_expanded = cond_emb.unsqueeze(-1)  # [B, d_model, 1]
        
        # 下采样（添加条件到每层）
        x = x + cond_emb_expanded
        x = F.silu(self.down1(x))
        x = F.silu(self.down2(x))
        x = F.silu(self.down3(x))
        
        # 中间
        x = F.silu(self.mid(x))
        
        # 上采样
        x = F.silu(self.up1(x))
        x = F.silu(self.up2(x))
        x = F.silu(self.up3(x))
        
        # 裁剪回原始长度
        x = x[:, :, :T]
        
        # 输出投影
        x = x.permute(0, 2, 1)  # [B, T, d_model]
        pred_noise = self.output_proj(x)  # [B, T, action_dim]
        
        return pred_noise


class DiffusionPolicy(nn.Module):
    """扩散策略模型，封装扩散过程（前向加噪 + 反向去噪生成）"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.horizon = config.horizon
        self.action_dim = config.action_dim
        self.obs_dim = config.obs_dim
        self.num_diffusion_steps = config.num_diffusion_steps
        
        # 噪声调度 (线性beta schedule)
        beta = torch.linspace(config.beta_start, config.beta_end, config.num_diffusion_steps)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", 1 - beta)
        self.register_buffer("alpha_cumprod", torch.cumprod(self.alpha, dim=0))
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(self.alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1 - self.alpha_cumprod))
        
        # 去噪网络
        if config.model_type == "unet":
            self.denoise_net = ConditionalUNet1D(
                action_dim=config.action_dim,
                horizon=config.horizon,
                obs_dim=config.obs_dim,
                d_model=config.d_model,
            )
        else:
            # Transformer实现（可选扩展）
            self.denoise_net = self._build_transformer(config)
    
    def _build_transformer(self, config):
        """构建Transformer去噪网络（简化版，需扩展）"""
        # 此处可根据需要实现Transformer版本的去噪网络
        # 为简化，暂时使用UNet
        return ConditionalUNet1D(
            action_dim=config.action_dim,
            horizon=config.horizon,
            obs_dim=config.obs_dim,
            d_model=config.d_model,
        )
    
    def q_sample(self, x0, t, noise=None):
        """
        前向扩散: 在原始动作 x0 上加噪到时间步 t
        x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1)
        x_t = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t
    
    def p_sample(self, noisy_action, obs, timestep):
        """
        单步反向去噪: 从 x_t 预测 x_{t-1}
        """
        pred_noise = self.denoise_net(noisy_action, obs, timestep)
        
        # 计算去噪后的均值
        beta_t = self.beta[timestep].view(-1, 1, 1)
        alpha_cumprod_t = self.alpha_cumprod[timestep].view(-1, 1, 1)
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[timestep].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[timestep].view(-1, 1, 1)
        
        # x_{t-1} 的预测
        pred_x0 = (noisy_action - sqrt_one_minus_alpha_cumprod_t * pred_noise) / sqrt_alpha_cumprod_t
        
        # 方差调度 (使用beta_t作为方差)
        posterior_var = beta_t * (1 - alpha_cumprod_t / self.alpha_cumprod[timestep]) / (1 - alpha_cumprod_t)
        
        # 采样 x_{t-1}
        if timestep[0] > 0:
            noise = torch.randn_like(noisy_action)
            x_prev = pred_x0 + torch.sqrt(posterior_var) * noise
        else:
            x_prev = pred_x0
        
        return x_prev, pred_noise
    
    def generate_action(self, obs, num_denoise_steps=None):
        """
        从纯噪声生成动作序列（去噪生成）
        obs: [B, obs_dim] 或 [obs_dim]
        num_denoise_steps: 去噪步数，默认为完整扩散步数
        返回: [B, horizon, action_dim]
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # [1, obs_dim]
        
        B = obs.shape[0]
        
        if num_denoise_steps is None:
            num_denoise_steps = self.num_diffusion_steps
        
        # 从纯噪声开始
        x_t = torch.randn(B, self.horizon, self.action_dim, device=obs.device)
        
        # 逐步去噪
        timesteps = torch.full((B,), self.num_diffusion_steps - 1, dtype=torch.long, device=obs.device)
        for t in reversed(range(self.num_diffusion_steps - num_denoise_steps, self.num_diffusion_steps)):
            timesteps.fill_(t)
            with torch.no_grad():
                x_t, _ = self.p_sample(x_t, obs, timesteps)
        
        return x_t
    
    def forward(self, noisy_action, obs, timestep):
        """训练时的前向传播：预测噪声"""
        return self.denoise_net(noisy_action, obs, timestep)