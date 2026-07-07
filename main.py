# main.py
import argparse
from train_diffusion_policy import train_diffusion_policy
from test_diffusion_policy import test_diffusion_policy
from config import Config
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Diffusion Policy for Inverted Pendulum")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="运行模式: train (训练) 或 test (测试)")
    parser.add_argument("--model_path", type=str, default="checkpoints/diffusion_policy_final.pt",
                        help="测试时加载的模型路径")
    parser.add_argument("--episodes", type=int, default=1,
                        help="测试回合数")
    args = parser.parse_args()
    
    config = Config()
    
    if args.mode == "train":
        train_diffusion_policy(config)
    elif args.mode == "test":
        test_diffusion_policy(model_path="checkpoints/diffusion_policy_final.pt", 
                              num_episodes=args.episodes,
                              render=True)
                                  
    else:
        print(f"未知模式: {args.mode}")


if __name__ == "__main__":
    main()
    