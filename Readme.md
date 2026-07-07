# Diffusion Policy for Inverted Pendulum

本项目实现了基于扩散策略（Diffusion Policy）的倒立摆控制。模型能够根据当前状态（角度、角速度）生成未来的动作序列（扭矩），并通过滚动时域控制完成摆动与稳定任务。

```

## 环境要求

- Python 3.8+
- PyTorch
- NumPy
- Gym（倒立摆环境，如 `Pendulum-v1` 或自定义环境）
- Matplotlib（可选，用于可视化）

建议使用以下命令安装依赖：

```bash
pip install torch numpy gym matplotlib
```

## 使用方法

### 1. 训练模型

首先运行训练模式，模型将学习如何根据状态生成动作序列。训练完成后，模型权重将自动保存到 `checkpoints/diffusion_policy_final.pt`。

```bash
python main.py --mode train
```

训练过程中会输出 loss 曲线（如果配置了日志），训练结束后模型文件会保存在指定路径。

### 2. 测试模型

训练完成后，使用测试模式加载已保存的模型，并在倒立摆环境中进行可视化测试。

```bash
python main.py --mode test
```

默认会运行 **1 个测试回合**，并打开图形界面（`render=True`）展示控制效果。你可以通过参数调整测试回合数或指定不同的模型路径。

## 可选参数

| 参数             | 类型                 | 默认值                                    | 描述                     |
| ---------------- | -------------------- | ----------------------------------------- | ------------------------ |
| `--mode`       | `train` / `test` | `train`                                 | 运行模式：训练或测试     |
| `--model_path` | str                  | `checkpoints/diffusion_policy_final.pt` | 测试时加载的模型权重路径 |
| `--episodes`   | int                  | `1`                                     | 测试时运行的回合数       |

### 示例

- 训练模型（使用默认配置）：

  ```bash
  python main.py --mode train
  ```
- 测试模型（使用默认模型，运行 5 个回合）：

  ```bash
  python main.py --mode test --episodes 5
  ```
- 测试自定义检查点：

  ```bash
  python main.py --mode test --model_path checkpoints/my_model.pt --episodes 10
  ```
