# 训练与验证入口

这里提供三个模型的训练入口。阅读时可以先看任一目录的 `train.py`，再看
`common.py` 和 `evaluation.py`；三组代码的训练流程相同，最终模型选择方式不同。

| 目录 | 训练入口 | 验证后处理 |
| --- | --- | --- |
| `molmo/` | 固定加载 Molmo2-O-7B，依次执行 F/G/R | 比较 G、R 端点的验证指标；另提供预训练模型验证 |
| `nvila/` | 从 `PLAN.json` 读取模型家族与精度 | 使用 R 端点，G 端点作为阶段对照；将预测转换成 QA 答案 |
| `qwen3/` | 从 `PLAN.json` 读取模型家族与精度 | 与 NVILA 的端点及 QA 输出流程相同 |

## 文件职责

- `train.py`：核对运行配置与依赖摘要、进行四条训练样本的梯度预检、训练、保存和恢复检查点、调用验证。
- `common.py`：JSON/JSONL 读写、文件摘要、独立解码进程、数据集适配和固定顺序采样器。采样顺序由运行配置指定。
- `evaluation.py`：续接未完成的逐条预测，统计来源及状态指标，同时检查模型参数未被验证过程改变并恢复随机数状态。
- `data_worker.py`：在 benchmark 的 Python 环境中解码 RGB 输入。请求只包含 `source` 和 `model_input`。
- `qa_readout.py`（NVILA、Qwen3）：按预先固定的问题规范转换已保存预测，不再次调用模型。
- `molmo/raw_val.py`：等待主实验完成、GPU 空闲后，评估原始预训练 Molmo。
- `molmo/derive_metrics.py`：从已写入的训练 JSONL 生成 CSV，重复训练步以最新记录为准；直到启动器写入 `EXIT.json` 才停止轮询。

`data_worker.py` 和 `derive_metrics.py` 的常驻循环由 `main()` 启动；导入模块本身不会开始读取管道或轮询。

## 训练流程与输出

`train.py` 依次执行 F、G、R，每阶段 625 次优化更新，每次累积 8 条样本。
开始训练前会使用四种来源、四种状态的训练样本检查梯度，随后恢复方法参数和随机数状态。
G 端点在进入 R 前验证；R 完成后再次验证。Molmo 的端点选择按
`checkpoint_rank()` 的指标顺序比较，完全相同时保留较早的 G 端点。

运行目录就是各模型脚本所在的目录。常用输出包括：

| 文件 | 用途 |
| --- | --- |
| `PLAN.json` | 运行配置，包括数据与代码摘要、采样顺序及随机种子 |
| `checkpoints/*.pt`、`LATEST.json` | 方法参数、优化器、调度器、采样位置和随机数状态，以及最新检查点索引 |
| `preflight/receipt.json` | 预检结果；恢复运行时要求已经通过预检 |
| `*_metrics_*.jsonl`、`STATUS.json` | 训练更新记录和当前进度 |
| `validation/*/predictions.jsonl`、`metrics.json` | 验证预测和汇总指标 |
| `SELECTED.json`、`COMPLETION.json` | 选中的端点和完成记录 |

`python train.py --resume` 会读取 `LATEST.json` 并核对运行配置与检查点摘要。

## 运行依赖

运行前配置 `common.py` 中的 `BENCHMARK_DIR`（完整 benchmark）、
`BACKEND_SOURCE_DIR`（backend 源码目录）和 `WORKER_PYTHON`（解码环境）。
仓库中的 split 索引不能替代完整的 `release/*.jsonl` 和视频数据。

- 需提供完整的运行配置 `PLAN.json`、模型权重和所需检查点。
- QA 转换还依赖各运行目录的 `QA.json`，以及
  `/root/autodl-tmp/molmo_raw_motion_qa_v3/qa_protocol.py`。
- **Qwen3 的 `data_worker.py` 未提供**，需补齐与该模型匹配的解码 worker。
- `raw_val.py` 和 `derive_metrics.py` 需要启动器写入完成/退出文件；仓库未提供该启动器。
