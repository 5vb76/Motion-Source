# 普通 LoRA 与 QA rehearsal 基线

| 入口 | 实验流程 |
| --- | --- |
| `source_pretraining/run.py` | 先评估 MoSDeR，再从原始 Molmo 加载普通 rank-32 LoRA，用 5,000 条来源运动样本分两段训练，输出对应 QA 读数。 |
| `qa_source_rehearsal/train.py` | 冻结来源预训练的 LoRA，在其上增加 QA 更新残差；2,048 条 QA 样本中每四条插入一次来源样本，共使用 512 条来源样本。 |

两个入口都记录可恢复的 optimizer/RNG 状态、训练曲线及逐条预测，并检查冻结参数未改变。

先读各文件的 `main()` 理解实验顺序，再读 `evaluate()` 和 `save_checkpoint()`。QA 请求与指标辅助函数放在训练流程之前；`EXPERIMENT_ROOT` 表示运行目录，`PLAN` 保存运行配置，`training_utils` 提供公共训练工具。

启动前需准备运行配置、媒体、模型及依赖模块，并配置代码中的绝对路径。QA rehearsal 的 `base` 模块从 `/root/autodl-tmp/molmo_fourfield_lora_qa_v1` 加载，需一并提供或调整加载路径。
