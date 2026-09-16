# 下游问答训练与软路由

这里保存 MoSDeR 接入 OmniVCHall 问答任务的代码：先用问题文本定位视觉 token，再训练运动来源模块，最后评估固定训练终点。主干模型和问答训练中的定位路由器保持冻结。

## 从哪里开始读

| 位置 | 职责 |
| --- | --- |
| `grounding/soft_query_router.py` | 问题与视觉 token 匹配、目标/上下文池化、稀疏框监督损失；纯张量实现，适合先读。 |
| `grounding/soft_query_backend.py` | 把路由结果接入 Molmo，按空间权重回注相机与目标残差。 |
| `nvila_2048/nvila_soft_backend.py` | 复用上述接入逻辑，适配 NVILA 的语言模型和 tokenizer。 |
| `nvila_2048/grounder.py` | 缓存冻结 NVILA 特征，训练定位路由器，并报告拟合/留出集损失。 |
| `molmo_2048/train.py` | Molmo 的 2,048 条 QA + 512 条来源样本训练，256 次更新。 |
| `nvila_2048/train.py` | NVILA 对应的 2,048 条 QA 训练；原生损失使用 float16 autocast。 |
| `nvila_remaining3653/train.py` | 从 NVILA 的 256 步 checkpoint 继续训练剩余 3,653 条 QA，457 次更新。 |
| `runtime_dependencies/` | 上述入口复用的输入校验、训练、控制实验和评分代码。 |

建议按“路由器 → backend 接入 → 一个训练入口 → 评分器”的顺序阅读。三个训练入口保留独立文件，因为模型精度、恢复来源、学习率日程和评估面板有所不同。

## 一条样本如何流转

1. 输入校验器读取 20 帧原始 RGB、时间戳和原始问题，核对归档与像素哈希。
2. backend 提取原生视觉 token，并用冻结词嵌入计算问题向量。答案与选项不进入问题向量。
3. `SoftQueryRouter` 为每个空间 token 计算目标权重，并将目标与上下文分别池化为 10 个时间单元。
4. MoSDeR 计算运动来源特征，backend 按目标/上下文权重回注残差，原生语言模型计算答案损失或生成答案。
5. 训练入口记录日志、状态和 checkpoint；评估按清单顺序写入预测，解析失败仍计入总样本数。

来源复习样本训练时，`source_training_request()` 暂时用训练框覆盖空间池化，并在 `finally` 中恢复路由器。正常问答与推理仍使用问题驱动的软路由。

## 运行依赖

运行前需准备模型、RGB 归档与样本清单，并配置代码中的绝对路径。

- `PLAN.json` 指定 parent checkpoint、grounder、学习率、随机种子和输入哈希；入口还读取同目录中的 QA 与来源样本清单。
- 配置 `run_pilot.py`、共享数据 worker、原生模型 backend 及训练/验证数据的加载路径。
- 训练目录写入 `STATUS.json`、`TRAIN.jsonl`、`LATEST.json` 和 checkpoint。使用 `--resume` 恢复运行时会核对运行配置、checkpoint 和 RNG 状态。
- 来源样本清单及数量由具体训练入口和运行配置决定。

`score_candidate.py` 会校验所加载评分文件的 SHA-256；评分文件需与代码中指定的摘要一致。

## 依赖代码索引

| 文件 | 主要内容 |
| --- | --- |
| `runtime_dependencies/run_pilot.py` | 三种更新方案、三个随机种子的短训练对照，以及冻结/设种子辅助函数。 |
| `runtime_dependencies/internal_grounding_physics_qa_v1/common.py` | 输入构造、parent 加载、文件与参数校验。 |
| `runtime_dependencies/internal_grounding_physics_qa_v1/model/qa_update_controls.py` | 同一 parent 上附加普通 LoRA 更新的控制实验。 |
| `runtime_dependencies/internal_source_research_20260908/research_train.py` | QA 与来源样本联合训练、checkpoint 恢复、原生答案损失。 |
| `runtime_dependencies/internal_source_research_20260908/contextual_difference.py` | 从目标时间差分中减去上下文差分的可选研究分支。 |
| `runtime_dependencies/internal_source_research_20260908/source_pairs_common.py` | 来源配对诊断的输入与标签契约。 |
| `runtime_dependencies/omnivchall_source_event_diagnostic_v1/runtime/adapter.py` | 无框输入验证、冻结 backend 构造、运动来源预测。 |
| `runtime_dependencies/omnivchall_val_eval_v1/score_val.py` | 答案解析、固定分母准确率、分类指标。 |
| `runtime_dependencies/internal_source_research_20260908/score_candidate.py` | 预测身份校验、候选结果对比、按视频分组的 bootstrap 区间。 |
