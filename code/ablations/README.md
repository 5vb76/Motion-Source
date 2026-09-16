# 推理干预实验

这两个实验都加载同一个已训练 checkpoint，冻结权重后改变推理计算。它们用于衡量已训练模型对组件的依赖，不等同于删除组件后重新训练的结构消融。

| 目录 | 对照组与干预组 |
| --- | --- |
| `branches/` | `full` 保留完整模型；`no_camera` / `no_object` 将相应来源适配器、显式残差及 LoRA B 的输出置零。 |
| `pooling_and_gate/` | `full` 保留完整模型；`global_evidence` 共用全局池化证据；`constant_gate` 使用与视频内容无关的门控输入。 |

两个目录采用相同的文件分工：

- `prepare.py`：准备样本与基线预测，写入运行配置 `PLAN.json` 并创建数据 worker 链接。
- `run.py`：校验 checkpoint、基线参数哈希及样本顺序；重放四个状态的基线样本后运行各组干预。
- `interventions.py`：安装运行时 hook 或替换方法，不改参数。
- `common.py`：JSON 收据读写、独立 RGB 解码进程、样本包装。
- `validate_function.py`：按样本顺序恢复预测，严格解析四行回答，计算指标并检查权重未变。
- `summarize_function.py`：读取各组结果，生成 `RESULTS.json` / `RESULTS.md`，使用固定种子的来源分层、采集组 bootstrap。

每组结果位于 `<arm>/val1200/`，完成和失败状态分别写入相应 JSON 文件。

运行前需准备模型环境、数据、checkpoint 和 `data_worker.py`，并配置 `prepare.py` 中的来源路径。执行准备脚本生成 `PLAN.json` 后，再运行各组干预。
