# 数据集配置中的本地路径

`dataset_config.json` 中以下字段为绝对路径，使用前需配置为本机位置：

| 字段 | 指向内容 |
| --- | --- |
| `train_manifest` | 完整训练 release JSONL |
| `val_manifest` | 完整验证 release JSONL |
| `test_manifest` | 完整测试 release JSONL |
| `train_sampling_weights` | 训练采样权重 JSONL |
| `loader` | 提供 `load_input()` 的 Python 文件 |

`physical_provenance.json` 中四个记录的 `path` 是来源核对文件的位置，每个路径都有对应的 `sha256`。JSON 中的路径只是定位信息，文件需另行提供。

`sampling_policy.json` 和 `train_sampling_weights.jsonl` 不包含本地绝对路径。其他代码位置见[本地路径配置](../../../docs/LOCAL_PATHS.md)。
