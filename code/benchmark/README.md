# Benchmark 构建与评估

这里保存 RGB20 物理运动基准 V2 的数据处理脚本。每条输入含 20 帧 RGB、目标框、时间戳和统一问题；运动状态及物理轨迹只供监督和审计使用，不进入 `load_input()` 返回的模型输入。

## 先读这几个文件

| 文件 | 作用 |
| --- | --- |
| `benchmark_loader.py` | `load_input()` 解码并校验输入；`interpolate_boxes()` 对稀疏目标框做插值。 |
| `evaluate.py` | 校验预测覆盖范围，计算状态、来源、域等指标以及分组 bootstrap 区间。 |
| `build_v2.py` | 将各来源样本转成统一记录，并导出 ADT/HOT3D 原生输入。 |
| `validate_release.py` | 检查清单、轨迹、输入、采样权重，并可抽样解码。 |
| `self_test.py` | 检查指标、框插值、GT 隔离和内容哈希校验。 |

## 数据处理顺序

1. `scan_test_adt.py` 扫描 ADT 物理窗口；`adt_native_index.py` 记录原生 RGB 时间戳和标定；`qualify_adt.py` 筛出可见目标框覆盖完整的窗口。
2. `scan_hot_test.py` 扫描 HOT3D 窗口；`select_v2.py` 按配额选择样本，并按采集组留出 AV2/TACO 测试数据。
3. `hash_adt_pixels.py` 记录所选 ADT 帧的像素哈希；`build_v2.py` 写出统一的训练、验证和测试清单。
4. `validate_release.py`、`audit_splits.py`、`self_test.py` 分别检查记录、跨集合重叠和加载行为。
5. `document_release.py` 写入配置与来源说明；`finalize_v2.py` 在验证收据通过后生成发布清单。

## 路径与运行边界

`BENCHMARK_ROOT` 是脚本目录的上一级；部分处理函数从相邻的 `general_motion_benchmark_v1/scripts` 加载。运行前需准备这些依赖、完整媒体数据和发布清单，并配置代码中的绝对路径。需要 Aria 的脚本还依赖 `projectaria_tools` 及相应解码环境。

`evaluate.py` 可以显式指定输入输出，避免依赖默认数据目录：

```bash
python code/benchmark/evaluate.py \
  --manifest /path/to/release/test.jsonl \
  --predictions /path/to/predictions.jsonl \
  --output /path/to/metrics.json
```

预测文件每行需含 `case_id` 和 `state`；状态取值为 `neither`、`camera_only`、`object_only`、`both`。它必须恰好覆盖请求的清单。

构建、验证和发布入口均会写入收据或数据文件，应在配置好的 benchmark 目录执行。`finalize_v2.py` 生成发布文件的 `SHA256SUMS`。
