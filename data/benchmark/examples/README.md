# Metadata 样例中的本地路径

`train20_metadata.json` 包含 20 条 metadata 样例，媒体与标注文件需另行提供。

| 字段 | 本地路径含义 |
| --- | --- |
| `model_input.rgb20.frames[*].locator` | RGB 文件或容器路径；`#` 后为帧或归档成员定位信息 |
| `model_input.rgb20.raw_rgb_container.base_path` | RGB 容器的本地位置 |
| `physical_annotation` 中的 `path` / `source_registry` | 相机与物体姿态标注、索引及核对材料 |
| `label_derivation`、`target_qualification` 中的路径 | 标签推导与目标筛选依据 |
| `trajectory_gt.raw_annotation_paths` | 原始轨迹标注文件 |

这些路径不会随仓库位置自动变化。配置实际数据时，保留帧编号、时间戳及归档成员定位信息。更多入口见[本地路径配置](../../../docs/LOCAL_PATHS.md)。
