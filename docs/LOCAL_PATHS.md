# 本地路径配置

代码中的 `# LOCAL_PATH:` 标出需要按运行环境配置的位置，包括绝对路径、外部源码导入和依赖目录布局的输入文件。快速定位：

```bash
rg -n 'LOCAL_PATH:' code
```

这些标注说明路径用途，不会自动替换路径。`configs/` 的三个 JSON 只包含训练参数；模型、数据、解释器和运行文件的路径分布如下。

## 模型与 Python 环境

| 位置 | 需要配置的内容 |
| --- | --- |
| [family_backends_v1.py](../code/core/family_backends_v1.py) 的 `FAMILY_BINDINGS` | 每个模型的 `model_root`、`interpreter`，以及 NVILA 的 `reference_code_root` |
| [Molmo common.py](../code/training/molmo/common.py)、[NVILA common.py](../code/training/nvila/common.py)、[Qwen common.py](../code/training/qwen3/common.py) | `BENCHMARK_DIR`：完整数据目录；`BACKEND_SOURCE_DIR`：backend 源码；`WORKER_PYTHON`：RGB 解码解释器 |
| [benchmark_loader.py](../code/benchmark/benchmark_loader.py) | `ARIA_PYTHON`：安装 Aria 读取依赖的解释器；ADT 解码脚本位置 |
| [runner data.py](../code/core/mosder_fgr_runner_candidate_v3/data.py) | `FORMAL_RGB20_INTERPRETER`，以及 PyAV、Pillow 和像素读取策略文件的路径 |
| [seal_rgb20_bridge.py](../code/core/mosder_fgr_runner_candidate_v3/seal_rgb20_bridge.py) | 首行解释器路径、`BOUND_INTERPRETER` 和 `FORMAL_LOADER` |
| [分支消融 common.py](../code/ablations/branches/common.py)、[池化消融 common.py](../code/ablations/pooling_and_gate/common.py) | `BACKEND_ROOT` 和 worker 子进程使用的 Python 路径 |

模型加载会检查解释器和依赖版本。更换路径时，目标环境也需满足相应模型的依赖要求。

## 训练与下游 QA

| 位置 | 本地依赖 |
| --- | --- |
| [Molmo data_worker.py](../code/training/molmo/data_worker.py)、[NVILA data_worker.py](../code/training/nvila/data_worker.py) | 从 `BENCHMARK_DIR / "scripts"` 加载数据读取器；需与 `common.py` 的数据配置配套 |
| [NVILA qa_readout.py](../code/training/nvila/qa_readout.py)、[Qwen qa_readout.py](../code/training/qwen3/qa_readout.py) | `sys.path` 指定的 `qa_protocol` 源码目录，以及脚本旁的 `QA.json` |
| [Molmo QA train.py](../code/downstream/molmo_2048/train.py)、[NVILA QA train.py](../code/downstream/nvila_2048/train.py)、[NVILA 续训 train.py](../code/downstream/nvila_remaining3653/train.py) | `run_pilot` 和数据工具的加载路径；`PLAN.json` 中的 parent、grounder；QA / 来源样本清单及验证答案文件 |
| [grounder.py](../code/downstream/nvila_2048/grounder.py) | 路由训练配置 JSON、标注和视频输入 |
| [soft_query_backend.py](../code/downstream/grounding/soft_query_backend.py)、[nvila_soft_backend.py](../code/downstream/nvila_2048/nvila_soft_backend.py) | backend 和软路由模块所在的源码目录 |
| [run_pilot.py](../code/downstream/runtime_dependencies/run_pilot.py)、[runtime common.py](../code/downstream/runtime_dependencies/internal_grounding_physics_qa_v1/common.py) | 运行模块目录、模型源码、Python 解释器，以及由这些目录派生的数据与权重路径 |
| [adapter.py](../code/downstream/runtime_dependencies/omnivchall_source_event_diagnostic_v1/runtime/adapter.py) | `EXPERIMENT_DIR` 源码目录、`PARENT_PATH` 权重、`GROUND_PATH` 路由器和输入清单中的视频缓存路径 |
| [research_train.py](../code/downstream/runtime_dependencies/internal_source_research_20260908/research_train.py) | 输入行的 `rgb20_npz_path`；相邻目录中的运行模块与来源样本清单 |
| [score_val.py](../code/downstream/runtime_dependencies/omnivchall_val_eval_v1/score_val.py)、[score_candidate.py](../code/downstream/runtime_dependencies/internal_source_research_20260908/score_candidate.py) | 评分数据根目录、OmniVCHall 的真实/生成视频目录，以及固定位置的评分源码 |
| [contextual_difference.py](../code/downstream/runtime_dependencies/internal_source_research_20260908/contextual_difference.py)、[qa_update_controls.py](../code/downstream/runtime_dependencies/internal_grounding_physics_qa_v1/model/qa_update_controls.py) | backend 源码目录 |

`sys.path.insert(0, ...)` 和 `spec_from_file_location(...)` 会从指定位置加载源码。即使仓库有同名文件，也应确认入口实际加载的是预期文件。

## 数据构建与消融

| 位置 | 本地依赖 |
| --- | --- |
| [benchmark/](../code/benchmark/) 中的 `BENCHMARK_ROOT` | 多数脚本使用 `Path(__file__).resolve().parents[1]`，在当前结构下会指向 `code/`。运行前需配置完整 benchmark 根目录，其中应有 `release/`、`candidates/` 和相应媒体文件 |
| [build_v2.py](../code/benchmark/build_v2.py)、[select_v2.py](../code/benchmark/select_v2.py)、[finalize_v2.py](../code/benchmark/finalize_v2.py) 等 | 由 `.with_name(...)` 定位的相邻数据目录及处理脚本；配置根目录后仍需核对这些派生路径 |
| [scan_test_adt.py](../code/benchmark/scan_test_adt.py)、[scan_hot_test.py](../code/benchmark/scan_hot_test.py) | 扫描脚本、ADT / HOT3D 源媒体目录和物理运动判定配置 |
| [document_release.py](../code/benchmark/document_release.py)、[self_test.py](../code/benchmark/self_test.py) | 源数据目录和 RGB / 框插值验证工具 |
| [分支消融 prepare.py](../code/ablations/branches/prepare.py)、[池化消融 prepare.py](../code/ablations/pooling_and_gate/prepare.py) | 验证样本、完整模型预测、checkpoint 和数据 worker 的来源目录 |
| [source_pretraining/run.py](../code/baselines/source_pretraining/run.py)、[qa_source_rehearsal/train.py](../code/baselines/qa_source_rehearsal/train.py) | 数据工具、QA 协议、训练模块、初始化权重和验证答案文件 |

`data/benchmark/splits/` 是精简索引，不能直接替代上述脚本所需的完整 `release/*.jsonl`。

## 核心加载器的外部文件

| 位置 | 本地依赖 |
| --- | --- |
| [projection.py](../code/core/mosder_consumed_field_projection_v2/projection.py) | `REPOSITORY_ROOT`、`SANITIZED_ROOT`、`DEFAULT_OUTPUT_ROOT` 及派生的策略、清单和输入加载脚本 |
| [strict_v3_rgb20_bridge.py](../code/core/mosder_fgr_runner_candidate_v3/strict_v3_rgb20_bridge.py)、[rgb20_loader_worker.py](../code/core/mosder_fgr_runner_candidate_v3/rgb20_loader_worker.py) | `REPOSITORY`、外部读取器、训练/验证清单和 source registry |
| [runner.py](../code/core/mosder_fgr_runner_candidate_v3/runner.py) | 架构校验、数据投影和 RGB 读取器的 JSON 记录；命令行指定的运行文件 |
| [verify_architecture_selection_seal.py](../code/core/mosder_final_v1/verify_architecture_selection_seal.py) | 相对目录中的校验 JSON 与其引用文件；这些文件未全部包含在仓库中 |
| [QA runtime.py](../code/core/mosder_short_event_qa_pilot_v1/runtime.py) | 调用者提供的 `training_protocol_json`、`checkpoint_directory` 和 `training_completion_json` |

## JSON 与输入记录中的路径

JSON 字段说明放在同目录 README 中，文件本身保持标准 JSON。

| 文件 | 路径字段与用途 |
| --- | --- |
| [dataset_config.json](../data/benchmark/config/dataset_config.json) | `train_manifest`、`val_manifest`、`test_manifest`、`train_sampling_weights`、`loader`，共 5 个绝对路径；见[字段说明](../data/benchmark/config/README.md) |
| [physical_provenance.json](../data/benchmark/config/physical_provenance.json) | 四个记录的 `path` 指向来源核对材料；与各自 `sha256` 配套 |
| [train20_metadata.json](../data/benchmark/examples/train20_metadata.json) | 帧 `locator`、RGB 容器 `base_path`、标注路径等；见[metadata 路径说明](../data/benchmark/examples/README.md) |
| 运行目录中的 `PLAN.json` | 模型入口实际使用的运行配置，包含权重、路由器、输入文件与摘要；`configs/*.json` 不能直接替代完整运行配置 |
| 完整训练 / QA 输入记录 | `rgb20_npz_path`、`model_input.rgb20_frames[*].path`、`rgb20.formal_RGB_container.path` 等由加载器直接读取 |

搬移媒体时，`locator` 中 `#` 前面的部分是文件路径，后面的部分是归档成员或帧定位信息，需保留。只改变存储位置且内容未变时，内容哈希无需变化；替换文件内容则需核对对应摘要和输入约定。

包含源码摘要的运行配置也需与实际代码文件一致，注释同样计入文件 SHA-256。

## 脚本旁的运行文件

`RUN_DIR` / `EXPERIMENT_ROOT` 常指向脚本所在目录，输入和输出都按该目录解析：

- 主训练需要 `PLAN.json`；QA 转换需要 `QA.json`。
- 下游入口还需要 QA / 来源样本清单，文件名以各入口为准。
- 消融需要 `PLAN.json`、`val1200.jsonl`、完整模型预测和各组数据 worker。
- Qwen 的 `data_worker.py` 尚未提供，需要补齐。
- Molmo 预训练模型验证需要 `validation/raw_molmo/plan.json` 和主训练的完成文件。
- QA 辅助入口还会读取 `PILOT_PLAN.json`、`ROUND1_PROTOCOL.json`、`BASELINE_FREEZE.json` 及配套清单，具体位置见对应 `LOCAL_PATH:` 注释。
- checkpoint、状态与预测通常写在脚本目录下，运行目录需可写。

普通包内导入、自动创建的临时目录和训练日志不是需要统一替换的机器路径。
