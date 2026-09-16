# MoSDeR

MoSDeR 在冻结的视频语言模型上学习相机与目标物体的运动来源，通过视觉特征注入和 Camera / Object / Shared 三条 LoRA 分支辅助回答问题。项目包含模型实现、训练与评测代码、数据索引和实验结果，支持 Molmo2、NVILA 和 Qwen3-VL 三种模型接口。

## 阅读顺序

1. [方法说明](docs/method_context.md)：输入、张量形状、F/G/R 损失，以及自建任务和下游 QA 的区别。
2. [核心代码](code/core/README.md)：来源适配器、视觉注入和 TriLoRA。
3. [训练入口](code/training/README.md)：三种模型的训练与验证流程。
4. [下游 QA](code/downstream/README.md)：问题条件软路由、QA 更新和来源复习。
5. [消融结果](docs/ablation_conclusions.md)：组件干预的结果与分析。

## 项目结构

| 目录 | 内容 |
| --- | --- |
| [code/core/](code/core/README.md) | 模型、损失函数和公共训练组件 |
| [code/training/](code/training/README.md) | 自建 benchmark 的 F → G → R 训练与评测 |
| [code/downstream/](code/downstream/README.md) | OmniVCHall QA、软目标路由及来源复习 |
| [code/benchmark/](code/benchmark/README.md) | 数据构建、RGB 读取、物理标签和划分检查 |
| [code/baselines/](code/baselines/README.md) | 普通 LoRA 及 QA＋来源复习对照 |
| [code/ablations/](code/ablations/README.md) | 池化、门控与分支推理干预 |
| [configs/](configs/README.md) | 下游训练参数 |
| [data/benchmark/](data/benchmark/) | split 索引、采样权重、统计及 metadata 样例 |
| [data/omnivchall/](data/omnivchall/) | QA 样本 ID 和数据划分 |
| [data/paper_tables/](data/paper_tables/) | 参数量和数据分布 |
| [results/](results/) | 评测指标、训练曲线和消融结果 |
| [tests/](tests/) | CPU 回归检查 |

## 数据与评测

自建 benchmark 包含 train 5000 / val 1200 / test 1200，各 split 的四种运动状态数量相等。`data/benchmark/splits/` 提供划分索引；`examples/train20_metadata.json` 提供按固定顺序选取的 20 条完整 metadata 样例，每种状态 5 条。这些文件用于了解数据结构，完整训练还需要视频和全量 metadata。

消融使用固定 Molmo 权重进行推理干预。结果同时报告状态准确率和解析失败数，具体见[消融分析](docs/ablation_conclusions.md)。

## 环境与检查

使用 Python 3.10 或更新版本。在已有 NumPy、PyTorch 的环境中运行：

```bash
python -m pip install -r requirements-dev.txt
python -m ruff format --check code tests
python -m ruff check code tests
python -m pytest -q
```

完整训练需要另行准备模型权重、视频数据、模型依赖和运行配置，并设置相应路径。需要适配的位置见[本地路径配置](docs/LOCAL_PATHS.md)，代码中以 `LOCAL_PATH:` 标注；环境要求和代码规范见[开发说明](docs/DEVELOPMENT.md)。
