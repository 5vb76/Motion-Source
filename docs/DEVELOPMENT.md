# 开发与维护

## 代码约定

- Python 使用 4 空格缩进、88 字符目标行宽、双引号，由 `pyproject.toml` 中的 Ruff 配置统一排版。
- 训练中的前向、反传、梯度检查和持久化分别展开。
- 路径使用 `*_DIR` / `*_PATH`；模型实例、运行器和优化器分别使用 `backend`、`runtime`、`optimizer`。
- I/O 函数写明格式，例如 `read_jsonl`、`write_json`、`append_jsonl`、`file_sha256`。
- 注释说明输入形状、计算约束或实现原因。Camera/Object、LoRA、F/G/R 等方法术语保持一致。
- 修改模型参数名、`state_dict` 键、JSON 字段和公共接口时，检查相应调用者与保存文件。

## 检查命令

```bash
python -m ruff format code tests
python -m ruff check code tests
python -m pytest -q
```

CPU 测试覆盖张量行为、梯度隔离、答案解析和采样恢复。部分脚本在导入时会读取文件或写出结果，语法检查应使用 AST 或编译检查。

## 训练环境

路径位置、JSON 字段和目录布局要求见[本地路径配置](LOCAL_PATHS.md)。代码中的 `LOCAL_PATH:` 注释标明各处用途。

各模型的解释器、模型目录和依赖版本由 `code/core/family_backends_v1.py` 的 `FAMILY_BINDINGS` 定义。模型可能需要不同版本的 Transformers、PEFT 和专用源码。

运行前需要准备：

1. 原始视频与完整 release metadata；`data/benchmark/splits/` 只提供索引。
2. 模型权重、模型运行环境和 RGB 解码环境。
3. 训练目录中的 `PLAN.json` 运行配置、采样顺序，以及相应的初始化 checkpoint 和软路由器权重。
4. `common.py`、数据 worker 和动态导入所使用的路径。

Qwen 训练所需的 `data_worker.py` 尚未包含，需要补齐后才能运行。下游评分器会校验源码摘要，运行配置中的文件校验值需与实际依赖一致。
