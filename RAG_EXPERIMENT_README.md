# CoderEval RAG 检索实验指南

本文档描述了使用稀疏检索方法 (BM25 / Jaccard) 进行代码生成上下文检索实验的完整流程。

## 📋 实验概述

### 目标

评估不同稀疏检索方法在不同上下文长度下对代码生成模型性能的影响。

### 检索方法

1. **BM25** - 基于词频-逆文档频率的经典检索算法
2. **Jaccard** - 基于集合相似度的检索算法

### 上下文长度

`1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k, 192k` tokens

## 🗂️ 文件结构

```
CoderEval Docker/
├── sparse_retrieval_context.py    # 稀疏检索上下文生成
├── rag_inference.py               # 模型推理
├── attention_analysis.py          # 注意力分析
├── rag_result_analysis.py         # 结果分析与可视化
│
├── rag_contexts/                  # 生成的上下文数据
│   ├── rag_bm25_1024tokens.jsonl
│   ├── rag_bm25_2048tokens.jsonl
│   ├── ...
│   ├── rag_jaccard_1024tokens.jsonl
│   ├── ...
│   └── metadata_*.json
│
├── rag_inference_results/         # 推理结果
│   ├── results_bm25_1024tokens.jsonl
│   ├── ...
│   └── inference_summary_*.json
│
├── attention_data/                # 注意力数据
│   └── attention_*.json
│
├── attention_analysis_output/     # 注意力分析结果
│   ├── attention_entropy_*.png
│   └── ...
│
└── rag_analysis_output/           # 最终分析结果
    ├── comprehensive_analysis.png # 综合大图
    ├── pass1_bm25.png             # 小图
    ├── pass1_jaccard.png
    ├── method_comparison.png
    └── ...
```

## 🚀 实验流程

### 第一步：生成稀疏检索上下文

在**本地或有 repos 目录的环境**执行：

```bash
# BM25 方法
python sparse_retrieval_context.py \
    --method bm25 \
    --output ./rag_contexts \
    --dataset home/travis/builds/CoderEval4Python.json \
    --repos home/travis/builds/repos \
    --context-lengths 1024 2048 4096 8192 16384 32768 65536 131072 196608

# Jaccard 方法
python sparse_retrieval_context.py \
    --method jaccard \
    --output ./rag_contexts \
    --dataset home/travis/builds/CoderEval4Python.json \
    --repos home/travis/builds/repos \
    --context-lengths 1024 2048 4096 8192 16384 32768 65536 131072 196608
```

**输出文件**：
- `rag_contexts/rag_bm25_*tokens.jsonl`
- `rag_contexts/rag_jaccard_*tokens.jsonl`
- `rag_contexts/metadata_*.json`

### 第二步：模型推理

支持三种推理后端：**vLLM（推荐）**、Transformers、API

#### 方式一：vLLM 推理（推荐，高效）

```bash
# BM25 上下文推理
python rag_inference.py \
    --input ./rag_contexts \
    --output ./rag_inference_results \
    --method bm25 \
    --backend vllm \
    --model-path /path/to/Qwen3-4B-Instruct-2507 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --batch-size 8 \
    --all-lengths \
    --num-samples 10

# Jaccard 上下文推理
python rag_inference.py \
    --input ./rag_contexts \
    --output ./rag_inference_results \
    --method jaccard \
    --backend vllm \
    --model-path /path/to/Qwen3-4B-Instruct-2507 \
    --all-lengths \
    --num-samples 10
```

**vLLM 特定参数**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--tensor-parallel-size` | GPU 并行数 | 1 |
| `--gpu-memory-utilization` | GPU 内存使用率 | 0.9 |
| `--max-model-len` | 模型最大上下文长度 | 自动 |
| `--batch-size` | 批处理大小 | 8 |

#### 方式二：Transformers 推理

```bash
python rag_inference.py \
    --input ./rag_contexts \
    --method bm25 \
    --backend transformers \
    --model-path /path/to/Qwen3-4B-Instruct-2507 \
    --all-lengths \
    --save-attention  # 可选：保存注意力数据
```

#### 方式三：API 推理（vLLM 服务器或 OpenAI API）

首先启动 vLLM 服务器：
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen3-4B-Instruct-2507 \
    --port 8000
```

然后运行推理：
```bash
python rag_inference.py \
    --input ./rag_contexts \
    --method bm25 \
    --backend api \
    --api-url http://localhost:8000/v1 \
    --model-name Qwen3-4B-Instruct-2507 \
    --all-lengths
```

### 第三步：注意力分析（可选）

分析模型对不同上下文区域的注意力分布：

```bash
python attention_analysis.py \
    --attention-dir ./attention_data \
    --rag-dir ./rag_contexts \
    --dataset home/travis/builds/CoderEval4Python.json \
    --output ./attention_analysis_output \
    --method bm25 \
    --dpi 400
```

**输出图表**：
- `attention_entropy_*.png` - 注意力熵随上下文长度变化
- `attention_distribution_*.png` - 注意力在不同区域的分布
- `oracle_relevance_*.png` - 检索内容与 oracle 的相关性
- `attention_summary_*.png` - 综合分析图

### 第四步：结果分析与可视化

```bash
python rag_result_analysis.py \
    --results-dir ./rag_inference_results \
    --rag-dir ./rag_contexts \
    --dataset home/travis/builds/CoderEval4Python.json \
    --output ./rag_analysis_output \
    --methods bm25 jaccard \
    --dpi 400
```

## 📊 输出图表说明

### 综合大图 (`comprehensive_analysis.png`)

包含 6 个子图：

| 子图 | 内容 |
|------|------|
| (a) | BM25 vs Jaccard pass@1 柱状图对比 |
| (b) | pass@k 趋势折线图 |
| (c) | BM25 详细 pass@1/5/10 |
| (d) | Jaccard 详细 pass@1/5/10 |
| (e) | 性能热力图 |
| (f) | 方法间差异对比 |

### 小图

| 文件名 | 内容 |
|--------|------|
| `pass1_bm25.png` | BM25 pass@1 折线图 |
| `pass1_jaccard.png` | Jaccard pass@1 折线图 |
| `pass_k_bm25.png` | BM25 pass@1/5/10 对比 |
| `pass_k_jaccard.png` | Jaccard pass@1/5/10 对比 |
| `method_comparison.png` | 方法对比柱状图 |
| `trend_comparison.png` | 趋势对比折线图 |
| `context_*.png` | 各上下文长度的详细对比 |

## ⚙️ 参数说明

### sparse_retrieval_context.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--method` | 检索方法 (bm25/jaccard) | bm25 |
| `--output` | 输出目录 | ./rag_contexts |
| `--context-lengths` | 目标上下文长度列表 | 1024 - 196608 |

### rag_inference.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--method` | 检索方法 | bm25 |
| `--all-lengths` | 处理所有长度 | false |
| `--num-samples` | 每任务生成代码数 | 10 |
| `--save-attention` | 保存注意力 | false |
| `--temperature` | 生成温度 | 0.2 |

### rag_result_analysis.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--methods` | 要分析的方法 | bm25 jaccard |
| `--dpi` | 图片分辨率 | 400 |
| `--skip-analysis` | 跳过分析直接画图 | false |

## 📈 评估指标

### Pass@k

使用 HumanEval 论文中的无偏估计公式：

$$\text{pass@}k = \mathbb{E}_{\text{Problems}} \left[ 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \right]$$

其中：
- $n$ = 每个任务生成的候选代码数量
- $c$ = 通过测试的代码数量
- $k$ = 选取的候选数

### 注意力分析指标

- **Attention Entropy**：衡量注意力分布的分散程度
- **Region Distribution**：注意力在检索内容/目标函数区域的分布
- **Oracle Relevance**：检索内容与标准答案相关代码的重叠度

## 🔧 依赖安装

```bash
# 基础依赖
pip install numpy matplotlib requests

# vLLM 后端（推荐）
pip install vllm

# Transformers 后端
pip install transformers torch
```

**vLLM 安装注意事项**：
- 需要 CUDA 11.8+ 或 12.x
- 推荐 Python 3.9+
- 详见 [vLLM 官方文档](https://docs.vllm.ai/en/latest/getting_started/installation.html)

## 📝 注意事项

1. **上下文生成**需要完整的 CoderEval repos 目录
2. **模型推理**需要 GPU 环境（推荐）
3. **注意力分析**需要在推理时开启 `--save-attention`
4. 确保 `--num-samples` 参数在所有步骤中保持一致

## 🔄 快速开始（完整流程）

```bash
# 1. 生成上下文（本地执行）
python sparse_retrieval_context.py --method bm25 --output ./rag_contexts
python sparse_retrieval_context.py --method jaccard --output ./rag_contexts

# 2. 使用 vLLM 推理（云端/GPU 环境）
python rag_inference.py \
    --method bm25 \
    --backend vllm \
    --model-path /path/to/Qwen3-4B-Instruct-2507 \
    --tensor-parallel-size 1 \
    --batch-size 8 \
    --all-lengths

python rag_inference.py \
    --method jaccard \
    --backend vllm \
    --model-path /path/to/Qwen3-4B-Instruct-2507 \
    --all-lengths

# 3. 分析与可视化
python rag_result_analysis.py --methods bm25 jaccard --dpi 400
```

### 单 GPU 推理示例

```bash
# 192k 长上下文可能需要更大显存
python rag_inference.py \
    --method bm25 \
    --backend vllm \
    --model-path /path/to/Qwen3-4B-Instruct-2507 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 200000 \
    --context-length 196608
```

### 多 GPU 推理示例

```bash
# 使用 4 张 GPU 进行 tensor 并行
python rag_inference.py \
    --method bm25 \
    --backend vllm \
    --model-path /path/to/Qwen3-4B-Instruct-2507 \
    --tensor-parallel-size 4 \
    --batch-size 16 \
    --all-lengths
```

## 📊 预期结果

实验应该能够展示：

1. **上下文长度影响**：随着上下文长度增加，pass@k 的变化趋势
2. **检索方法对比**：BM25 vs Jaccard 在不同场景下的性能差异
3. **注意力分布**：模型是否正确关注了检索到的相关代码
4. **最优配置**：确定最佳的上下文长度和检索方法组合

