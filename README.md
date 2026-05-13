# RAG_PROJECT

# Financial-RAG: 智能财务报告分析系统

这是一个专门针对企业财务报告（PDF转Markdown）设计的检索增强生成（RAG）系统。系统通过 **“摘要索引，原文召回”** 的架构，深度优化了财务报表中复杂的 HTML 表格处理及多文档对比分析能力。

## 核心功能

- **摘要索引，原文召回 (Summary Indexing, Full Text Retrieval)**：
    - **痛点解决**：HTML 表格代码体积大、噪音多，直接向量化会导致关键信息被淹没。
    - **方案**：入库时利用 LLM 为表格生成语义摘要，检索阶段比对摘要，生成阶段还原完整 HTML，确保数据绝对精确。
- **全局递增表格 ID 系统**：
    - 针对多文档入库场景，系统通过“全局计数器 + 文件偏移量”逻辑，自动重新编号占位符，彻底解决跨文档表格 ID 冲突问题。
- **混合检索与重排序 (Hybrid Search & RRF)**：
    - 集成 **BGE-zh** 向量检索（语义）与 **BM25** 关键词检索（精确匹配）。
    - 针对表格检索，BM25 自动切换为对“摘要”进行算分，规避标签干扰。
    - 使用 **RRF (Reciprocal Rank Fusion)** 算法融合多路检索结果。
- **多级 Query 优化流水线**：
    - **意图识别**：精准区分闲聊与专业财务检索。
    - **多轮对话改写**：补全代词和省略语。
    - **检索关键词精简**：自动将自然语言转化为标准的财务科目术语。
    - **记忆功能**: 多轮对话中参考上下文记忆进行回答。
- **来源自动标注 (Source Tagging)**：
    - 在多文档检索场景下，系统为每一段召回的内容自动打上来源文件标签，防止大模型在对比分析时张冠李戴。

##  系统架构

1.  **数据层**：PostgreSQL + `pgvector` 扩展。
2.  **处理层**：`split.py` 进行语义切分，剥离表格并植入占位符。
3.  **索引层**：`vector.py` 生成表格摘要并构建混合索引（向量 + BM25）。
4.  **生成层**：内存中还原占位符，拼装带来源标签的 Context，调用 DeepSeek 生成回答。

## 🛠️ 技术栈

- **嵌入模型 (Embedding)**: `BAAI/bge-small-zh-v1.5`
- **语言模型 (LLM)**: DeepSeek-V4-flash
- **向量数据库**: PostgreSQL + Pgvector
- **分词工具**: Jieba
- **检索算法**: BM25, RRF

## 🚀 快速开始

### 1. 环境准备
创建并配置 `.env` 文件：
```env
DEEPSEEK_API_KEY=你的API密钥
DEEPSEEK_API_BASE=[https://api.deepseek.com](https://api.deepseek.com)
DEEPSEEK_MODEL=deepseek-v4-flash

PG_HOST=localhost
PG_PORT=5432
PG_USER=postgres
PG_PASSWORD=你的密码
PG_DATABASE=rag_db
```

### 2. 数据库初始化与数据入库
```python
from vector import build_vector_db

# 指定包含 .md 文件的文件夹路径
model = build_vector_db("D:/data/financial_reports")
```

### 3. 对话与检索测试
```python
from test_script import test_generation

query = "盛和资源和合诚股份，哪家公司2019年的研发投入占比更高？"
# 系统会跨文档检索，合并结果并打上来源标签返回给大模型
test_generation(query, top_k=10)
```
