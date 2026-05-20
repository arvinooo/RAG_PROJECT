"""
RAG核心模块 - 导出所有核心函数
"""

# 数据库操作
from .database import (
    build_vector_db,
    get_doc_count,
    get_model,
    resolve_placeholders,
    get_all_docs,
)

# 检索操作
from .retrieval import (
    hybrid_search,
    dense_search,
    sparse_search,
    two_stage_retrieval,
    get_reranker,
)

# LLM操作
from .llm import (
    router,
    rewrite,
    intent,
    rewrite_for_retrieval,
)

__all__ = [
    # 数据库
    'build_vector_db',
    'get_doc_count',
    'get_model',
    'resolve_placeholders',
    # 检索
    'hybrid_search',
    'dense_search',
    'sparse_search',
    'two_stage_retrieval',
    # LLM
    'router',
    'rewrite',
    'intent',
    'rewrite_for_retrieval',
    # Rerank
    'get_reranker',
]
