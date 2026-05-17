"""
检索模块 - 密集检索、稀疏检索、混合检索
"""
import numpy as np
from . import database


def dense_search(query: str, top_k: int = 10, source_filter: str = None):
    """
    密集检索（向量）

    Returns:
        list: [(doc_id, chunk_text, score), ...] score 是距离（越小越好）
    """
    model = database.get_model()
    query_vec = model.encode([query], normalize_embeddings=True)[0]

    conn = database.get_db_connection()
    database.register_vector(conn)

    # 将向量转换为字符串格式，并用 CAST 显式转换类型
    vec_str = str(query_vec.tolist())
    if source_filter:
        cur = conn.execute(
            """
            SELECT id, chunk_text, embedding <=> CAST(%s AS vector) AS distance, source_file
            FROM financial_vectors
            WHERE source_file = %s
            ORDER BY distance
            LIMIT %s
            """,
            (vec_str, source_filter, top_k)
        )
    else:
        cur = conn.execute(
            """
            SELECT id, chunk_text, embedding <=> CAST(%s AS vector) AS distance, source_file
            FROM financial_vectors
            ORDER BY distance
            LIMIT %s
            """,
            (vec_str, top_k)
        )
    results = [(row[0], row[1], float(row[2]), row[3]) for row in cur.fetchall()]
    conn.close()
    return results


def sparse_search(query: str, top_k: int = 10, source_filter: str = None, normalize: bool = True):
    """
    稀疏检索（BM25）
    source_filter: 可选, 指定md文档查询
    normalize: 可选, 是否对bm25得分进行sigmoid归一化
    Returns:
        list: [(doc_id, chunk_text, score), ...] score 是 BM25 分数（越大越好）
    """
    if database._bm25 is None:
        database.init_bm25()

    # 用 jieba 分词
    import jieba
    tokenized_query = list(jieba.cut(query))

    scores = database._bm25.get_scores(tokenized_query)
    # BM25 检索
    if normalize:
        import math

        def sigmoid_normalize(score, scale=25):
            return 1 / (1 + math.exp(-score / scale))

        scores = [sigmoid_normalize(score) for score in scores]

    # 获取 top-k
    candidate_k = top_k * 10 if source_filter else top_k
    top_indices = np.argsort(scores)[::-1][:candidate_k]

    chunks_dict = database.load_all_chunks()
    results = []
    for idx in top_indices:
        doc_id = sorted(chunks_dict.keys())[idx]

        # chunks_dict 返回三个元素，取第三个(chunk_text)用于返回
        text_for_bm25, source_file, chunk_text = chunks_dict[doc_id]

        if source_filter and source_file != source_filter:
            continue
        results.append((doc_id, chunk_text, float(scores[idx]), source_file))
        if len(results) >= top_k:
            break
    return results


def hybrid_search(query: str, top_k: int = 5, rrf_k: int = 60, source_filter: str = None):
    """
    混合检索（密集 + 稀疏）+ RRF 重排序

    Args:
        query: 查询文本
        top_k: 最终返回结果数
        rrf_k: RRF 参数（默认 60）

    Returns:
        list: [(doc_id, chunk_text, rrf_score), ...]
    """
    # 1. 密集检索（取更多候选）
    dense_results = dense_search(query, top_k=10, source_filter=source_filter)

    # 2. 稀疏检索（取更多候选）
    sparse_results = sparse_search(query, top_k=10, source_filter=source_filter)

    # 3. RRF 融合
    scores = {}  # {doc_id: rrf_score}

    for rank, (doc_id, text, _, _) in enumerate(dense_results, start=1):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + rank)

    for rank, (doc_id, text, _, _) in enumerate(sparse_results, start=1):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + rank)

    # 4. 排序并取 top-k
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # 5. 组装最终结果
    chunks_dict = database.load_all_chunks()
    final_results = [
        # chunks_dict[doc_id][2] 获取真实的 chunk_text 返回
        (doc_id, chunks_dict[doc_id][2], score, chunks_dict[doc_id][1])
        for doc_id, score in sorted_results
    ]

    return final_results
