"""
检索模块 - 密集检索、稀疏检索、混合检索、两阶段检索
"""
import numpy as np
from . import database
import os


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


def two_stage_retrieval(query: str, top_k: int = 10, recall_k: int = 30, score_threshold: float = 0):
    """
    两阶段检索：粗排 + Rerank精排

    Args:
        query: 查询文本
        top_k: 最终喂给大模型的结果数
        recall_k: 阶段1粗排召回数量
        score_threshold: 精排分数阈值，低于此分数的彻底丢弃，防幻觉

    Returns:
        list: [(doc_id, chunk_text, score, source), ...]
    """
    # 阶段1：粗排召回候选
    candidates = hybrid_search(query, top_k=recall_k)

    if not candidates:
        return []

    # 加载summary字典
    summary_dict = database.get_summary_dict()

    # 阶段2：使用Rerank精排
    reranker = get_reranker()

    # 构建pairs：表格使用summary，非表格使用原始内容
    from .database import is_table_chunk
    pairs = []
    for doc_id, chunk_text, _, source in candidates:
        # 获取该doc_id的summary（如果有）
        summary = summary_dict.get(doc_id, None)
        # 如果是表格且有summary，使用summary；否则使用原始chunk_text
        text_for_rerank = summary if summary and is_table_chunk(chunk_text) else chunk_text
        pairs.append([query, text_for_rerank])

    # 获取Rerank分数 (假设自带Sigmoid归一化到 [0,1])
    rerank_scores = reranker.compute_score(pairs, normalize=True)

    # 将新分数与原数据打包 (直接使用精排分数，抛弃粗排分数)
    reranked = []
    for (doc_id, chunk_text, _, source), rerank_score in zip(candidates, rerank_scores):
        # 增加阈值截断，剔除无关文档
        if rerank_score >= score_threshold:
            reranked.append((doc_id, chunk_text, rerank_score, source))

    # 按精排分数倒序
    reranked.sort(key=lambda x: x[2], reverse=True)

    # 返回 Top K
    return reranked[:top_k]


# ============ Reranker 模块 ============

# 模型路径
QWEN_RERANKER_PATH = r"C:\Users\Arvin\.cache\modelscope\hub\Qwen\Qwen3-Reranker-0___6B"


class BaseReranker:
    """Rerank基类"""

    def compute_score(self, pairs, normalize: bool = False):
        raise NotImplementedError


class QwenReranker(BaseReranker):
    """Qwen3-Reranker - 使用transformers加载"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or QWEN_RERANKER_PATH
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load_model(self):
        """延迟加载模型"""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            # 确定设备
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"[Reranker] Loading Qwen3-Reranker to {self._device}...")
            print(f"[Reranker] Model path: {self.model_path}")

            # 加载tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )

            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32
            ).to(self._device)

            # 解决 Batch > 1 时的 Padding 问题
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

            if self._model.config.pad_token_id is None:
                self._model.config.pad_token_id = self._tokenizer.pad_token_id

            self._model.eval()
            print(f"[Reranker] Model loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load reranker model: {e}")

    def compute_score(self, pairs, normalize: bool = False):
        """
        计算相关性分数

        Args:
            pairs: [[query1, doc1], [query2, doc2], ...]
            normalize: 是否归一化到[0,1]

        Returns:
            List[float]: 相关性分数列表
        """
        self._load_model()

        if not pairs:
            return []

        import torch

        scores = []
        batch_size = 4

        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i+batch_size]

                # 准备输入 - Qwen rerank格式
                texts = []
                for query, doc in batch:
                    text = f"Query: {query}\nDocument: {doc}"
                    texts.append(text)

                # Tokenize
                encoded = self._tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self._device)

                # 推理
                outputs = self._model(**encoded)
                batch_scores = outputs.logits.view(-1).cpu().tolist()
                scores.extend(batch_scores)

        # 归一化（使用sigmoid将分数映射到[0,1]）
        if normalize and scores:
            import torch
            scores_tensor = torch.tensor(scores)
            scores = torch.sigmoid(scores_tensor).tolist()

        return scores

    @property
    def device(self):
        """获取当前设备"""
        self._load_model()
        return self._device


# 全局单例
_reranker_instance = None


def get_reranker() -> BaseReranker:
    """获取reranker单例"""
    global _reranker_instance

    if _reranker_instance is None:
        _reranker_instance = QwenReranker()

    return _reranker_instance
