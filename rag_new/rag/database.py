"""
数据库操作模块
"""
import os
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from .config import DatabaseConfig, EmbeddingConfig, PathConfig

load_dotenv()

# 全局模型实例
_model = None
_bm25 = None
_all_chunks = None


def get_model():
    """获取或初始化 embedding 模型"""
    global _model
    if _model is None:
        _model = SentenceTransformer(EmbeddingConfig.MODEL_NAME, device=EmbeddingConfig.DEVICE)
    return _model


def get_db_connection():
    """获取数据库连接"""
    return psycopg.connect(
        host=DatabaseConfig.HOST,
        port=DatabaseConfig.PORT,
        user=DatabaseConfig.USER,
        password=DatabaseConfig.PASSWORD,
        dbname=DatabaseConfig.DATABASE
    )


def init_db():
    """初始化数据库表和扩展"""
    conn = get_db_connection()
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS financial_vectors (
            id SERIAL PRIMARY KEY,
            chunk_text TEXT NOT NULL,
            source_file TEXT NOT NULL,
            embedding vector(512),
            summary TEXT,
            abstract TEXT,
            table_id VARCHAR(50)
        )
    """)
    conn.commit()
    conn.close()


def get_doc_count():
    """获取数据库中的文档数量"""
    conn = get_db_connection()
    register_vector(conn)
    cur = conn.execute("SELECT COUNT(*) FROM financial_vectors")
    count = cur.fetchone()[0]
    conn.close()
    return count


def load_all_chunks():
    """从数据库加载所有 chunks（用于 BM25）"""
    global _all_chunks
    if _all_chunks is not None:
        return _all_chunks

    conn = get_db_connection()
    cur = conn.execute("SELECT id, chunk_text, source_file, summary FROM financial_vectors ORDER BY id")

    _all_chunks = {}
    for row in cur.fetchall():
        doc_id, chunk_text, source_file, db_summary = row
        # 如果有 summary，就用 summary 做 BM25 分词和算分！否则用原文。
        text_for_bm25 = db_summary if db_summary else chunk_text
        _all_chunks[doc_id] = (text_for_bm25, source_file, chunk_text)

    conn.close()
    return _all_chunks


def init_bm25():
    """初始化 BM25 模型"""
    global _bm25
    if _bm25 is not None:
        return

    from rank_bm25 import BM25Okapi
    import jieba

    chunks_dict = load_all_chunks()
    chunks_list = [chunks_dict[id][0] for id in sorted(chunks_dict.keys())]
    # 用 jieba 分词
    tokenized_chunks = [list(jieba.cut(chunk)) for chunk in chunks_list]
    _bm25 = BM25Okapi(tokenized_chunks)


def clear_all():
    """清空所有向量数据"""
    global _bm25, _all_chunks
    conn = get_db_connection()
    conn.execute("TRUNCATE TABLE financial_vectors")
    conn.commit()
    conn.close()

    _bm25 = None
    _all_chunks = None
    print("已清空向量表")


def is_table_chunk(chunk: str) -> bool:
    """判断是否为表格chunk"""
    return "<table>" in chunk or "<tr>" in chunk


def summary(table_content: str) -> dict:
    """
    使用LLM生成表格的结构化摘要

    Args:
        table_content: HTML表格内容

    Returns:
        dict: 包含table_title, headers, core_summary, key_entities, retrieval_keywords
    """
    import json
    from .llm import get_llm_client, _model_name, _extra_body

    client = get_llm_client()

    # 构建提示词
    prompt = f"""# Role
你是一名精通数据处理和自然语言理解的表格分析专家。你的任务是将 HTML 格式的表格转换为结构化的语义摘要，以优化检索增强生成（RAG）系统的召回效果。

# Task
接收一段 HTML `<table>` 代码，执行以下操作：
1. **提取**：识别表格标题、表头（列名）。
2. **总结**：用自然语言概括表格的核心内容、关键数据趋势或结论。
3. **输出**：生成标准的 JSON 格式结果。

# Output Format
请严格输出以下 JSON 格式，不要包含 Markdown 代码块标记（```json）或其他解释性文字：

{{
  "table_title": "表格的标题或主题（若无标题，请根据内容生成一个简洁的概括性标题）",
  "headers": ["列名1", "列名2", "列名3"],
  "core_summary": "用 100-200 字以内的自然语言概括表格的核心信息。重点描述：表格涉及的时间范围、主要实体、关键指标的变化趋势或结论。避免罗列具体数据，侧重语义描述。",
  "key_entities": ["实体1", "实体2", "实体3"],
  "retrieval_keywords": ["关键词1", "关键词2", "关键词3", "关键词4"]
}}

# Constraints
1. **表名提取**：如果 HTML 中有 `<caption>` 或第一行是标题行，优先提取。如果没有，根据表头内容推断一个最贴切的标题。
2. **表头清洗**：只提取纯文本列名，去除空格和特殊符号。
3. **核心总结**：
   - 必须包含表格的**主要维度**（如：时间、地区、产品）。
   - 必须包含**关键洞察**（如："A产品销量最高"、"Q4数据出现下滑"）。
   - 语言要流畅，适合作为向量检索的索引文本。
4. **关键词**：提取表格中高频出现的业务名词、指标名称、时间、地点，用于辅助关键词检索。

# Example
Input:
<table>
  <tr><td colspan="3">常用词语释义</td></tr>
  <tr><td>报告期</td><td>指</td><td>2019年1月1日—2019年12月31日</td></tr>
  <tr><td>合诚股份</td><td>指</td><td>合诚工程咨询集团股份有限公司</td></tr>
</table>

Output:
{{
  "table_title": "常用词语释义表",
  "headers": ["词语", "指代", "释义"],
  "core_summary": "该表格为常用词语释义表，定义了报告期（2019年1月1日至2019年12月31日）、合诚股份（合诚工程咨询集团股份有限公司）等关键术语的含义。",
  "key_entities": ["报告期", "合诚股份", "2019年"],
  "retrieval_keywords": ["释义", "报告期", "合诚股份", "词语", "定义"]
}}

现在请处理以下表格：

{table_content}"""

    response = client.chat.completions.create(
        model=_model_name(),
        extra_body=_extra_body(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    result_text = response.choices[0].message.content.strip()

    # 解析JSON
    result = json.loads(result_text)
    return result


def resolve_placeholders(results: list) -> list:
    """
    仅在内存中解析结果，用完整的 HTML 替换全局统一编号的占位符
    """
    import re
    placeholder_pattern = re.compile(r'(__TABLE_PLACEHOLDER_\d+__)')

    conn = get_db_connection()
    resolved_results = []

    for doc_id, text, score, source_file in results:
        resolved_text = text
        matches = placeholder_pattern.findall(text)

        unique_matches = list(set(matches))

        for table_id in unique_matches:
            # 此时的 table_id 会是全局唯一的，例如 __TABLE_PLACEHOLDER_24__
            cur = conn.execute(
                "SELECT chunk_text FROM financial_vectors WHERE table_id = %s LIMIT 1",
                (table_id,)
            )
            result = cur.fetchone()
            if result:
                raw_html = result[0]
                resolved_text = resolved_text.replace(table_id, f"\n{raw_html}\n")

        resolved_results.append((doc_id, resolved_text, score, source_file))

    conn.close()
    return resolved_results


def build_vector_db(file_path: str, rebuild: bool = False):
    """构建或加载向量数据库（全局统一表格编号）"""
    from .split import custom_markdown_splitter
    import re

    init_db()
    doc_count = get_doc_count()

    if doc_count > 0 and not rebuild:
        print(f"数据库中已有 {doc_count} 条记录")
        init_bm25()
        return get_model()

    if doc_count > 0 and rebuild:
        print("清空现有数据...")
        clear_all()

    print("未检测到数据，开始构建向量数据库...")

    if os.path.isfile(file_path):
        md_files = [file_path]
    elif os.path.isdir(file_path):
        md_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.md')]
        md_files.sort()
    else:
        raise ValueError(f"路径不存在: {file_path}")

    print(f'找到 {len(md_files)} 个 markdown 文件，开始处理...')

    texts_to_embed = []
    db_records = []

    global_table_counter = 0

    for md_file in md_files:
        file_name = os.path.basename(md_file)
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()

        chunks = custom_markdown_splitter(md_content)

        file_start_offset = global_table_counter

        for chunk in chunks:
            if is_table_chunk(chunk):
                # 表格部分：直接使用全局计数器生成唯一的 ID，并自增
                table_id = f"__TABLE_PLACEHOLDER_{global_table_counter}__"
                global_table_counter += 1

                print(f"  [预处理] 正在调用 LLM 生成 {table_id} 的摘要... (来源: {file_name})")
                summary_dict = summary(chunk)
                summary_text = summary_dict.get('core_summary', '')

                texts_to_embed.append(summary_text)
                db_records.append({
                    "chunk_text": chunk,
                    "source_file": file_name,
                    "summary": summary_text,
                    "table_id": table_id
                })
            else:
                # 把局部编号转换为全局唯一编号
                modified_chunk = re.sub(
                    r'__TABLE_PLACEHOLDER_(\d+)__',
                    lambda m: f"__TABLE_PLACEHOLDER_{int(m.group(1)) + file_start_offset}__",
                    chunk
                )

                texts_to_embed.append(modified_chunk)
                db_records.append({
                    "chunk_text": modified_chunk,
                    "source_file": file_name,
                    "summary": None,
                    "table_id": None
                })

    print("开始进行 Embedding 向量化...")
    model = get_model()
    embeddings = model.encode(texts_to_embed, normalize_embeddings=True, batch_size=4, show_progress_bar=True)

    print("正在存入数据库...")
    conn = get_db_connection()
    register_vector(conn)

    with conn.cursor() as cur:
        for record, emb in zip(db_records, embeddings):
            cur.execute(
                """
                INSERT INTO financial_vectors
                (chunk_text, source_file, embedding, summary, table_id)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (record["chunk_text"], record["source_file"], emb.tolist(), record["summary"], record["table_id"])
            )
    conn.commit()
    conn.close()

    print(f"构建完成！所有文档共计处理了 {global_table_counter} 个表格。")
    init_bm25()
    return model
