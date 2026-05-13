import os
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from split import custom_markdown_splitter
from rank_bm25 import BM25Okapi
import jieba
import torch
load_dotenv()

# 全局模型实例
_model = None
_bm25 = None
_all_chunks = None

def get_model():
    """获取或初始化 embedding 模型"""
    global _model
    if _model is None:
        # 2. 加载模型，并指定使用 GPU（如果有）
        device = "cuda"
        _model = SentenceTransformer('BAAI/bge-small-zh-v1.5',device)
    return _model

def get_db_connection():
    """获取数据库连接"""
    return psycopg.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5432"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD"),
        dbname=os.getenv("PG_DATABASE", "rag_db")
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
        # 将 text_for_bm25 放第一位供初始化 BM25 用，chunk_text 放第三位供最后返回用
        _all_chunks[doc_id] = (text_for_bm25, source_file, chunk_text)
        
    conn.close()   
    return _all_chunks

def init_bm25():
    """初始化 BM25 模型"""
    global _bm25
    if _bm25 is not None:
        return

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
    return "<table>" in chunk or "<tr>" in chunk


def summary(table_content: str) -> dict:
    """
    使用LLM生成表格的结构化摘要

    Args:
        table_content: HTML表格内容

    Returns:
        dict: 包含table_title, headers, core_summary, key_entities, retrieval_keywords
    """
    import os
    from dotenv import load_dotenv
    from openai import OpenAI
    import json

    load_dotenv()

    # 初始化 DeepSeek 客户端
    deepseek_client = OpenAI(
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url=os.getenv('DEEPSEEK_API_BASE')
    )
    DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL')

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

    response = deepseek_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        extra_body={"thinking": {"type": "disabled"}},
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
    
    import re

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
                # split.py 提取出的 m.group(1) 是局部的 0,1,2...
                # 把它转换为整数，加上当前文件的偏移量，就变成了全局唯一的数字！
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


def dense_search(query: str, top_k: int = 10, source_filter: str=None):
    """
    密集检索（向量）

    Returns:
        list: [(doc_id, chunk_text, score), ...] score 是距离（越小越好）
    """
    model = get_model()
    query_vec = model.encode([query], normalize_embeddings=True)[0]

    conn = get_db_connection()
    register_vector(conn)

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

def sparse_search(query: str, top_k: int=10, source_filter: str=None, normalize: bool=True):
    """
    稀疏检索（BM25）
    source_filter: 可选, 指定md文档查询
    normalize: 可选, 是否对bm25得分进行sigmoid归一化
    Returns:
        list: [(doc_id, chunk_text, score), ...] score 是 BM25 分数（越大越好）
    """
    global _bm25, _all_chunks

    if _bm25 is None:
        init_bm25()

    # 用 jieba 分词
    tokenized_query = list(jieba.cut(query))

    scores = _bm25.get_scores(tokenized_query)
    # BM25 检索
    if normalize:
        import math
        def sigmoid_normalize(score, scale=25):
            return 1 / (1 + math.exp(-score / scale))
        scores = [sigmoid_normalize(score) for score in scores]


    # 获取 top-k
    import numpy as np
    candidate_k = top_k * 10 if source_filter else top_k
    top_indices = np.argsort(scores)[::-1][:candidate_k]

    chunks_dict = load_all_chunks()
    results = []
    for idx in top_indices:
        doc_id = sorted(chunks_dict.keys())[idx]
        
        # 【修改点】由于 load_all_chunks 返回了三个元素，这里解包取第三个(chunk_text)用于返回
        text_for_bm25, source_file, chunk_text = chunks_dict[doc_id]  
        
        if source_filter and source_file != source_filter:    
            continue        
        results.append((doc_id, chunk_text, float(scores[idx]), source_file))
        if len(results) >= top_k:
            break
    return results

def hybrid_search(query: str, top_k: int = 5, rrf_k: int = 60, source_filter: str=None):
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
    chunks_dict = load_all_chunks()
    final_results = [
        # chunks_dict[doc_id][2] 获取真实的 chunk_text 返回
        (doc_id, chunks_dict[doc_id][2], score, chunks_dict[doc_id][1])
        for doc_id, score in sorted_results
    ]

    return final_results




def router(query: str, history: list = None) -> list:
    """
    使用 DeepSeek-v4-flash 模型进行智能路由
    注意：history参数已弃用，query应由rewrite函数预处理

    Args:
        query: 用户查询（应已被rewrite处理为完整问题）
        history: 保留参数（不再使用）

    Returns:
        list: 匹配的文档列表，如 ['xxx.md']
               如果都不匹配返回 [None]（检索全部）
    """
    import os
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()

    # 初始化 DeepSeek 客户端
    deepseek_client = OpenAI(
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url=os.getenv('DEEPSEEK_API_BASE')
    )
    DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL')
    # 获取所有文档
    conn = get_db_connection()
    cur = conn.execute("SELECT DISTINCT source_file FROM financial_vectors")
    all_docs = [row[0] for row in cur.fetchall()]
    conn.close()

    # 构建文档列表文本
    docs_text = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(all_docs)])

    # 构建prompt
    prompt = f"""你是一个文档路由助手。请分析用户问题，判断需要查询哪家或哪几家公司的财报文档。

可用文档列表：
{docs_text}

用户问题：{query}

请按以下要求回答：
1. 如果问题明确提到某家公司，返回该公司对应的完整文档名, 例如: query: "汇金机电股份有限公司的股票代码是什么?"  返回: ['河北汇金机电股份有限公司 2019年度报告.md']
2. 如果问题涉及多家公司，返回所有相关文档名，用逗号分隔, 有时候用户给出的query中可能会是公司的缩写, 例如: query: "汇金机电股份有限公司和盛和控股公司的股票代码分别是什么?" , 你需要先判断比对一下这两个公司和docs_text中的哪些公司最相像, 返回: ['河北汇金机电股份有限公司 2019年度报告.md', '盛和资源控股股份有限公司2019年年度报告.md']
3. 如果问题没有明确提到任何公司，或者无法判断，返回"全部"
重要：返回格式必须是 Python 列表，例如: ['文件1.md', '文件2.md'] 或 "全部"
只返回文档名或"全部"，不要其他解释。"""

    # 调用 DeepSeek
    response = deepseek_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        extra_body={"thinking": {"type": "disabled"}},
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    result = response.choices[0].message.content.strip()
    print(f"🤖 LLM返回: {result}")
    
    # 解析结果
    if result == "全部" or result == "[None]" or result is None:
        print("⚠️ LLM建议检索全部文档")
        return [None]
    
    import ast

    # 解析 LLM 返回的列表并匹配
    matched_docs = []
    llm_docs = ast.literal_eval(result.strip())
    
    # 去掉空格和后缀再进行匹配
    for llm_doc in llm_docs:
        llm_clean = llm_doc.replace(' ', '').replace('.md', '')
        for db_doc in all_docs:
            db_clean = db_doc.replace(' ', '').replace('.md', '')
            if llm_clean == db_clean:
                matched_docs.append(db_doc)
                break

    if matched_docs:
        print(f"✅ 匹配到文档: {matched_docs}")
        return matched_docs
    else:
        print("⚠️ LLM返回的文档名未找到，检索全部")
        return [None]


def rewrite(query: str, history: list = None) -> str:
    """
    使用 LLM 对用户query进行改写，处理多轮对话中的省略、代词等问题

    Args:
        query: 用户当前问题
        history: 对话历史

    Returns:
        str: 改写后的完整问题
    """
    import os
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()

    if history is None:
        history = []

    # 如果没有历史对话，直接返回原query
    if not history:
        return query

    # 初始化 DeepSeek 客户端
    deepseek_client = OpenAI(
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url=os.getenv('DEEPSEEK_API_BASE')
    )
    DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL')

    # 构建对话历史文本
    conversation = []
    for msg in history:
        role = msg.get('role', '')
        content = msg.get('content', '')
        if role in ['user', 'assistant']:
            # 截断过长的回答
            if len(content) > 200:
                content = content[:200] + "..."
            conversation.append(f"{role}: {content}")

    if conversation:
        # 只取最近2轮对话
        recent_conversation = conversation[-4:] if len(conversation) > 4 else conversation
        history_text = "\n".join(recent_conversation)
    else:
        history_text = "（无历史对话）"

    # 构建prompt
    prompt = f"""你是一个查询改写助手。请根据对话历史，将用户的当前问题改写成一个完整、独立的问题。

对话历史：
{history_text}

当前问题：{query}

改写要求：
1. 如果当前问题已经完整、独立，直接返回原问题
2. 如果当前问题包含代词（"它"、"这家公司"、"该企业"等）或省略，请根据对话历史补全具体信息
3. 改写后的问题应该是一个完整的句子，即使脱离上下文也能理解
4. 只返回改写后的问题，不要其他解释

示例：
- 历史: user: 盛和公司的业务板块有哪些？ assistant: ...
  当前: 它的注册资本是多少？
  改写: 盛和资源的注册资本是多少？

现在请改写："""

    response = deepseek_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        extra_body={"thinking": {"type": "disabled"}},
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    rewritten_query = response.choices[0].message.content.strip()


    return rewritten_query


def intent(query: str, history: list = None) -> str:
    """
    意图识别：判断用户问题是闲聊还是需要检索财报

    Args:
        query: 用户当前问题
        history: 对话历史（可选）

    Returns:
        str: "chat"（闲聊）或 "retrieval"（需要检索）
    """
    import os
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()

    if history is None:
        history = []

    # 初始化 DeepSeek 客户端
    deepseek_client = OpenAI(
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url=os.getenv('DEEPSEEK_API_BASE')
    )
    DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL')

    # 构建简化的prompt
    prompt = f"""判断用户问题的意图，只需回答"chat"或"retrieval"。

用户问题：{query}

判断标准：
- **chat（闲聊）**：打招呼、感谢、询问系统功能、与财报无关的问题
  例："你好"、"谢谢"、"你能做什么"、"今天天气怎么样"

- **retrieval（检索）**：询问财报内容、公司信息、财务数据、业务情况等
  例："营收是多少"、"股票代码"、"业务板块"、"法人是谁"

只返回一个词：chat 或 retrieval，不要其他解释。"""

    response = deepseek_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        extra_body={"thinking": {"type": "disabled"}},
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    intent = response.choices[0].message.content.strip().lower()

    # 标准化返回值
    if "chat" in intent:
        print("闲聊")
        return "chat"
    else:
        print("需要检索")
        return "retrieval"


def rewrite_for_retrieval(query: str) -> str:
    """
    检索优化型rewrite：精简query，提取核心关键词，提高检索匹配度

    目的：在router确定文档范围后，进一步优化query用于检索

    Args:
        query: 用户问题（应已被rewrite处理为完整问题）

    Returns:
        str: 精简优化后的查询
    """
    import os
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()

    # 初始化 DeepSeek 客户端
    deepseek_client = OpenAI(
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url=os.getenv('DEEPSEEK_API_BASE')
    )
    DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL')

    # 构建prompt
    prompt = f"""
#Role
你是一个检索查询优化助手。请将用户问题精简为最核心的检索关键词，使query与财报chunk中的表达方式一致。

原始问题：{query}

优化要求：
1. 改写后的query不能改变改写前的query的意义, 改写后的query不能缺少意思, 例如改写前query包含两个子问题, 改写后的query也应该包含两个子问题
2. 如果query中只涉及一家公司, 即单公司问题, 则去除公司名称（已由router确定检索范围）
3. 简化时间表达（如："2019年12月31日" → "2019年末"；"2019年1月1日" → "2019年初"）
4. 去除疑问词和助词（"是多少"、"的"、"了"、"呢"等）
5. 财务名词标准化：使用财报中的标准表达
   - "营业收入"/"营收" → "营业收入"
   - "净利润"/"归母净利润" → "归属于上市公司股东的净利润"
   - "每股收益" → "基本每股收益"
   - "经营现金流" → "经营活动产生的现金流量净额"
   - "ROE" → "加权平均净资产收益率"
   - "资产负债率" → "资产负债率"
6. 如果原问题是比较类问题，那就需要保留公司名, 不要保留公司的全名, 只保留公司名缩写, 例如: 河北汇金机电股份有限公司→汇金机电"**
7. 如果query中涉及两家及以上的公司, 这时候也要保留公司名缩写, 记住所有的多家公司比较问题都要保留公司名缩写, 无论是比较问题还是同时查询两家公司的数据
8. 保留核心名词，保持自然简洁

示例：
- "合诚工程咨询集团在2019年12月31日的总资产是多少？"
  → "2019年末总资产"

- "盛和资源的注册地址在哪里？"
  → "注册地址"

- "汇金机电的主要业务板块有哪些"
  → "主要业务板块"

- "汇金机电2019年营收是多少"
  → "2019年营业收入"

- "公司2019年的净利润"
  → "2019年归属于上市公司股东的净利润"

- "每股收益是多少"
  → "基本每股收益"

- "汇金股份和山东新华锦，哪家公司的资产负债率更低？"
  → "汇金机电和新华锦资产负债比较"

- "盛和资源和合诚股份，哪家公司的营收更高"
  → "盛和资源和合诚股份营业收入比较"

每次改写query前一定要认真查看改写要求和改写示例,   只返回优化后的查询，不要其他解释。"""

    response = deepseek_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        extra_body={"thinking": {"type": "disabled"}},
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    optimized_query = response.choices[0].message.content.strip()
    print(f" 检索优化rewrite: {query} → {optimized_query}")

    return optimized_query







