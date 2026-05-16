"""
LLM调用模块 - 智能路由、查询改写、意图识别
"""
from openai import OpenAI
from .config import LLMConfig


def get_llm_client():
    """获取LLM客户端（单例模式）"""
    if not hasattr(get_llm_client, '_client'):
        get_llm_client._client = OpenAI(
            api_key=LLMConfig.API_KEY,
            base_url=LLMConfig.BASE_URL
        )
    return get_llm_client._client


def router(query: str, history: list = None) -> list:
    """
    使用 DeepSeek 模型进行智能路由
    注意：history参数已弃用，query应由rewrite函数预处理

    Args:
        query: 用户查询（应已被rewrite处理为完整问题）
        history: 保留参数（不再使用）

    Returns:
        list: 匹配的文档列表，如 ['xxx.md']
               如果都不匹配返回 [None]（检索全部）
    """
    from .database import get_db_connection

    client = get_llm_client()

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


    response = client.chat.completions.create(
        model=LLMConfig.MODEL,
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
    client = get_llm_client()

    if history is None:
        history = []

    # 如果没有历史对话，直接返回原query
    if not history:
        return query

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

    response = client.chat.completions.create(
        model=LLMConfig.MODEL,
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
    client = get_llm_client()

    if history is None:
        history = []

    # 构建简化的prompt
    prompt = f"""判断用户问题的意图，只需回答"chat"或"retrieval"。

用户问题：{query}

判断标准：
- **chat（闲聊）**：打招呼、感谢、询问系统功能、与财报无关的问题
  例："你好"、"谢谢"、"你能做什么"、"今天天气怎么样"

- **retrieval（检索）**：询问财报内容、公司信息、财务数据、业务情况等
  例："营收是多少"、"股票代码"、"业务板块"、"法人是谁"

只返回一个词：chat 或 retrieval，不要其他解释。"""

    response = client.chat.completions.create(
        model=LLMConfig.MODEL,
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
    client = get_llm_client()

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

    response = client.chat.completions.create(
        model=LLMConfig.MODEL,
        extra_body={"thinking": {"type": "disabled"}},
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    optimized_query = response.choices[0].message.content.strip()
    print(f" 检索优化rewrite: {query} → {optimized_query}")

    return optimized_query
