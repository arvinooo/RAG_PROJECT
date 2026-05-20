"""
LLM调用模块 - 智能路由、查询改写、意图识别
"""
from openai import OpenAI
from .config import LLMConfig, LlamaCppConfig, DEFAULT_LLM_PROVIDER


def get_llm_client(provider: str = None):
    """获取LLM客户端（单例模式）

    Args:
        provider: "deepseek" 或 "llamacpp"，None 则用配置默认值
    """
    import os

    if provider is None:
        provider = DEFAULT_LLM_PROVIDER
    # 如果指定的 provider 没有配置 API Key，自动降级到 llamacpp
    if provider == "deepseek" and not LLMConfig.API_KEY:
        print("⚠️ DeepSeek API Key 未配置，自动切换到 Llama.cpp")
        provider = "llamacpp"

    # Llama.cpp 走内网，绕过系统代理
    if provider == "llamacpp":
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ["no_proxy"] = "*"

    cache_attr = f'_client_{provider}'
    if not hasattr(get_llm_client, cache_attr):
        if provider == "llamacpp":
            from .config import LlamaCppConfig
            setattr(get_llm_client, cache_attr, OpenAI(
                api_key=LlamaCppConfig.API_KEY,
                base_url=LlamaCppConfig.API_BASE
            ))
        else:
            setattr(get_llm_client, cache_attr, OpenAI(
                api_key=LLMConfig.API_KEY,
                base_url=LLMConfig.BASE_URL
            ))
    return getattr(get_llm_client, cache_attr)


def _model_name():
    """根据默认提供者返回模型名"""
    if DEFAULT_LLM_PROVIDER == "llamacpp":
        return LlamaCppConfig.MODEL
    return LLMConfig.MODEL


def _extra_body():
    """根据默认提供者返回 extra_body"""
    if DEFAULT_LLM_PROVIDER == "llamacpp":
        return LlamaCppConfig.extra_body()
    return {"thinking": {"type": "disabled"}}


def router(query: str, history: list = None) -> list:
    """
    使用 DeepSeek 模型进行智能路由

    Args:
        query: 用户查询（应已被rewrite处理为完整问题）
        history: 保留参数（不再使用）

    Returns:
        list: 匹配的文档列表，如 ['xxx.md']
               如果需要检索全部返回 ['全部']
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
    prompt = f"""

你是一个极其严格的文档路由助手。请分析用户问题（Query），从【可用文档列表】中精准判断需要查询哪家或哪几家公司的财报文档。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【可用文档列表】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{docs_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【路由核心规则】（严格按优先级执行）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 【无需检索】：如果用户提问是日常闲聊（你好、在吗）、感谢（谢谢）、或与财务/公司基本面完全无关的百科常识（今天天气、某明星是谁）。
   → 必须返回：[] （即一个空列表，代表不需要检索任何文档）

2. 【定向检索】：如果问题明确或模糊提到了某家公司（包含全称、缩写、简称，你需要比对并匹配出最相像的完整文档名）。
   → 必须返回：包含对应完整文档名的 Python 列表。
   示例 1：用户提问 "汇金机电的股票代码" → 返回：['河北汇金机电股份有限公司 2019年度报告.md']
   示例 2：用户提问 "汇金机电和盛和控股" → 返回：['河北汇金机电股份有限公司 2019年度报告.md', '盛和资源控股股份有限公司2019年年度报告.md']

3. 【全量检索】：如果问题属于财务/行业相关的专业提问，但属于宏观提问、行业对比，或者未指明具体公司（如："哪家公司的经营活动现金流质量最好？"、"哪家公司的毛利率最高？"）。
   → 必须返回字符串："全部"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【输出约束（绝对不能违反）】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 你的输出必须能被 Python 的 `json.loads()` 或 `eval()` 直接解析。
- 只能返回标准的 Python 列表（如 `['file.md']`、`[]`）或字符串（如 `"全部"`）。
- 严禁包含任何解释性文本、严禁带有开场白或结束语。
- 严禁使用 Markdown 代码块标签（如 ```python 或 ```）包裹答案。

用户问题：{query}
输出结果：
"""
    response = client.chat.completions.create(
        model=_model_name(),
        extra_body=_extra_body(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    result = response.choices[0].message.content.strip()
    print(f"🤖 LLM返回: {result}")

    # 解析结果
    if result == "全部" or result == "[None]" or result is None:
        print("⚠️ LLM建议检索全部文档")
        return ["全部"]

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
        return ["全部"]


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
        model=_model_name(),
        extra_body=_extra_body(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    rewritten_query = response.choices[0].message.content.strip()

    return rewritten_query


# def intent(query: str, history: list = None) -> str:
#     """
#     意图识别：判断用户问题是闲聊还是需要检索财报

#     Args:
#         query: 用户当前问题
#         history: 对话历史（可选）

#     Returns:
#         str: "chat"（闲聊）或 "retrieval"（需要检索）
#     """
#     client = get_llm_client()

#     if history is None:
#         history = []

#     # 构建简化的prompt
#     prompt = f"""判断用户问题的意图，只需回答"chat"或"retrieval"。

# 用户问题：{query}

# 判断标准：
# - **chat（闲聊）**：打招呼、感谢、询问系统功能、与财报无关的问题
#   例："你好"、"谢谢"、"你能做什么"、"今天天气怎么样"

# - **retrieval（检索）**：询问财报内容、公司信息、财务数据、业务情况等
#   例："营收是多少"、"股票代码"、"业务板块"、"法人是谁"

# 只返回一个词：chat 或 retrieval，不要其他解释。"""

#     response = client.chat.completions.create(
#         model=_model_name(),
#         extra_body=_extra_body(),
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0
#     )

#     intent = response.choices[0].message.content.strip().lower()

#     # 标准化返回值
#     if "chat" in intent:
#         print("闲聊")
#         return "chat"
#     else:
#         print("需要检索")
#         return "retrieval"


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
你是一个极其严谨的文档检索查询优化助手（Query Rewriter）。你的唯一任务是将用户的口语化提问，精简重构为最适合在财报文档中进行模糊匹配的核心关键词组合。

# 原始问题：{query}

# 核心优化规则：
1. 【精准实体控制（最核心）】：
   - **强制去除：【目标/母公司主体名称】**（如“盛和资源”）。因为系统已锁定对应文档。
   - **强制保留：【内部业务实体名称】**。如果提问明确指出了具体的**子公司、分公司、联营/合营企业、特定项目、业务板块或产品线**（如“乐山盛和”、“赣州晨光”、“海南文盛”），**必须保留**这些名称！它们是在长篇财报内部定位特定表格和段落的唯一检索锚点。   
   - **强制保留：【关键人物姓名】**（如“王晓晖”、“张三”）。如果问题涉及高管、董事、股东或法人的任职、薪酬、持股等，**必须保留其姓名**

2. 【提纯核心指标】：去除疑问词、助词、修饰语（如："是多少"、"的"、"了"、"呢"、"哪家更高"、"对比"、"分别"等），仅保留实质性的业务或财务名词。如果问题包含多个子问题，保留所有相关的核心指标，用空格隔开。

3. 【时间表达绝对标准化】：
   - "2019年12月31日"、"19年底" → "2019年末"
   - "2019年1月1日"、"19年初" → "2019年初"
   - 带有月份或季度的保持原意精简，如"2019年第一季度"

4. 【财务名词强制映射】：严格使用中国证监会财报标准术语替换口语表达：
   - "营业收入" / "营收" / "销售额" → "营业收入"
   - "净利润" / "归母净利润" / "赚了多少钱" → "归属于上市公司股东的净利润"
   - "每股收益" / "EPS" → "基本每股收益"
   - "经营现金流" / "现金流" → "经营活动产生的现金流量净额"
   - "ROE" / "净资产收益率" → "加权平均净资产收益率"

# 示例：
- "合诚工程咨询集团在2019年12月31日的总资产是多少？"
  → "2019年末 总资产"

- "盛和资源的注册地址在哪里？"
  → "注册地址"

- "汇金机电的主要业务板块有哪些"
  → "主要业务板块"

- "汇金机电2019年营收是多少"
  → "2019年 营业收入"

- "公司2019年的净利润"
  → "2019年 归属于上市公司股东的净利润"

- "汇金股份和山东新华锦，哪家公司的资产负债率更低？"
  → "资产负债率"  (注：去除了公司名和“比较”、“更低”等推理词)

- "盛和资源和合诚股份，2019年的营收和净利润分别是多少？"
  → "2019年 营业收入 归属于上市公司股东的净利润"

# 输出约束：
每次改写前请认真核对规则。直接输出最终的改写结果，绝对不要输出任何其他解释性文字、开场白或符号。"""

    response = client.chat.completions.create(
        model=_model_name(),
        extra_body=_extra_body(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    optimized_query = response.choices[0].message.content.strip()
    print(f" 检索优化rewrite: {query} → {optimized_query}")

    return optimized_query
