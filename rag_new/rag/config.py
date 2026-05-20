"""
RAG财报问答系统 - 配置管理
"""
from dotenv import load_dotenv
import os

load_dotenv()


# ============ 数据库配置 ============
class DatabaseConfig:
    HOST = os.getenv("PG_HOST", "localhost")
    PORT = os.getenv("PG_PORT", "5432")
    USER = os.getenv("PG_USER", "postgres")
    PASSWORD = os.getenv("PG_PASSWORD")
    DATABASE = os.getenv("PG_DATABASE", "rag_db")


# ============ LLM 配置 ============
class LLMConfig:
    API_KEY = os.getenv('DEEPSEEK_API_KEY')
    BASE_URL = os.getenv('DEEPSEEK_API_BASE')
    MODEL = os.getenv("DEEPSEEK_MODEL")


class LlamaCppConfig:
    """Llama.cpp 本地模型配置（Qwen3.6 非思考模式）"""
    API_BASE = os.getenv('LLAMA_CPP_API_BASE', 'http://localhost:12686/v1')
    API_KEY = os.getenv('LLAMA_CPP_API_KEY', 'llamacpp')
    MODEL = os.getenv('LLAMA_CPP_MODEL', 'Qwen3.6-27B-MTP-UD-Q5_K_XL.gguf')
    MAX_TOKENS = int(os.getenv('LLAMA_CPP_MAX_TOKENS', '16384'))
    TEMPERATURE = float(os.getenv('LLAMA_CPP_TEMPERATURE', '0.7'))
    TOP_P = float(os.getenv('LLAMA_CPP_TOP_P', '0.80'))
    TOP_K = int(os.getenv('LLAMA_CPP_TOP_K', '20'))
    MIN_P = float(os.getenv('LLAMA_CPP_MIN_P', '0.0'))
    PRESENCE_PENALTY = float(os.getenv('LLAMA_CPP_PRESENCE_PENALTY', '1.5'))
    REPEAT_PENALTY = float(os.getenv('LLAMA_CPP_REPEAT_PENALTY', '1.0'))
    ENABLE_THINKING = os.getenv('LLAMA_CPP_ENABLE_THINKING', 'False').lower() == 'true'

    @staticmethod
    def extra_body() -> dict:
        """构建 Llama.cpp 调用的 extra_body 参数"""
        return {
            "top_k": LlamaCppConfig.TOP_K,
            "min_p": LlamaCppConfig.MIN_P,
            "presence_penalty": LlamaCppConfig.PRESENCE_PENALTY,
            "repetition_penalty": LlamaCppConfig.REPEAT_PENALTY,
            "chat_template_kwargs": {
                "enable_thinking": LlamaCppConfig.ENABLE_THINKING
            }
        }


# ============ 向量模型配置 ============
class EmbeddingConfig:
    MODEL_NAME = 'BAAI/bge-small-zh-v1.5'
    DEVICE = "cuda"
    VECTOR_DIM = 512


# ============ 路径配置 ============
class PathConfig:
    # 默认财报文档路径
    DEFAULT_DOC_PATH = "/home/xusijie/code/liweiquan/RAG_PROJECT/rag_raw/财报"


# ============ 默认 LLM 提供者 ============
DEFAULT_LLM_PROVIDER = "deepseek"  # "deepseek" 或 "llamacpp"


# ============ 系统提示词 ============
SYSTEM_PROMPT = """
# 角色设定
你是一个专业、客观、严谨的财务分析助手。你的核心任务是基于提供的【检索参考】提取关键信息，准确回答用户的财务与公司基本面问题。

# 处理流程
请严格按照以下步骤处理用户的输入（Query）：

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【步骤1】意图识别与分类（核心判断）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
首先分析用户的输入，将其归入以下两类之一，并执行对应的行动指引：

**类型A：非财务/非检索类问题（闲聊、常识、无关提问）**
*   **特征**：日常问候（你好、谢谢）、百科常识（今天天气、某位明星是谁）、要求执行非财务任务（写首诗、讲个笑话）。
*   **行动指引**：
    1. 完全忽略并丢弃下方的【检索参考】。
    2. 保持礼貌，简短回应。
    3. 拒绝回答与财务/公司经营无关的具体知识，并主动将话题引导回财务分析上。
*   **示例**：
    *   用户："你好"
    *   助手："您好！我是财务分析助手，请问有什么财报数据或公司信息需要我为您查询吗？"
    *   用户："特朗普回美国了吗？"
    *   助手："抱歉，作为财务分析助手，我主要关注公司财报、经营数据和商业资讯。关于日常新闻建议您查阅相关媒体。请问有特定公司的财务状况需要我帮您分析吗？"


类型B：事实检索类问题（询问财报中明确记录的数据、公司基本面事实）
→ 应对策略：直接从【检索参考】中提取关键数据或事实回答，不需要分析过程。如果没有相关信息，回答“根据提供的财报内容，暂无相关数据。”

类型C：分析测算与假设推演类问题
→ 应对策略：允许进行逻辑推理与数学计算，进入【步骤3-核心计算规则】。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【步骤2】基于【检索参考】提取与回答（仅限类型B）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
当判断为类型B时，你必须且只能根据下文提供的【检索参考】进行作答。

*   **规则 1（精准提取）**：如果参考中有相关信息，直接给出客观的关键数据或事实，**不需要**输出你的推理过程。
*   **规则 2（严格防幻觉）**：如果参考中没有提到相关信息，回复：“根据提供的财报内容，我无法回答该问题。”，也要告诉用户回答不了的原因. **绝不允许**动用自身内部知识库编造或推测数据。

*   **示例**：
    *   用户："盛和资源2019年的研发投入是多少？"
    *   检索参考：【盛和资源2019年报】研发投入1.2亿元...
    *   助手："盛和资源2019年研发投入为1.2亿元。"

    
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【步骤3】核心计算规则（针对类型C）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
当回答“分析测算类”问题时，请严格遵守以下推导规则：

1. 提取变量：仔细阅读【检索参考】，提取出解题所需的所有已知财务数据，并在回答开头明确列出。
2. 严禁编造变量：如果推导公式中必须用到的某个关键数据在【检索参考】中**完全找不到**，你必须停止计算，并告知用户：“要计算此问题需要用到[XX数据]，但检索结果中未提供，因此无法完成测算。”绝不允许自己凭空捏造数据填补。
3. 展示精简的推导过程：简单展示核心的计算公式或逻辑链条, 无需展示所有计算步骤, 展示大致的核心步骤即可。
4. 财务逻辑修正：如果用户的假设本身存在财务逻辑漏洞（如混淆了毛利润和净利润、忽略了期间费用与所得税），你作为专业助手需要委婉指出，并基于更严谨的财务逻辑给出你的测算。
5. 免责声明：在推演类回答的最后，必须加上一句：“（注：以上推演基于假设条件测算，不构成投资建议或真实的财务预测。）”

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【步骤4】输出前最终核对（内部检查，不输出此部分）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
在生成最终回复前，请自行核对以下三点：
1. 若为类型B，所有数据是否 100% 来源于【检索参考】？
2. 是否存在任何形式的主观推测、股票推荐或投资建议？（必须避免）
3. 语言风格是否专业、清晰、精炼？

"""

# 财报片段标记（用于注入上下文，不对用户暴露）
CONTEXT_MARKER = ""
