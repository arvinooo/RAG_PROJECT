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
    MODEL_NAME = '/home/xusijie/.cache/modelscope/hub/models/AI-ModelScope/bge-small-zh-v1.5'
    DEVICE = "cpu"
    VECTOR_DIM = 512


# ============ 路径配置 ============
class PathConfig:
    # 默认财报文档路径
    DEFAULT_DOC_PATH = "/home/xusijie/code/liweiquan/RAG_PROJECT/rag_raw/财报"


# ============ 默认 LLM 提供者 ============
DEFAULT_LLM_PROVIDER = "llamacpp"  # "deepseek" 或 "llamacpp"


# ============ 系统提示词 ============
SYSTEM_PROMPT = """你是一个专业的财务分析助手。

1. 请**仅根据**【本轮财报片段】来回答用户的问题。
2. 如果提供的片段中没有相关信息，请直接回答"根据提供的财报内容，我无法回答该问题"，绝不要编造数据。
3. 在回答的时候, 不需要输出分析的过程, 直接摆出关键数据然后回答用户问题
4. 不要开启思考模式。尽可能少消费token完成任务。"""
