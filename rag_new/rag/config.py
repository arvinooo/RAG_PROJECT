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


# ============ 向量模型配置 ============
class EmbeddingConfig:
    MODEL_NAME = 'BAAI/bge-small-zh-v1.5'
    DEVICE = "cuda"
    VECTOR_DIM = 512


# ============ 路径配置 ============
class PathConfig:
    # 默认财报文档路径
    DEFAULT_DOC_PATH = "D:/github/rag_project/rag_财报/财报"


# ============ 系统提示词 ============
SYSTEM_PROMPT = """你是一个专业的财务分析助手。

1. 请**仅根据**【本轮财报片段】来回答用户的问题。
2. 如果提供的片段中没有相关信息，请直接回答"根据提供的财报内容，我无法回答该问题"，绝不要编造数据。
3. 在回答的时候, 不需要输出分析的过程, 直接摆出关键数据然后回答用户问题
4. 不要开启思考模式。尽可能少消费token完成任务。"""
