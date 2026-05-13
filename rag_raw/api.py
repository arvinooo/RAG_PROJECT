import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from vector import build_vector_db, hybrid_search, get_doc_count

load_dotenv()

# 初始化向量数据库
EMBED_MODEL = build_vector_db(r"D:\github\rag_project\rag_财报\财报\财报.md")

key = os.getenv('ZHIPU_API_KEY')
client = OpenAI(api_key=key, base_url=os.getenv('ZHIPU_API_BASE'))

app = FastAPI(title="财报知识库问答系统", description="基于混合检索的财报问答 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    top_k: int = 3


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]


@app.get("/")
def read_root():
    doc_count = get_doc_count()
    return {
        "message": "财报知识库问答系统（混合检索：向量 + BM25）",
        "chunks_count": doc_count,
        "status": "ready" if doc_count > 0 else "知识库为空"
    }


@app.post("/chat", response_model=ChatResponse)
def chat_with_report(request: ChatRequest):
    """接收用户提问，使用混合检索（向量+BM25+RRF）获取相关片段，并调用大模型生成回答"""
    user_query = request.query

    # --- 第一步：混合检索 (Hybrid Retrieval) ---
    # hybrid_search 返回: [(doc_id, chunk_text, rrf_score), ...]
    search_results = hybrid_search(user_query, top_k=request.top_k)

    if not search_results:
        raise HTTPException(status_code=404, detail="未找到相关内容")

    # 提取文本片段
    retrieved_contexts = [text for _, text, _ in search_results]

    # 用 --- 分隔多个片段
    context_text = "\n\n---\n\n".join(retrieved_contexts)

    # --- 第二步：构建 Prompt ---
    system_prompt = f"""你是一个专业的财务分析助手。
请**仅根据**以下提供的财报片段来回答用户的问题。
如果提供的片段中没有相关信息，请直接回答"根据提供的财报内容，我无法回答该问题"，绝不要编造数据。
同时你也可以自行判断用户的问题, 先判断该问题是闲聊还是需要检索, 如果是跟你闲聊, 那你可以不按照知识库的内容回答, 同时可以不检索知识库, 直接按你的理解回答就好了, 如果用户的问题是需要到知识库检索的, 那么你必须严格按照知识库的内容回答, 不得胡编乱造.
【参考财报片段】：
{context_text}
"""

    # --- 第三步：生成回答 (Generation) ---
    response = client.chat.completions.create(
        model=os.getenv('ZHIPU_MODEL'),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.1
    )

    final_answer = response.choices[0].message.content
    return ChatResponse(answer=final_answer, sources=retrieved_contexts)


if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("财报知识库问答系统")
    print("检索方式: 混合检索（BGE向量 + BM25 + RRF）")
    print(f"当前知识库文本块数: {get_doc_count()}")
    print("=" * 50)
    uvicorn.run(app, host="127.0.0.1", port=8080)
