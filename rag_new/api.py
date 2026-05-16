"""
RAG财报问答系统 - FastAPI接口
"""
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional, List, Dict, Any
from openai import OpenAI
import os

from rag import (
    build_vector_db,
    hybrid_search,
    dense_search,
    sparse_search,
    get_doc_count,
    router,
    rewrite,
    intent,
    rewrite_for_retrieval,
    resolve_placeholders,
)
from rag.config import LLMConfig, PathConfig, SYSTEM_PROMPT

# ============ FastAPI 应用 ============
app = FastAPI(
    title="RAG财报问答系统",
    description="基于混合检索的财报问答API",
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 允许所有来源（开发环境）
    allow_credentials=True,
    allow_methods=["*"],           # 允许所有方法
    allow_headers=["*"],           # 允许所有头
)

# 添加详细的验证错误处理
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

# ============ 全局变量 ============
model = None
client = None


class ChatMessage(BaseModel):
    # 使用 Literal 限制只能传这三种字符串
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[ChatMessage]] = None   # 可选，不传就是新对话
    mode: str = "hybrid"      # dense/sparse/hybrid
    top_k: int = 10
    debug: bool = False

class ChatResponse(BaseModel):
    answer: str
    history: List[ChatMessage]    # 返回更新后的历史
    debug: Optional[Dict[str, Any]] = None

class StatusResponse(BaseModel):
    status: str
    doc_count: int
    model_loaded: bool


# ============ 初始化 ============
@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    global model, client

    # 初始化向量数据库
    model = build_vector_db(PathConfig.DEFAULT_DOC_PATH, rebuild=False)

    # 初始化LLM客户端
    client = OpenAI(
        api_key=LLMConfig.API_KEY,
        base_url=LLMConfig.BASE_URL
    )

    print(f"[启动完成] 知识库文档数: {get_doc_count()}")


# ============ 接口实现 ============
@app.get("/", response_model=StatusResponse)
async def root():
    """系统状态"""
    return StatusResponse(
        status="running",
        doc_count=get_doc_count(),
        model_loaded=model is not None
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    主对话接口

    流程：意图识别 → 查询改写 → 智能路由 → 检索优化 → 混合检索 → LLM生成
    """
    debug_info = {} if request.debug else None
    original_query = request.query
    current_query = original_query

    # ============ 处理历史 ============
    # 如果没传history，默认为空列表（新对话）
    if request.history:
        # 使用 model_dump() 确保正确获取值（Pydantic v2）
        history = [msg.model_dump() for msg in request.history]
    else:
        history = []

    # ============ 1. 意图识别 ============
    user_intent = intent(current_query)  # intent 不需要 history

    if request.debug:
        debug_info["intent"] = user_intent

    # 闲聊直接返回
    if user_intent == "chat":
        # 构建消息（包含历史）
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": current_query})

        response = client.chat.completions.create(
            model=LLMConfig.MODEL,
            extra_body={"thinking": {"type": "disabled"}},
            messages=messages,
            temperature=0.7
        )
        answer = response.choices[0].message.content

        # 更新历史
        updated_history = history.copy()
        updated_history.append({"role": "user", "content": current_query})
        updated_history.append({"role": "assistant", "content": answer})

        if request.debug:
            debug_info["rewrite_steps"] = None
            debug_info["router_result"] = None
            debug_info["retrieved_chunks"] = []

        return ChatResponse(
            answer=answer,
            history=[ChatMessage(**m) for m in updated_history],
            debug=debug_info
        )

    # ============ 2. 查询改写（补全省略/代词）=============
    print(f"[DEBUG] 输入问题: {current_query}")
    print(f"[DEBUG] 历史轮数: {len(history)}")
    if history:
        print(f"[DEBUG] 上一轮: {history[-1] if len(history) > 0 else '无'}")
    query_complete = rewrite(current_query, history)
    print(f"[DEBUG] 改写后: {query_complete}")

    # ============ 3. 智能路由 ============
    target_docs = router(query_complete)

    # ============ 4. 检索优化（精简query）=============
    query_optimized = rewrite_for_retrieval(query_complete)

    if request.debug:
        debug_info["rewrite_steps"] = {
            "original": original_query,
            "step1_complete": query_complete,
            "step2_optimized": query_optimized
        }
        debug_info["router_result"] = target_docs if target_docs != [None] else ["全部文档"]

    # ============ 5. 混合检索 ============
    all_results = []

    if target_docs == [None]:
        # 检索全部文档
        all_results = _search(query_optimized, request.mode, request.top_k * 2, None)
    elif len(target_docs) == 1:
        # 单个文档
        all_results = _search(query_optimized, request.mode, request.top_k * 2, target_docs[0])
    else:
        # 多个文档：分别检索后合并
        for doc in target_docs:
            results = _search(query_optimized, request.mode, request.top_k, doc)
            all_results.extend(results)

        # 按分数重新排序
        all_results.sort(key=lambda x: x[2], reverse=True)

    # ============ 6. 解析占位符 ============
    all_results = resolve_placeholders(all_results)

    # ============ 7. 构建上下文 ============
    retrieved_contexts = []
    for _, text, _, source in all_results:
        clean_source = source.replace('.md', '')
        context_block = f"【来源文件：{clean_source}】\n{text}"
        retrieved_contexts.append(context_block)

    context_text = "\n\n---\n\n".join(retrieved_contexts)

    # ============ 8. LLM生成 ============
    # 构建消息（包含历史）
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 添加历史对话（去掉旧的财报片段system消息）
    for msg in history:
        if not (msg.get('role') == 'system' and '【本轮财报片段】' in msg.get('content', '')):
            messages.append(msg)

    # 添加新的财报片段和用户问题
    messages.append({"role": "system", "content": f"【本轮财报片段】\n{context_text}"})
    messages.append({"role": "user", "content": query_optimized})

    response = client.chat.completions.create(
        model=LLMConfig.MODEL,
        extra_body={"thinking": {"type": "disabled"}},
        messages=messages,
        temperature=0.1
    )
    answer = response.choices[0].message.content

    # ============ 9. 更新历史 ============
    updated_history = history.copy()
    updated_history.append({"role": "user", "content": current_query})
    updated_history.append({"role": "assistant", "content": answer})

    # ============ 10. 组装debug信息 ============
    if request.debug:
        # 获取向量相似度（用于hybrid模式）
        dense_scores = {}
        if request.mode in ["hybrid", "dense"]:
            dense_results = dense_search(query_optimized, top_k=10, source_filter=None if target_docs == [None] else (target_docs[0] if len(target_docs) == 1 else None))
            dense_scores = {doc_id: 1 / (1 + score) for doc_id, _, score, _ in dense_results}

        debug_chunks = []
        for doc_id, text, rrf_score, source in all_results[:request.top_k]:
            chunk_info = {
                "chunk_id": doc_id,
                "rrf_score": round(rrf_score, 4),
                "source": source,
                "content": text[:200] + "..." if len(text) > 200 else text
            }
            # 添加向量相似度
            if doc_id in dense_scores:
                chunk_info["vector_similarity"] = round(dense_scores[doc_id], 4)

            debug_chunks.append(chunk_info)

        debug_info["retrieved_chunks"] = debug_chunks

    return ChatResponse(
        answer=answer,
        history=[ChatMessage(**m) for m in updated_history],
        debug=debug_info
    )


def _search(query: str, mode: str, top_k: int, source_filter: str = None):
    """内部检索函数"""
    if mode == "dense":
        return dense_search(query, top_k=top_k, source_filter=source_filter)
    elif mode == "sparse":
        return sparse_search(query, top_k=top_k, source_filter=source_filter)
    else:  # hybrid
        return hybrid_search(query, top_k=top_k, source_filter=source_filter)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
