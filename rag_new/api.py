"""
RAG财报问答系统 - FastAPI接口
"""
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional, List, Dict, Any
from openai import OpenAI, AsyncOpenAI
import os
import json
import time
import traceback

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
from rag.config import LLMConfig, LlamaCppConfig, PathConfig, SYSTEM_PROMPT, DEFAULT_LLM_PROVIDER

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
client = None          # 同步客户端（非流式用）
async_client = None    # 异步客户端（流式用）
llm_provider = DEFAULT_LLM_PROVIDER


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
    stream: bool = False       # 是否流式输出

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
    global model, client, async_client

    # 初始化向量数据库
    model = build_vector_db(PathConfig.DEFAULT_DOC_PATH, rebuild=False)

    # 初始化LLM客户端（同步 + 异步）
    import os
    if llm_provider == "llamacpp":
        # Llama.cpp 走内网，绕过代理
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        os.environ["no_proxy"] = "*"
        client = OpenAI(
            api_key=LlamaCppConfig.API_KEY,
            base_url=LlamaCppConfig.API_BASE
        )
        async_client = AsyncOpenAI(
            api_key=LlamaCppConfig.API_KEY,
            base_url=LlamaCppConfig.API_BASE
        )
    else:
        client = OpenAI(
            api_key=LLMConfig.API_KEY,
            base_url=LLMConfig.BASE_URL
        )
        async_client = AsyncOpenAI(
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


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    主对话接口

    流程：意图识别 → 查询改写 → 智能路由 → 检索优化 → 混合检索 → LLM生成
    支持 stream=True 流式输出
    每个阶段记录耗时并通过 SSE 推送进度
    """
    debug_info = {}
    original_query = request.query
    current_query = original_query
    request_start = time.perf_counter()

    # ============ 阶段计时收集 ============
    pipeline_stages = []  # [{"name", "status", "elapsed_ms", "detail"}]

    def _record_stage(name: str, status: str, start: float, detail: str = ""):
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        pipeline_stages.append({
            "name": name,
            "status": status,  # "done" | "skipped"
            "elapsed_ms": elapsed_ms,
            "detail": detail
        })
        print(f"[PIPELINE] {name}: {elapsed_ms}ms {detail}")

    # ============ 处理历史 ============
    if request.history:
        history = [msg.model_dump() for msg in request.history]
    else:
        history = []

    # ============ 1. 意图识别 ============
    t0 = time.perf_counter()
    user_intent = intent(current_query)
    _record_stage("意图识别", "done", t0, user_intent)
    debug_info["intent"] = user_intent

    # ============ 2. 查询改写（补全省略/代词）=============
    t0 = time.perf_counter()
    print(f"[DEBUG] 输入问题: {current_query}")
    print(f"[DEBUG] 历史轮数: {len(history)}")
    if history:
        print(f"[DEBUG] 上一轮: {history[-1] if len(history) > 0 else '无'}")
    query_complete = rewrite(current_query, history)
    _record_stage("查询改写", "done", t0, f"{current_query[:20]}... → {query_complete[:20]}...")
    print(f"[DEBUG] 改写后: {query_complete}")

    # ============ 3. 智能路由 ============
    t0 = time.perf_counter()
    target_docs = router(query_complete)
    docs_display = target_docs if target_docs != [None] else ["全部文档"]
    _record_stage("智能路由", "done", t0, str(docs_display))

    # ============ 4. 检索优化（精简query）=============
    t0 = time.perf_counter()
    query_optimized = rewrite_for_retrieval(query_complete)
    _record_stage("检索优化", "done", t0, f"{query_complete[:20]}... → {query_optimized[:20]}...")

    debug_info["rewrite_steps"] = {
        "original": original_query,
        "step1_complete": query_complete,
        "step2_optimized": query_optimized
    }
    debug_info["router_result"] = docs_display

    # ============ 5. 混合检索 ============
    t0 = time.perf_counter()
    all_results = []

    if target_docs == [None]:
        all_results = _search(query_optimized, request.mode, request.top_k * 2, None)
    elif len(target_docs) == 1:
        all_results = _search(query_optimized, request.mode, request.top_k * 2, target_docs[0])
    else:
        for doc in target_docs:
            results = _search(query_optimized, request.mode, request.top_k, doc)
            all_results.extend(results)
        all_results.sort(key=lambda x: x[2], reverse=True)

    _record_stage("混合检索", "done", t0, f"召回 {len(all_results)} 个片段")

    # ============ 6. 解析占位符 ============
    t0 = time.perf_counter()
    all_results = resolve_placeholders(all_results)
    _record_stage("占位符解析", "done", t0, f"还原表格占位符")

    # ============ 7. 构建上下文 ============
    t0 = time.perf_counter()
    retrieved_contexts = []
    for _, text, _, source in all_results:
        clean_source = source.replace('.md', '')
        context_block = f"【来源文件：{clean_source}】\n{text}"
        retrieved_contexts.append(context_block)

    context_text = "\n\n---\n\n".join(retrieved_contexts)
    _record_stage("上下文构建", "done", t0, f"Context {len(context_text)} 字符")
    debug_info["context_text"] = context_text  # 保存上下文供前端展开显示

    # ============ 8. 构建 messages ============
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history:
        if not (msg.get('role') == 'system' and '【本轮财报片段】' in msg.get('content', '')):
            messages.append(msg)

    # 闲聊时不加财报片段
    if user_intent == "chat":
        messages.append({"role": "user", "content": current_query})
        final_query = current_query
        # 闲聊跳过检索相关阶段标记
        for s in pipeline_stages:
            if s["name"] in ("智能路由", "检索优化", "混合检索", "占位符解析", "上下文构建"):
                s["status"] = "skipped"
    else:
        messages.append({"role": "system", "content": f"【本轮财报片段】\n{context_text}"})
        messages.append({"role": "user", "content": query_optimized})
        final_query = current_query

    model_name = LlamaCppConfig.MODEL if llm_provider == "llamacpp" else LLMConfig.MODEL
    extra = LlamaCppConfig.extra_body() if llm_provider == "llamacpp" else {"thinking": {"type": "disabled"}}

    # ============ 流式输出 ============
    if request.stream:
        async def stream_generator():
            """SSE 流式生成器 - 使用 AsyncOpenAI 实现真正异步流式"""
            full_answer = ""
            gen_start = time.perf_counter()

            # --- 先推送所有已完成的阶段进度 ---
            for stage in pipeline_stages:
                yield f"event: progress\ndata: {json.dumps(stage, ensure_ascii=False)}\n\n"

            # --- 推送 LLM 生成开始 ---
            yield f"event: progress\ndata: {json.dumps({'name': 'LLM生成', 'status': 'running', 'elapsed_ms': 0, 'detail': '流式输出中...'}, ensure_ascii=False)}\n\n"

            try:
                stream = await async_client.chat.completions.create(
                    model=model_name,
                    extra_body=extra,
                    messages=messages,
                    temperature=0.1 if user_intent != "chat" else 0.7,
                    stream=True
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_answer += delta.content
                        yield f"event: message\ndata: {json.dumps(delta.content, ensure_ascii=False)}\n\n"

            except Exception as e:
                tb = traceback.format_exc()
                print(f"\n{'='*60}")
                print(f"[STREAM ERROR] {e}")
                print(f"[TRACEBACK]\n{tb}")
                print(f"{'='*60}\n")

                gen_elapsed_ms = round((time.perf_counter() - gen_start) * 1000, 1)
                total_elapsed_ms = round((time.perf_counter() - request_start) * 1000, 1)
                pipeline_stages.append({"name": "LLM生成", "status": "error", "elapsed_ms": gen_elapsed_ms, "detail": str(e)})
                pipeline_stages.append({"name": "总耗时", "status": "error", "elapsed_ms": total_elapsed_ms, "detail": ""})

                # 推送错误阶段进度
                for stage in pipeline_stages[-3:]:
                    yield f"event: progress\ndata: {json.dumps(stage, ensure_ascii=False)}\n\n"

                yield f"event: done\ndata: {json.dumps({'error': str(e), 'debug': {'pipeline_stages': pipeline_stages}}, ensure_ascii=False)}\n\n"
                return

            # --- LLM 生成完成 ---
            gen_elapsed_ms = round((time.perf_counter() - gen_start) * 1000, 1)
            total_elapsed_ms = round((time.perf_counter() - request_start) * 1000, 1)
            pipeline_stages.append({"name": "LLM生成", "status": "done", "elapsed_ms": gen_elapsed_ms, "detail": f"输出 {len(full_answer)} 字符"})
            pipeline_stages.append({"name": "总耗时", "status": "done", "elapsed_ms": total_elapsed_ms, "detail": ""})

            # 推送完成进度
            for stage in pipeline_stages[-2:]:
                yield f"event: progress\ndata: {json.dumps(stage, ensure_ascii=False)}\n\n"

            # 发送最终结果（含完整 answer 和 history）
            updated_history = history.copy()
            updated_history.append({"role": "user", "content": final_query})
            updated_history.append({"role": "assistant", "content": full_answer})

            # debug 信息
            dense_scores = {}
            if user_intent != "chat" and request.mode in ["hybrid", "dense"]:
                try:
                    dense_results = dense_search(query_optimized, top_k=10, source_filter=None if target_docs == [None] else (target_docs[0] if len(target_docs) == 1 else None))
                    dense_scores = {doc_id: 1 / (1 + score) for doc_id, _, score, _ in dense_results}
                except Exception as de:
                    print(f"[DEBUG WARNING] dense_search for debug info failed: {de}")

            debug_chunks = []
            for doc_id, text, rrf_score, source in all_results[:request.top_k]:
                chunk_info = {
                    "chunk_id": doc_id,
                    "rrf_score": round(rrf_score, 4),
                    "source": source,
                    "content": text[:200] + "..." if len(text) > 200 else text
                }
                if doc_id in dense_scores:
                    chunk_info["vector_similarity"] = round(dense_scores[doc_id], 4)
                debug_chunks.append(chunk_info)

            debug_info["retrieved_chunks"] = debug_chunks
            debug_info["pipeline_stages"] = pipeline_stages
            debug_info["total_elapsed_ms"] = total_elapsed_ms

            final_data = {
                "answer": full_answer,
                "history": [ChatMessage(**m).model_dump() for m in updated_history],
                "debug": debug_info
            }

            yield f"event: done\ndata: {json.dumps(final_data, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    # ============ 非流式（原有逻辑）=============
    gen_start = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        extra_body=extra,
        messages=messages,
        temperature=0.1 if user_intent != "chat" else 0.7
    )
    answer = response.choices[0].message.content
    gen_elapsed_ms = round((time.perf_counter() - gen_start) * 1000, 1)
    _record_stage("LLM生成", "done", gen_start, f"输出 {len(answer)} 字符")
    total_elapsed_ms = round((time.perf_counter() - request_start) * 1000, 1)

    updated_history = history.copy()
    updated_history.append({"role": "user", "content": final_query})
    updated_history.append({"role": "assistant", "content": answer})

    dense_scores = {}
    if user_intent != "chat" and request.mode in ["hybrid", "dense"]:
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
        if doc_id in dense_scores:
            chunk_info["vector_similarity"] = round(dense_scores[doc_id], 4)
        debug_chunks.append(chunk_info)
    debug_info["retrieved_chunks"] = debug_chunks
    debug_info["pipeline_stages"] = pipeline_stages
    debug_info["total_elapsed_ms"] = total_elapsed_ms

    return ChatResponse(
        answer=answer,
        history=[ChatMessage(**m) for m in updated_history],
        debug=debug_info if request.debug else None
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
