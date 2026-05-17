"""
RAG财报问答系统 - 统一入口

用法：
    python main.py                    # 启动API服务器
    python main.py --build            # 构建向量数据库
    python main.py --rebuild          # 重建向量数据库
"""
import argparse
import uvicorn

from rag.config import PathConfig
from rag import build_vector_db, get_doc_count


def main():
    parser = argparse.ArgumentParser(description="RAG财报问答系统")
    parser.add_argument("--build", action="store_true", help="构建向量数据库")
    parser.add_argument("--rebuild", action="store_true", help="重建向量数据库")
    parser.add_argument("--host", default="0.0.0.0", help="API服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="API服务器端口")

    args = parser.parse_args()

    # 构建或重建向量数据库
    if args.build or args.rebuild:
        print(f"正在{'重' if args.rebuild else ''}建向量数据库...")
        build_vector_db(PathConfig.DEFAULT_DOC_PATH, rebuild=args.rebuild)
        print(f"当前文档数: {get_doc_count()}")
        return

    # 启动API服务器
    print(f"启动API服务器: http://{args.host}:{args.port}")
    uvicorn.run("api:app", host=args.host, port=args.port, reload=True)


if __name__ == "__main__":
    main()
