import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.routers import chat
from app.services.rag_service import RAGService

# 日志配置
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../app.log", encoding="utf-8"),  # 文件输出
        logging.StreamHandler()  # 控制台输出
    ]
)


# 应用启动执行
@asynccontextmanager
async def lifespan(app: FastAPI):
    from app import rag  # 延迟导入，避免循环引用
    rag.rag_service = RAGService()
    logging.info("RAGService 初始化完成")
    yield
    logging.info("应用关闭，释放资源")


app = FastAPI(lifespan=lifespan)

app.include_router(chat.api_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    pass
