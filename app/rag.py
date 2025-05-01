# app/rag.py
from app.services.rag_service import RAGService

# 全局实例
rag_service: RAGService = None


def get_rag_service() -> RAGService:
    return rag_service
