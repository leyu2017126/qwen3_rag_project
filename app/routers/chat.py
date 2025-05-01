from fastapi import APIRouter, Depends

from app.schemas.user_input import UserInput
from app.services.qwen_chatbot import QwenChatbot
from app.rag import get_rag_service
from app.services.rag_service import RAGService

api_router = APIRouter(tags=["Qwen3"])


@api_router.post("/chat")
async def chat(query: UserInput, rag: RAGService = Depends(get_rag_service)):
    chatbot = QwenChatbot("../models/qwen3")
    response = chatbot.generate_response(user_input=query.query, rag=rag)

    return response
