import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.services.chat_service import chat_service
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# FastAPI 앱 생성
app = FastAPI(
    title="조선실록톡 ML Service",
    description="LangChain 기반 RAG 채팅 API",
    version="1.0.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React
        "http://localhost:8080",  # Spring Boot
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "service": "FastAPI ML Service",
        "version": "1.0.0",
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    RAG 기반 채팅 API

    - Spring Boot 서버에서 사용자 채팅 요청을 받아 처리
    - LangChain으로 RAG 파이프라인 실행
    - 응답과 출처를 반환
    """
    try:
        response_content, sources, response_keywords = chat_service.chat(
            user_message=request.message,
            persona_system_prompt=request.persona_system_prompt,
            persona_id=request.persona_id,
            chat_history=request.history,
            chunk_overlap=request.chunk_overlap,
            similarity_cutoff=request.similarity_cutoff,
            top_k=request.top_k,
            keywords=request.keywords,
            category=request.category,
            keyword_weight=request.keyword_weight,
        )
        return ChatResponse(content=response_content, sources=sources, keywords=response_keywords)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"채팅 처리 중 오류 발생: {str(e)}",
        )


@app.get("/")
async def root():
    return {
        "message": "조선실록톡 FastAPI ML Service",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
    )
