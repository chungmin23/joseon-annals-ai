from pydantic import BaseModel, Field
from typing import List, Optional


class Message(BaseModel):
    """채팅 메시지"""
    role: str = Field(..., description="user 또는 assistant")
    content: str = Field(..., description="메시지 내용")


class ChatRequest(BaseModel):
    """채팅 요청"""
    room_id: str = Field(..., description="채팅방 ID")
    persona_id: int = Field(..., description="페르소나 ID")
    persona_system_prompt: str = Field(..., description="페르소나 시스템 프롬프트")
    message: str = Field(..., description="사용자 질문")
    history: Optional[List[Message]] = Field(
        default=[],
        description="대화 히스토리 (최근 10턴)",
    )

    # RAG 파라미터 (선택적 조절)
    chunk_overlap: Optional[int] = Field(default=None, description="청크 오버랩")
    similarity_cutoff: Optional[float] = Field(default=None, description="유사도 컷오프")
    top_k: Optional[int] = Field(default=None, description="검색할 문서 수")


class Source(BaseModel):
    """출처 정보"""
    document_id: int
    content: str = Field(..., description="문서 내용 일부")
    similarity: float = Field(..., description="유사도 점수")


class ChatResponse(BaseModel):
    """채팅 응답"""
    content: str = Field(..., description="AI 응답 내용")
    sources: List[Source] = Field(..., description="출처 문서 목록")


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    service: str
    version: str
