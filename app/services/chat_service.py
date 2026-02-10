import psycopg
from typing import List, Dict, Tuple

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.schemas import Message, Source


class ChatService:
    """LangChain 기반 RAG 채팅 서비스 (historical_documents 직접 쿼리)"""

    def __init__(self):
        # OpenAI LLM
        self.llm = ChatOpenAI(
            model=settings.openai_chat_model,
            temperature=0.7,
            openai_api_key=settings.openai_api_key,
        )

        # OpenAI Embeddings (DB 저장 차원: 1536)
        self.embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )

    # ──────────────────────────────────────────────
    # DB 연결
    # ──────────────────────────────────────────────

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(
            host=settings.db_host,
            port=settings.db_port,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
        )

    # ──────────────────────────────────────────────
    # 페르소나 → 왕 이름 조회
    # historical_documents.metadata->>'king' 필터로 사용
    # 예) persona.name='태조 이성계' → king='태조'
    #     persona.name='정종'       → king='정종'
    # ──────────────────────────────────────────────

    def _get_king_name(self, persona_id: int) -> str:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name FROM personas WHERE id = %s", (persona_id,))
                row = cur.fetchone()
                if row:
                    import re
                    return re.sub(r'\(.*?\)', '', row[0].split()[0]).strip()  # '정조(조선)' → '정조'
                return ""

    # ──────────────────────────────────────────────
    # 유사도 검색 (pgvector 직접 쿼리)
    # ──────────────────────────────────────────────

    def get_relevant_documents(
        self,
        query: str,
        persona_id: int,
        top_k: int = None,
        similarity_cutoff: float = None,
    ) -> Tuple[List[Dict], List[Source]]:
        top_k = top_k or settings.top_k_documents
        similarity_cutoff = similarity_cutoff or settings.similarity_cutoff

        # 쿼리 임베딩 생성
        query_vec = self.embeddings.embed_query(query)
        embedding_str = "[" + ",".join(str(v) for v in query_vec) + "]"

        # 페르소나에 해당하는 왕 이름
        king_name = self._get_king_name(persona_id)

        sql = """
            SELECT
                id,
                content,
                metadata,
                1 - (embedding <=> %(vec)s::vector) AS similarity
            FROM historical_documents
            WHERE
                metadata->>'king' = %(king)s
                AND 1 - (embedding <=> %(vec)s::vector) >= %(cutoff)s
            ORDER BY embedding <=> %(vec)s::vector
            LIMIT %(top_k)s
        """

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {
                    "vec": embedding_str,
                    "king": king_name,
                    "cutoff": similarity_cutoff,
                    "top_k": top_k,
                })
                rows = cur.fetchall()

        documents = [
            {
                "content": row[1],
                "metadata": row[2],
                "similarity": float(row[3]),
            }
            for row in rows
        ]

        sources = [
            Source(
                document_id=row[0],
                content=row[1][:200] + "...",
                similarity=float(row[3]),
            )
            for row in rows
        ]

        return documents, sources

    # ──────────────────────────────────────────────
    # RAG 체인 생성
    # ──────────────────────────────────────────────

    def create_rag_chain(self, persona_system_prompt: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", persona_system_prompt),
            ("system", "다음은 조선왕조실록의 관련 기록입니다:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        return (
            {
                "context": lambda x: x["context"],
                "chat_history": lambda x: x["chat_history"],
                "question": lambda x: x["question"],
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    # ──────────────────────────────────────────────
    # 헬퍼: 포맷팅
    # ──────────────────────────────────────────────

    def format_context(self, documents: List[Dict]) -> str:
        parts = []
        for i, doc in enumerate(documents, 1):
            parts.append(
                f"[문서 {i}] (유사도: {doc['similarity']:.2f})\n{doc['content']}\n"
            )
        return "\n".join(parts)

    def format_chat_history(self, history: List[Message]) -> List:
        messages = []
        for msg in history[-10:]:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
        return messages

    # ──────────────────────────────────────────────
    # 메인 채팅 메서드
    # ──────────────────────────────────────────────

    def chat(
        self,
        user_message: str,
        persona_system_prompt: str,
        persona_id: int,
        chat_history: List[Message] = None,
        chunk_overlap: int = None,
        similarity_cutoff: float = None,
        top_k: int = None,
    ) -> Tuple[str, List[Source]]:
        # 1. 관련 문서 검색
        documents, sources = self.get_relevant_documents(
            query=user_message,
            persona_id=persona_id,
            top_k=top_k,
            similarity_cutoff=similarity_cutoff,
        )

        # 2. 컨텍스트 구성
        context = self.format_context(documents)

        # 3. 히스토리 포맷팅
        formatted_history = self.format_chat_history(chat_history or [])

        # 4. RAG 체인 실행
        rag_chain = self.create_rag_chain(persona_system_prompt)
        response = rag_chain.invoke({
            "context": context,
            "chat_history": formatted_history,
            "question": user_message,
        })

        return response, sources


# 싱글톤 인스턴스
chat_service = ChatService()
