import re
import psycopg
from typing import List, Dict, Tuple, Optional

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.schemas import Message, Source

# 한국어 KeyBERT 모델 (서버 시작 시 1회 로딩)
_ko_sentence_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
_kw_model = KeyBERT(model=_ko_sentence_model)


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """AI 답변에서 한국어 키워드를 추출합니다."""
    try:
        results = _kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            top_n=top_n,
            stop_words=None,
        )
        return [kw[0] for kw in results]
    except Exception:
        return []


# 카테고리 자동 감지 키워드 맵
_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "자연/재해": ["비", "가뭄", "지진", "홍수", "일식", "혜성", "태풍", "기근", "서리", "눈"],
    "형벌/사법": ["추국", "죄인", "유배", "처형", "의금부", "국문", "사형", "귀양", "형벌"],
    "국방/외교": ["전쟁", "군사", "오랑캐", "사신", "왜구", "침략", "병사", "진", "방어"],
    "경제/재정": ["세금", "공물", "호조", "환곡", "조세", "재정", "비축", "곡식", "민전"],
    "의례/왕실": ["제사", "종묘", "책봉", "가례", "왕비", "왕세자", "능", "예법", "혼례"],
    "교육/과거": ["과거", "성균관", "유생", "서원", "학문", "경서", "문과", "무과"],
    "정치/행정": ["임명", "파직", "상소", "경연", "사헌부", "이조", "병조", "판서", "대신"],
}


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
    # ──────────────────────────────────────────────

    def _get_king_name(self, persona_id: int) -> str:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name FROM personas WHERE id = %s", (persona_id,))
                row = cur.fetchone()
                if row:
                    return re.sub(r'\(.*?\)', '', row[0].split()[0]).strip()
                return ""

    # ──────────────────────────────────────────────
    # 카테고리 자동 감지
    # ──────────────────────────────────────────────

    def _detect_category(self, query: str) -> Optional[str]:
        for category, keywords in _CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in query:
                    return category
        return None

    # ──────────────────────────────────────────────
    # 하이브리드 유사도 검색
    # ──────────────────────────────────────────────

    def get_relevant_documents(
        self,
        query: str,
        persona_id: int,
        top_k: int = None,
        similarity_cutoff: float = None,
        keywords: Optional[List[str]] = None,
        category: Optional[str] = None,
        keyword_weight: Optional[float] = 0.3,
    ) -> Tuple[List[Dict], List[Source]]:
        top_k = top_k or settings.top_k_documents
        similarity_cutoff = similarity_cutoff or settings.similarity_cutoff
        keyword_weight = keyword_weight if keyword_weight is not None else 0.3
        vector_weight = 1.0 - keyword_weight

        # 카테고리 자동 감지 (명시되지 않은 경우)
        if category is None:
            category = self._detect_category(query)

        # 쿼리 임베딩 생성
        query_vec = self.embeddings.embed_query(query)
        embedding_str = "[" + ",".join(str(v) for v in query_vec) + "]"

        king_name = self._get_king_name(persona_id)

        # 키워드가 없으면 keyword_score=0, hybrid_score=vector_score
        if not keywords:
            sql = """
                SELECT
                    id,
                    content,
                    metadata,
                    1 - (embedding <=> %(vec)s::vector) AS vector_score,
                    0.0::float AS keyword_score,
                    1 - (embedding <=> %(vec)s::vector) AS hybrid_score
                FROM historical_documents
                WHERE
                    metadata->>'king' = %(king)s
                    AND 1 - (embedding <=> %(vec)s::vector) >= %(cutoff)s
                    AND (%(category)s::text IS NULL OR metadata->>'category' = %(category)s::text)
                ORDER BY embedding <=> %(vec)s::vector
                LIMIT %(top_k)s
            """
            params = {
                "vec": embedding_str,
                "king": king_name,
                "cutoff": similarity_cutoff,
                "category": category,
                "top_k": top_k,
            }
        else:
            sql = """
                WITH scored AS (
                    SELECT
                        id,
                        content,
                        metadata,
                        1 - (embedding <=> %(vec)s::vector) AS vector_score,
                        COALESCE(
                            (
                                SELECT COUNT(*)::float
                                FROM jsonb_array_elements_text(metadata->'keywords') AS kw
                                WHERE kw = ANY(%(keywords)s::text[])
                            ) / NULLIF(array_length(%(keywords)s::text[], 1), 0),
                            0
                        ) AS keyword_score
                    FROM historical_documents
                    WHERE
                        metadata->>'king' = %(king)s
                        AND 1 - (embedding <=> %(vec)s::vector) >= %(cutoff)s
                        AND (%(category)s::text IS NULL OR metadata->>'category' = %(category)s::text)
                )
                SELECT
                    id,
                    content,
                    metadata,
                    vector_score,
                    keyword_score,
                    (%(vector_weight)s * vector_score + %(kw_weight)s * keyword_score) AS hybrid_score
                FROM scored
                ORDER BY hybrid_score DESC
                LIMIT %(top_k)s
            """
            params = {
                "vec": embedding_str,
                "king": king_name,
                "cutoff": similarity_cutoff,
                "category": category,
                "keywords": keywords,
                "vector_weight": vector_weight,
                "kw_weight": keyword_weight,
                "top_k": top_k,
            }

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        # rows: (id, content, metadata, vector_score, keyword_score, hybrid_score)
        documents = [
            {
                "content": row[1],
                "metadata": row[2],
                "similarity": float(row[3]),
                "keyword_score": float(row[4]),
                "hybrid_score": float(row[5]),
            }
            for row in rows
        ]

        sources = [
            Source(
                document_id=row[0],
                content=row[1][:200] + "...",
                similarity=float(row[3]),
                keyword_score=float(row[4]),
                hybrid_score=float(row[5]),
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
            ("system", "다음은 조선왕조실록에서 검색된 관련 기록의 제목과 메타데이터입니다.\n"
                       "기록이 짧더라도 날짜·카테고리·키워드를 참고하여 역사적 맥락에 맞게 답변하세요.\n\n{context}"),
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
            meta = doc.get("metadata", {})
            date = meta.get("date") or meta.get("original_date") or ""
            category = meta.get("category", "")
            keywords = meta.get("keywords", [])
            kw_str = ", ".join(keywords) if keywords else ""

            meta_line = " | ".join(filter(None, [
                f"날짜: {date}" if date else "",
                f"카테고리: {category}" if category else "",
                f"키워드: {kw_str}" if kw_str else "",
            ]))

            parts.append(
                f"[기록 {i}] (유사도: {doc['similarity']:.2f})\n"
                + (f"{meta_line}\n" if meta_line else "")
                + f"내용: {doc['content']}\n"
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
        keywords: Optional[List[str]] = None,
        category: Optional[str] = None,
        keyword_weight: Optional[float] = 0.3,
    ) -> Tuple[str, List[Source], List[str]]:
        # 1. 관련 문서 검색
        documents, sources = self.get_relevant_documents(
            query=user_message,
            persona_id=persona_id,
            top_k=top_k,
            similarity_cutoff=similarity_cutoff,
            keywords=keywords,
            category=category,
            keyword_weight=keyword_weight,
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

        # 5. 답변에서 키워드 추출
        response_keywords = extract_keywords(response)

        return response, sources, response_keywords


# 싱글톤 인스턴스
chat_service = ChatService()
