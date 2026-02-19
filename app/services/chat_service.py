import os
import re
import psycopg
from typing import List, Dict, Tuple, Optional

from huggingface_hub import InferenceClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.schemas import Message, Source

# HuggingFace Korean NER 모델 (개체명 인식으로 키워드 추출)
_NER_MODEL = "Leo97/KoELECTRA-small-v3-modu-ner"
_hf_client: Optional[InferenceClient] = None

# 추출 대상 개체 유형 (인명/지명/기관/문화재/사건/문화)
_TARGET_ENTITIES = {"PS", "LC", "OG", "AF", "EV", "CV", "FD"}

# Kiwi 형태소 분석기 싱글톤
_kiwi_instance = None


def _get_hf_client() -> InferenceClient:
    global _hf_client
    if _hf_client is None:
        token = os.getenv("HF_TOKEN") or None
        _hf_client = InferenceClient(token=token)
    return _hf_client


def _get_kiwi():
    global _kiwi_instance
    if _kiwi_instance is None:
        from kiwipiepy import Kiwi
        _kiwi_instance = Kiwi()
    return _kiwi_instance


def _extract_keywords_kiwi(text: str, top_n: int = 5) -> List[str]:
    """Kiwi 형태소 분석으로 명사 키워드 추출 (NNG: 일반명사, NNP: 고유명사)"""
    kiwi = _get_kiwi()
    tokens = kiwi.tokenize(text[:512])
    noun_tags = {"NNG", "NNP"}
    freq: Dict[str, int] = {}
    for token in tokens:
        tag = token.tag.name if hasattr(token.tag, "name") else str(token.tag)
        if tag in noun_tags and len(token.form) >= 2:
            freq[token.form] = freq.get(token.form, 0) + 1
    sorted_nouns = sorted(freq.items(), key=lambda x: -x[1])
    return [word for word, _ in sorted_nouns[:top_n]]


def _extract_keywords_fallback(text: str, top_n: int = 5) -> List[str]:
    """Kiwi 형태소 분석 기반 키워드 추출 (HuggingFace NER 실패 시 폴백).
    Kiwi 실패 시 정규식/사전 기반 최후 폴백 사용."""
    # 1. Kiwi 형태소 분석 시도
    try:
        keywords = _extract_keywords_kiwi(text, top_n)
        if keywords:
            return keywords
    except Exception:
        pass

    # 2. 정규식/사전 기반 최후 폴백
    candidates: List[str] = []

    for kws in _CATEGORY_KEYWORDS.values():
        for kw in kws:
            if len(kw) >= 2 and kw in text:
                candidates.append(kw)

    extra_patterns = [
        r'[가-힣]{2,4}대왕',
        r'임진왜란|병자호란|정유재란|갑오개혁|을미사변|임오군란',
        r'의금부|사헌부|사간원|성균관|한성부|이조|병조|형조|호조|예조|공조',
        r'[가-힣]{1,3}(?:대군|공주|옹주)',
        r'[가-힣]{1,3}(?:왕후|왕비|왕세자)',
    ]
    for pattern in extra_patterns:
        matches = re.findall(pattern, text)
        candidates.extend(matches)

    seen: set = set()
    result: List[str] = []
    for kw in candidates:
        if kw not in seen:
            seen.add(kw)
            result.append(kw)

    return result[:top_n]


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """HuggingFace Korean NER API로 답변에서 핵심 개체명 키워드를 추출합니다.
    API 실패 또는 결과 없을 시 정규식 기반 폴백을 사용합니다."""
    import logging
    logger = logging.getLogger(__name__)
    ner_keywords: List[str] = []
    try:
        logger.info("[NER] 입력 텍스트 (앞 100자): %s", text[:100])
        client = _get_hf_client()
        results = client.token_classification(
            text[:512],
            model=_NER_MODEL,
        )
        logger.info("[NER] API 원결과 수: %d", len(results) if results else 0)

        seen: set = set()
        for item in results:
            entity_group = getattr(item, 'entity_group', None) or item.get('entity_group', '')
            word = getattr(item, 'word', None) or item.get('word', '')
            score = getattr(item, 'score', None) or item.get('score', 0)

            word = word.replace('##', '').strip()
            if (entity_group in _TARGET_ENTITIES
                    and score >= 0.7
                    and len(word) >= 2
                    and word not in seen):
                seen.add(word)
                ner_keywords.append(word)

        logger.info("[NER] 추출 키워드: %s", ner_keywords[:top_n])
    except Exception as e:
        logger.error("[NER] 예외 발생: %s", e, exc_info=True)

    if ner_keywords:
        return ner_keywords[:top_n]

    # NER 결과 없을 때 폴백
    fallback = _extract_keywords_fallback(text, top_n)
    logger.info("[NER] 폴백 키워드 사용: %s", fallback)
    return fallback


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
        import logging
        logger = logging.getLogger(__name__)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name FROM personas WHERE persona_id = %s", (persona_id,))
                row = cur.fetchone()
                if row:
                    raw_name = row[0]
                    name = re.sub(r'\(.*?\)', '', raw_name.split()[0]).strip()
                    name = re.sub(r'대왕$', '', name).strip()
                    logger.info("[RAG] persona_id=%s, DB name='%s' → king_name='%s'", persona_id, raw_name, name)
                    return name
                logger.warning("[RAG] persona_id=%s 페르소나를 찾을 수 없음", persona_id)
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

        import logging
        logger = logging.getLogger(__name__)
        logger.info("[RAG] 검색 시작 → king='%s', category='%s', keywords=%s, cutoff=%.2f, top_k=%d",
                    king_name, category, keywords, similarity_cutoff, top_k)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        logger.info("[RAG] 검색 결과=%d건", len(rows))
        if rows:
            for i, row in enumerate(rows[:3]):  # 상위 3건만 로그
                logger.info("[RAG] 결과[%d] doc_id=%s, vector=%.3f, keyword=%.3f, hybrid=%.3f | %s",
                            i, row[0], float(row[3]), float(row[4]), float(row[5]), str(row[1])[:60])
        else:
            logger.warning("[RAG] 검색 결과 0건 → king='%s' 으로 historical_documents에서 문서를 찾지 못함", king_name)

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
            ("system", "아래는 조선왕조실록에서 검색된 관련 기록입니다. "
                       "이 기록을 참고하여 당신의 말투와 성격을 유지한 채로 답변하세요. "
                       "절대 AI처럼 정보를 나열하거나 해설하지 마세요. "
                       "왕으로서 직접 경험하거나 결정한 것처럼 1인칭으로 자연스럽게 말하세요.\n\n{context}"),
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

        # 6. 왕 이름이 키워드에 없으면 맨 앞에 추가
        king_name = self._get_king_name(persona_id)
        if king_name and king_name not in response_keywords:
            response_keywords.insert(0, king_name)

        return response, sources, response_keywords


# 싱글톤 인스턴스
chat_service = ChatService()
