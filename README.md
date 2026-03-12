# 조선왕조실록 AI 챗봇 서버

조선왕조실록 문서를 기반으로 한 **RAG(Retrieval-Augmented Generation)** 챗봇 백엔드 서버입니다.
사용자는 조선 국왕 페르소나와 대화하며 역사적 맥락에 맞는 답변을 받을 수 있습니다.

## 주요 기능

- **RAG 파이프라인**: pgvector 기반 벡터 유사도 검색 + 키워드 매칭 하이브리드 검색
- **스트리밍 응답**: Server-Sent Events(SSE)를 통한 실시간 토큰 스트리밍
- **페르소나 역할극**: 왕 ID와 시스템 프롬프트를 파라미터로 받아 역할극 지원
- **한국어 NER**: KoELECTRA(HuggingFace), Kiwi 형태소 분석기, 정규식 폴백을 통한 키워드 추출
- **카테고리 자동 감지**: 질의 키워드 기반으로 문서 카테고리 자동 필터링

## 기술 스택

| 구분 | 기술 |
|------|------|
| 프레임워크 | FastAPI 0.109.0 + Uvicorn |
| AI/LLM | LangChain 0.2.17, OpenAI GPT-4o-mini |
| 임베딩 | OpenAI text-embedding-3-small (512차원) |
| 데이터베이스 | PostgreSQL + pgvector (AWS RDS) |
| 한국어 NLP | HuggingFace KoELECTRA NER, Kiwi |
| 컨테이너 | Docker |
| 데이터 검증 | Pydantic v2 |

## 프로젝트 구조

```
joseon-annals-ai/
├── app/
│   ├── main.py          # FastAPI 앱 및 라우터
│   ├── config.py        # 환경 변수 기반 설정
│   ├── schemas.py       # 요청/응답 Pydantic 스키마
│   └── services/
│       └── chat_service.py  # RAG 서비스 (LangChain)
├── docs/                # API 명세 및 설계 문서
├── Dockerfile
├── requirements.txt
└── .env.example
```

## 시작하기

### 사전 요구사항

- Python 3.11+
- PostgreSQL + pgvector 확장 설치된 DB (또는 AWS RDS)
- OpenAI API 키

### 환경 변수 설정

`.env.example`을 복사하여 `.env` 파일을 생성하고 값을 채워주세요.

```bash
cp .env.example .env
```

```env
# OpenAI API
OPENAI_API_KEY=sk-...

# PostgreSQL (RDS)
DB_HOST=your-db-host
DB_PORT=5432
DB_NAME=annals_db
DB_USER=your-user
DB_PASSWORD=your-password

# FastAPI
ENVIRONMENT=development

# HuggingFace (선택사항)
HF_TOKEN=hf_...

# RAG 설정
CHUNK_OVERLAP=200
SIMILARITY_CUTOFF=0.5
TOP_K_DOCUMENTS=5

# OpenAI 모델
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=512
```

### 로컬 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 서버 시작
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker 실행

```bash
# 이미지 빌드
docker build -t joseon-annals-ai .

# 컨테이너 실행
docker run -p 8000:8000 --env-file .env joseon-annals-ai
```

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/health` | 헬스 체크 |
| `GET` | `/` | 서버 정보 |
| `POST` | `/api/chat` | 동기 채팅 응답 |
| `POST` | `/api/chat/stream` | SSE 스트리밍 채팅 응답 |

### 채팅 요청 예시

```json
POST /api/chat
{
  "query": "임진왜란에 대해 설명해주세요",
  "persona_id": "king_seonjo",
  "system_prompt": "당신은 조선 선조 임금입니다.",
  "chat_history": []
}
```

### 채팅 응답 예시

```json
{
  "answer": "임진왜란은 1592년...",
  "sources": [
    {
      "content": "관련 실록 원문...",
      "similarity_score": 0.87
    }
  ],
  "keywords": ["임진왜란", "선조", "1592년"]
}
```

스트리밍 응답(`/api/chat/stream`)은 `text/event-stream` 형식으로 토큰을 순차 전송합니다.

## CORS 설정

React 프론트엔드(`:3000`)와 Spring Boot 백엔드(`:8080`)에 대해 CORS가 허용되어 있습니다.

## 문서

- [API 상세 명세](docs/AI_CHAT_API_DETAIL.md)
- [배포 가이드](docs/deployment_guide.md)
- [하이브리드 검색 설계](docs/hybrid_search_plan.md)
