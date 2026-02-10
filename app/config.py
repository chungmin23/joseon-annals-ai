from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """환경변수 설정"""

    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4o-mini", env="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        env="OPENAI_EMBEDDING_MODEL",
    )
    embedding_dimension: int = Field(default=1536, env="EMBEDDING_DIMENSION")

    # PostgreSQL
    db_host: str = Field(..., env="DB_HOST")
    db_port: int = Field(default=5432, env="DB_PORT")
    db_name: str = Field(..., env="DB_NAME")
    db_user: str = Field(..., env="DB_USER")
    db_password: str = Field(..., env="DB_PASSWORD")

    # RAG Settings
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    similarity_cutoff: float = Field(default=0.7, env="SIMILARITY_CUTOFF")
    top_k_documents: int = Field(default=5, env="TOP_K_DOCUMENTS")

    # FastAPI
    environment: str = Field(default="development", env="ENVIRONMENT")

    @property
    def database_url(self) -> str:
        """PostgreSQL 연결 URL"""
        return (
            f"postgresql+psycopg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    model_config = {"env_file": ".env", "case_sensitive": False}


# 싱글톤 인스턴스
settings = Settings()
