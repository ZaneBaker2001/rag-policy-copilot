from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_reranker: bool = True

    data_dir: str = "data/docs"
    storage_dir: str = "storage"

    top_k: int = 6
    max_context_chunks: int = 5
    dense_candidate_k: int = 30
    hybrid_candidate_k: int = 20

    chunk_size: int = 900
    chunk_overlap: int = 120
    min_chunk_chars: int = 180
    max_paragraph_chars: int = 1400

    sqlite_path: str = "storage/rag.db"
    faiss_index_path: str = "storage/index.faiss"
    id_map_path: str = "storage/id_map.pkl"

    # Hybrid scoring weights
    dense_weight: float = 0.55
    sparse_weight: float = 0.20
    rerank_weight: float = 0.25

    # Abstain / confidence
    min_dense_score: float = 0.20
    min_combined_score: float = 0.32
    min_top_margin: float = 0.03

    # Auth
    auth_enabled: bool = True
    api_keys: dict[str, str] = Field(
        default_factory=lambda: {
            # x-api-key: user_id
            "dev-admin-key": "admin",
            "dev-alice-key": "alice",
            "dev-bob-key": "bob",
        }
    )
    admin_user_ids: list[str] = Field(default_factory=lambda: ["admin"])

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()