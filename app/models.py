from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChunkRecord(BaseModel):
    chunk_id: str
    source: str
    title: str
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)


class UserContext(BaseModel):
    user_id: str
    is_admin: bool = False
    roles: List[str] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    top_k: Optional[int] = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    source: str
    title: str
    text: str
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0
    page: Optional[int] = None
    section: Optional[str] = None


class SearchDiagnostics(BaseModel):
    confident: bool
    abstained: bool
    reason: Optional[str] = None
    top_score: float = 0.0
    second_score: float = 0.0
    margin: float = 0.0
    candidate_count: int = 0


class AskResponse(BaseModel):
    answer: str
    citations: List[RetrievedChunk]
    used_filters: Dict[str, Any] = Field(default_factory=dict)
    diagnostics: SearchDiagnostics


class HealthResponse(BaseModel):
    status: str


class EvalCase(BaseModel):
    question: str
    expected_sources: List[str] = Field(default_factory=list)
    expected_chunk_ids: List[str] = Field(default_factory=list)