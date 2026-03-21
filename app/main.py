from contextlib import asynccontextmanager
from time import perf_counter
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status

from app.config import settings
from app.generator import generate_answer
from app.models import AskRequest, AskResponse, HealthResponse, UserContext
from app.retriever import VectorStore


vector_store = VectorStore()


def get_current_user(
    x_api_key: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
    x_user_roles: Optional[str] = Header(default=""),
) -> UserContext:
    if not settings.auth_enabled:
        return UserContext(user_id="anonymous", is_admin=True, roles=["admin"])

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing x-api-key header.",
        )

    mapped_user_id = settings.api_keys.get(x_api_key)
    if not mapped_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    effective_user_id = x_user_id or mapped_user_id
    if effective_user_id != mapped_user_id and mapped_user_id not in settings.admin_user_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key cannot impersonate another user.",
        )

    roles = [r.strip() for r in (x_user_roles or "").split(",") if r.strip()]
    is_admin = effective_user_id in settings.admin_user_ids

    return UserContext(
        user_id=effective_user_id,
        is_admin=is_admin,
        roles=roles,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        vector_store.load()
        print("Vector index loaded.")
    except FileNotFoundError:
        print("No existing vector index found. Run scripts/build_index.py first.")
    yield


app = FastAPI(
    title="RAG Policy Copilot",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    user: UserContext = Depends(get_current_user),
) -> AskResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    started = perf_counter()

    top_k = request.top_k or settings.top_k
    chunks, diagnostics = vector_store.search(
        question=request.question,
        top_k=top_k,
        filters=request.filters,
        user=user,
    )

    generation_started = perf_counter()
    answer = generate_answer(request.question, chunks, diagnostics)
    generation_latency_ms = (perf_counter() - generation_started) * 1000
    total_latency_ms = (perf_counter() - started) * 1000

    diagnostics = diagnostics.model_copy(
        update={
            "generation_latency_ms": generation_latency_ms,
            "total_latency_ms": total_latency_ms,
        }
    )

    return AskResponse(
        answer=answer,
        citations=chunks,
        used_filters=request.filters,
        diagnostics=diagnostics,
    )