import os
import pickle
import time
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import settings
from app.db import MetadataStore
from app.models import ChunkRecord, RetrievedChunk, SearchDiagnostics, UserContext
from app.utils import ensure_dir, min_max_normalize, sparse_overlap_score


class VectorStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_map: List[str] = []
        self.metadata_store = MetadataStore()
        self.reranker: Optional[CrossEncoder] = None

        if settings.enable_reranker:
            try:
                self.reranker = CrossEncoder(settings.reranker_model)
            except Exception:
                self.reranker = None

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return vectors / norms

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype("float32")
        return self._normalize(embeddings)

    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        if not chunks:
            return
        embeddings = self.embed_texts([c.text for c in chunks])
        self.index.add(embeddings)
        self.id_map.extend([c.chunk_id for c in chunks])
        self.metadata_store.insert_chunks(chunks)

    def save(self) -> None:
        ensure_dir(settings.storage_dir)
        faiss.write_index(self.index, settings.faiss_index_path)
        with open(settings.id_map_path, "wb") as f:
            pickle.dump(self.id_map, f)

    def load(self) -> None:
        if not os.path.exists(settings.faiss_index_path):
            raise FileNotFoundError("FAISS index not found. Run build_index.py first.")

        self.index = faiss.read_index(settings.faiss_index_path)
        with open(settings.id_map_path, "rb") as f:
            self.id_map = pickle.load(f)

    def reset(self) -> None:
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_map = []
        self.metadata_store.reset()

    def _dense_candidates(
        self,
        question: str,
        top_k: int,
    ) -> List[dict]:
        if not self.id_map:
            return []

        q = self.embed_texts([question])
        search_k = min(max(top_k, 1) * settings.dense_candidate_k, len(self.id_map))
        scores, indices = self.index.search(q, search_k)

        candidates: List[dict] = []
        for dense_score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.id_map):
                continue
            candidates.append(
                {
                    "chunk_id": self.id_map[idx],
                    "dense_score": float(dense_score),
                }
            )
        return candidates

    def _apply_filters_and_acl(
        self,
        candidates: List[dict],
        filters: Dict,
        user: UserContext,
    ) -> List[dict]:
        filtered: List[dict] = []
        seen = set()

        for item in candidates:
            chunk_id = item["chunk_id"]
            if chunk_id in seen:
                continue

            chunk = self.metadata_store.get_chunk_by_id(chunk_id)
            if not chunk:
                continue
            if not self.metadata_store.filter_chunk(chunk, filters):
                continue
            if not self.metadata_store.user_can_access_chunk(chunk, user):
                continue

            seen.add(chunk_id)
            item["chunk"] = chunk
            filtered.append(item)

        return filtered

    def _add_sparse_scores(self, question: str, candidates: List[dict]) -> None:
        for item in candidates:
            item["sparse_score"] = sparse_overlap_score(question, item["chunk"].text)

    def _add_rerank_scores(self, question: str, candidates: List[dict]) -> None:
        if not candidates:
            return

        if self.reranker is None:
            for item in candidates:
                item["rerank_score"] = 0.0
            return

        pairs = [(question, item["chunk"].text) for item in candidates]
        try:
            scores = self.reranker.predict(pairs).tolist()
        except Exception:
            scores = [0.0] * len(candidates)

        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)

    def _combine_scores(self, candidates: List[dict]) -> List[dict]:
        dense_scores = [c["dense_score"] for c in candidates]
        sparse_scores = [c.get("sparse_score", 0.0) for c in candidates]
        rerank_scores = [c.get("rerank_score", 0.0) for c in candidates]

        dense_norm = min_max_normalize(dense_scores)
        sparse_norm = min_max_normalize(sparse_scores)
        rerank_norm = min_max_normalize(rerank_scores)

        for i, item in enumerate(candidates):
            item["score"] = (
                settings.dense_weight * dense_norm[i]
                + settings.sparse_weight * sparse_norm[i]
                + settings.rerank_weight * rerank_norm[i]
            )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

    def search(
        self,
        question: str,
        top_k: int,
        filters: Dict,
        user: UserContext,
    ) -> tuple[List[RetrievedChunk], SearchDiagnostics]:
        started = time.perf_counter()

        if not self.id_map:
            return [], SearchDiagnostics(
                confident=False,
                abstained=True,
                reason="index_not_loaded",
                candidate_count=0,
                retrieval_latency_ms=(time.perf_counter() - started) * 1000,
            )

        dense_candidates = self._dense_candidates(question=question, top_k=top_k)
        dense_candidates = self._apply_filters_and_acl(dense_candidates, filters, user)

        if not dense_candidates:
            return [], SearchDiagnostics(
                confident=False,
                abstained=True,
                reason="no_accessible_candidates",
                candidate_count=0,
                retrieval_latency_ms=(time.perf_counter() - started) * 1000,
            )

        dense_candidates = dense_candidates[: settings.hybrid_candidate_k]
        self._add_sparse_scores(question, dense_candidates)
        self._add_rerank_scores(question, dense_candidates)
        ranked = self._combine_scores(dense_candidates)

        results: List[RetrievedChunk] = []
        for item in ranked[:top_k]:
            chunk = item["chunk"]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    title=chunk.title,
                    text=chunk.text,
                    score=float(item["score"]),
                    dense_score=float(item["dense_score"]),
                    sparse_score=float(item.get("sparse_score", 0.0)),
                    rerank_score=float(item.get("rerank_score", 0.0)),
                    page=chunk.page,
                    section=chunk.section,
                )
            )

        top_score = results[0].score if results else 0.0
        second_score = results[1].score if len(results) > 1 else 0.0
        margin = top_score - second_score
        top_dense = results[0].dense_score if results else 0.0

        abstain_reason = None
        confident = True

        if not results:
            confident = False
            abstain_reason = "no_results"
        elif top_dense < settings.min_dense_score:
            confident = False
            abstain_reason = "low_dense_score"
        elif top_score < settings.min_combined_score:
            confident = False
            abstain_reason = "low_combined_score"
        elif len(results) > 1 and margin < settings.min_top_margin:
            confident = False
            abstain_reason = "low_margin"

        diagnostics = SearchDiagnostics(
            confident=confident,
            abstained=not confident,
            reason=abstain_reason,
            top_score=top_score,
            second_score=second_score,
            margin=margin,
            candidate_count=len(ranked),
            retrieval_latency_ms=(time.perf_counter() - started) * 1000,
        )

        return results, diagnostics