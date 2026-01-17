from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal

from .config import RagConfig
from .db import VectorDB

RetrievalMode = Literal["similarity", "mmr", "threshold"]

@dataclass
class RetrievalParams:
    mode: RetrievalMode = "similarity"
    k: int = 9
    fetch_k: int = 20          # MMR
    score_threshold: float = 0.35  # Threshold (0-1)

@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: Optional[int]
    score: Optional[float]  # 0-1 relevance score

@dataclass
class Retriever:
    cfg: RagConfig
    vector_db: VectorDB

    def retrieve(self, query: str, params: RetrievalParams) -> Tuple[str, List[str], List[RetrievedChunk]]:
        """
        Returns:
          context_text, unique_sources, debug_chunks (with optional scores)
        """
        if not self.vector_db.exists():
            return ("", [], [])

        db = self.vector_db.open()

        docs = []
        scored = []  # list[(doc, score)] if we have scores

        mode = params.mode

        # --- Similarity / Threshold: use relevance scores when possible ---
        if mode in ("similarity", "threshold"):
            # Chroma wrapper usually supports this:
            # returns List[Tuple[Document, float]] where float is relevance score 0..1
            try:
                scored = db.similarity_search_with_relevance_scores(query, k=params.k)
            except Exception:
                # fallback: no scores
                docs = db.similarity_search(query, k=params.k)

            if scored:
                if mode == "threshold":
                    scored = [pair for pair in scored if pair[1] is not None and pair[1] >= params.score_threshold]
                docs = [d for d, _ in scored]

        # --- MMR: diverse results (scores not typically returned) ---
        elif mode == "mmr":
            # returns List[Document]
            docs = db.max_marginal_relevance_search(query, k=params.k, fetch_k=params.fetch_k)

        else:
            # default fallback
            docs = db.similarity_search(query, k=params.k)

        # Build context
        context = "\n\n".join([d.page_content for d in docs]) if docs else ""

        # Build sources + debug chunks
        sources: list[str] = []
        debug: list[RetrievedChunk] = []

        # If we have scored pairs, map doc->score
        score_map = {}
        if scored:
            for d, s in scored:
                # doc objects may not be hashable; map by id
                score_map[id(d)] = s

        for d in docs:
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", None)
            score = score_map.get(id(d), None)

            sources.append(f"{src} (page {page})" if page is not None else src)

            debug.append(
                RetrievedChunk(
                    text=d.page_content,
                    source=src,
                    page=page,
                    score=score,
                )
            )

        # de-dup sources preserve order
        seen = set()
        uniq_sources = []
        for s in sources:
            if s not in seen:
                uniq_sources.append(s)
                seen.add(s)

        return (context, uniq_sources, debug)