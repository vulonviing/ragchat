from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

from .config import RagConfig
from .db import VectorDB

@dataclass
class Retriever:
    cfg: RagConfig
    vector_db: VectorDB

    def retrieve(self, query: str, k: int) -> Tuple[str, List[str]]:
        if not self.vector_db.exists():
            return ("", [])

        db = self.vector_db.open()
        retriever = db.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)

        context = "\n\n".join([d.page_content for d in docs]) if docs else ""

        sources: list[str] = []
        for d in docs:
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", None)
            sources.append(f"{src} (page {page})" if page is not None else src)

        # de-dup
        seen = set()
        uniq = []
        for s in sources:
            if s not in seen:
                uniq.append(s)
                seen.add(s)

        return (context, uniq)