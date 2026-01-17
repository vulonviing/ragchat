from __future__ import annotations
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from .config import RagConfig

@dataclass
class VectorDB:
    cfg: RagConfig

    def exists(self) -> bool:
        return self.cfg.db_dir.exists() and any(self.cfg.db_dir.iterdir())

    def embeddings(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(model=self.cfg.embed_model)

    def open(self) -> Chroma:
        return Chroma(
            persist_directory=str(self.cfg.db_dir),
            embedding_function=self.embeddings(),
        )

    def count_chunks(self) -> int:
        if not self.exists():
            return 0
        db = self.open()
        try:
            return db._collection.count()
        except Exception:
            return -1

    def delete_doc_id(self, doc_id: str) -> int:
        """
        Delete all vectors/chunks belonging to a given doc_id.
        Returns how many were deleted (best effort).
        """
        if not self.exists():
            return 0
        db = self.open()
        try:
            before = db._collection.count()
            db._collection.delete(where={"doc_id": doc_id})
            after = db._collection.count()
            return max(0, before - after)
        except Exception:
            # Even if we can't compute exact count, attempt delete
            try:
                db._collection.delete(where={"doc_id": doc_id})
            except Exception:
                pass
            return -1

    def list_indexed_docs(self) -> dict[str, dict]:
        """
        Returns a dict: doc_id -> {"file_name":..., "file_hash":...}
        Reads metadata from collection and deduplicates.
        """
        if not self.exists():
            return {}

        db = self.open()
        try:
            n = db._collection.count()
            if n == 0:
                return {}
            res = db._collection.get(include=["metadatas"], limit=n)
            metadatas = res.get("metadatas", []) or []
            out: dict[str, dict] = {}
            for md in metadatas:
                if not md:
                    continue
                did = md.get("doc_id")
                if not did:
                    continue
                if did not in out:
                    out[did] = {
                        "file_name": md.get("file_name"),
                        "file_hash": md.get("file_hash"),
                    }
            return out
        except Exception:
            return {}