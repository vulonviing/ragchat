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
            return db._collection.count()  # internal, but practical
        except Exception:
            return -1