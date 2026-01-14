from __future__ import annotations
from dataclasses import dataclass
import shutil
from typing import Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from .config import RagConfig
from .db import VectorDB
from .ingestion import DocumentManager

@dataclass
class IndexManager:
    cfg: RagConfig
    doc_manager: DocumentManager
    vector_db: VectorDB

    def _split(self, docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
        )
        return splitter.split_documents(docs)

    def build_or_update(self) -> Tuple[str, int]:
        docs = self.doc_manager.load_langchain_documents()
        if not docs:
            return ("No documents found in documents/.", 0)

        chunks = self._split(docs)
        embeddings = self.vector_db.embeddings()

        if self.vector_db.exists():
            db = Chroma(persist_directory=str(self.cfg.db_dir), embedding_function=embeddings)
            db.add_documents(chunks)
            return ("Index updated.", len(chunks))
        else:
            Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(self.cfg.db_dir),
            )
            return ("Index created.", len(chunks))

    def reset(self) -> None:
        if self.cfg.db_dir.exists():
            shutil.rmtree(self.cfg.db_dir)
        self.cfg.db_dir.mkdir(exist_ok=True)