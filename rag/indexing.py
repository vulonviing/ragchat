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

    def build_or_update(self) -> Tuple[str, dict]:
        """
        Incremental indexing:
        - New file -> add
        - Changed file -> delete old doc_id chunks, then add
        - Unchanged file -> skip
        Returns (message, stats dict)
        """
        files = self.doc_manager.list_files()
        if not files:
            return ("No documents found in documents/.", {"new": 0, "updated": 0, "skipped": 0, "chunks": 0})

        indexed = self.vector_db.list_indexed_docs()  # doc_id -> {file_hash,...}

        embeddings = self.vector_db.embeddings()
        db = None

        new_cnt = updated_cnt = skipped_cnt = 0
        total_chunks = 0

        # Open/create db lazily
        if self.vector_db.exists():
            db = Chroma(persist_directory=str(self.cfg.db_dir), embedding_function=embeddings)

        for fp in files:
            doc_id = self.doc_manager.make_doc_id(fp)
            file_hash = self.doc_manager.hash_file(fp)

            prev = indexed.get(doc_id)
            prev_hash = prev.get("file_hash") if prev else None

            if prev_hash == file_hash:
                skipped_cnt += 1
                continue

            # load & split this single file
            raw_docs = self.doc_manager.load_langchain_documents_for_file(fp)
            if not raw_docs:
                # unsupported or empty
                skipped_cnt += 1
                continue

            chunks = self._split(raw_docs)

            # add our metadata on each chunk
            for ch in chunks:
                ch.metadata = dict(ch.metadata or {})
                ch.metadata["doc_id"] = doc_id
                ch.metadata["file_hash"] = file_hash
                ch.metadata["file_name"] = fp.name

            # if changed: delete old first
            if prev_hash is not None and prev_hash != file_hash:
                self.vector_db.delete_doc_id(doc_id)
                updated_cnt += 1
            else:
                new_cnt += 1

            # create db if needed
            if db is None:
                Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=str(self.cfg.db_dir),
                )
                db = Chroma(persist_directory=str(self.cfg.db_dir), embedding_function=embeddings)
            else:
                db.add_documents(chunks)

            total_chunks += len(chunks)

        msg = "Index complete."
        stats = {"new": new_cnt, "updated": updated_cnt, "skipped": skipped_cnt, "chunks": total_chunks}
        return (msg, stats)

    def remove_from_index(self, file_path) -> int:
        """
        Remove a single file's vectors from the DB by doc_id.
        """
        fp = file_path if hasattr(file_path, "resolve") else self.cfg.docs_dir / str(file_path)
        doc_id = self.doc_manager.make_doc_id(fp)
        return self.vector_db.delete_doc_id(doc_id)

    def reset(self) -> None:
        if self.cfg.db_dir.exists():
            shutil.rmtree(self.cfg.db_dir)
        self.cfg.db_dir.mkdir(exist_ok=True)