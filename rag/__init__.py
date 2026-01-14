from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

from .config import RagConfig
from .db import VectorDB
from .ingestion import DocumentManager
from .indexing import IndexManager
from .retrieval import Retriever
from .chat import ChatEngine
from .ollama import OllamaHealth


@dataclass(frozen=True)
class AppServices:
    """
    Convenience container that bundles all core services.
    This makes the package feel like a small library with a clean entrypoint.
    """
    cfg: RagConfig
    vector_db: VectorDB
    doc_manager: DocumentManager
    index_manager: IndexManager
    retriever: Retriever
    chat: ChatEngine


def create_app_services(project_root: Path) -> AppServices:
    """
    Single entrypoint to initialize the whole RAG system.
    Use from the UI layer (app.py) or any other runner.
    """
    cfg = RagConfig.from_project_root(project_root)

    vector_db = VectorDB(cfg)
    doc_manager = DocumentManager(cfg)
    index_manager = IndexManager(cfg, doc_manager, vector_db)
    retriever = Retriever(cfg, vector_db)
    chat = ChatEngine(cfg, retriever)

    return AppServices(
        cfg=cfg,
        vector_db=vector_db,
        doc_manager=doc_manager,
        index_manager=index_manager,
        retriever=retriever,
        chat=chat,
    )


__all__ = [
    "RagConfig",
    "VectorDB",
    "DocumentManager",
    "IndexManager",
    "Retriever",
    "ChatEngine",
    "AppServices",
    "create_app_services",
    "OllamaHealth"
]