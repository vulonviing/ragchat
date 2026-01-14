from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class RagConfig:
    base_dir: Path
    docs_dir: Path
    db_dir: Path

    embed_model: str = "nomic-embed-text"
    llm_model: str = "llama3.1:8b"

    chunk_size: int = 800
    chunk_overlap: int = 120

    default_k: int = 4

    @staticmethod
    def from_project_root(project_root: Path) -> "RagConfig":
        base = project_root.resolve()
        docs = base / "documents"
        db = base / "chroma_db"
        docs.mkdir(exist_ok=True)
        db.mkdir(exist_ok=True)
        return RagConfig(base_dir=base, docs_dir=docs, db_dir=db)