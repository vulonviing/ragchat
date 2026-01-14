from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_core.documents import Document

from .config import RagConfig

@dataclass
class DocumentManager:
    cfg: RagConfig

    def list_files(self) -> List[Path]:
        files: list[Path] = []
        for ext in ("*.pdf", "*.txt", "*.md"):
            files.extend(self.cfg.docs_dir.rglob(ext))
        return sorted(files)

    def save_upload_bytes(self, filename: str, data: bytes) -> Path:
        out_path = self.cfg.docs_dir / filename
        if out_path.exists():
            stem, suffix = out_path.stem, out_path.suffix
            i = 2
            while True:
                candidate = self.cfg.docs_dir / f"{stem}_{i}{suffix}"
                if not candidate.exists():
                    out_path = candidate
                    break
                i += 1
        out_path.write_bytes(data)
        return out_path

    def delete_files(self, paths: Iterable[str]) -> int:
        deleted = 0
        for p in paths:
            try:
                Path(p).unlink()
                deleted += 1
            except Exception:
                pass
        return deleted

    def load_langchain_documents(self) -> List[Document]:
        docs: list[Document] = []

        pdf_loader = DirectoryLoader(str(self.cfg.docs_dir), glob="**/*.pdf", loader_cls=PyPDFLoader)
        docs.extend(pdf_loader.load())

        txt_loader = DirectoryLoader(str(self.cfg.docs_dir), glob="**/*.txt", loader_cls=TextLoader)
        md_loader = DirectoryLoader(str(self.cfg.docs_dir), glob="**/*.md", loader_cls=TextLoader)
        docs.extend(txt_loader.load())
        docs.extend(md_loader.load())

        return docs