from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Union
import hashlib

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from .config import RagConfig

PathLike = Union[str, Path]

@dataclass
class DocumentManager:
    cfg: RagConfig

    def list_files(self) -> List[Path]:
        files: list[Path] = []
        for ext in ("*.pdf", "*.txt", "*.md"):
            files.extend(self.cfg.docs_dir.rglob(ext))
        return sorted(files)

    def make_doc_id(self, file_path: Path) -> str:
        # stable id: relative path inside documents/
        return str(file_path.resolve().relative_to(self.cfg.docs_dir.resolve()))

    def hash_bytes(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def hash_file(self, file_path: Path) -> str:
        h = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

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

    def delete_files(self, paths: Iterable[PathLike]) -> int:
        deleted = 0
        for p in paths:
            try:
                pth = Path(p)
                # Eğer sadece "foo.pdf" gibi geldiyse documents/ altına tamamla
                if not pth.is_absolute():
                    candidate = self.cfg.docs_dir / pth
                    if candidate.exists():
                        pth = candidate
                if pth.exists():
                    pth.unlink()
                    deleted += 1
            except Exception:
                pass
        return deleted

    def load_langchain_documents_for_file(self, file_path: Path) -> List[Document]:
        """
        Load one file and return LangChain Documents (with metadata like source/page).
        """
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return PyPDFLoader(str(file_path)).load()
        elif suffix in (".txt", ".md"):
            return TextLoader(str(file_path)).load()
        else:
            return []