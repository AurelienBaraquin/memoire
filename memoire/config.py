import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Sources:
    obsidian: Path | None = None
    messages: Path | None = None
    emails: Path | None = None
    photos: Path | None = None


@dataclass(frozen=True)
class Models:
    chat: str = "llama3.1"
    embeddings: str = "nomic-embed-text"
    vision: str | None = "llava"
    base_url: str = "http://localhost:11434"


@dataclass(frozen=True)
class Config:
    sources: Sources = field(default_factory=Sources)
    models: Models = field(default_factory=Models)
    db_path: Path = Path("chroma")
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 6


def _path(value: str | None, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def load_config(path: Path) -> Config:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    src = data.get("sources", {})
    sources = Sources(**{name: _path(src.get(name), base) for name in Sources.__annotations__})
    models = Models(**data.get("models", {}))
    index = data.get("index", {})
    return Config(
        sources=sources,
        models=models,
        db_path=_path(index.get("db_path"), base) or base / "chroma",
        chunk_size=index.get("chunk_size", 1000),
        chunk_overlap=index.get("chunk_overlap", 150),
        top_k=index.get("top_k", 6),
    )
