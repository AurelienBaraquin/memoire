"""Charge les notes d'un vault Obsidian."""

import re
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]")
TAG_RE = re.compile(r"(?<![\w/])#([\w/-]+)")


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Frontmatter yaml tres simple (cle: valeur), suffisant pour mes notes."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip("\"'")
    return meta, text[match.end():]


def clean_links(text: str) -> tuple[str, list[str]]:
    """Remplace [[note|alias]] par le texte lisible et renvoie les liens."""
    links = [m.group(1).strip() for m in WIKILINK_RE.finditer(text)]
    cleaned = WIKILINK_RE.sub(lambda m: (m.group(2) or m.group(1)).strip(), text)
    return cleaned, links


def load_note(path: Path, vault: Path) -> Document:
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    body, links = clean_links(body)
    tags = sorted(set(TAG_RE.findall(body)))
    if meta.get("tags"):
        tags = sorted(set(tags) | {t.strip(" #[]") for t in meta["tags"].split(",")})
    date = meta.get("date") or datetime.fromtimestamp(path.stat().st_mtime).date().isoformat()
    return Document(
        page_content=f"# {path.stem}\n{body.strip()}",
        metadata={
            "source": "obsidian",
            "path": str(path.relative_to(vault)),
            "title": path.stem,
            "date": date,
            "tags": ", ".join(tags),
            "links": ", ".join(links),
        },
    )


def load_obsidian(vault: Path) -> list[Document]:
    docs = []
    for path in sorted(vault.rglob("*.md")):
        # on ignore les dossiers caches (.obsidian, .trash)
        if any(part.startswith(".") for part in path.relative_to(vault).parts):
            continue
        docs.append(load_note(path, vault))
    return docs
