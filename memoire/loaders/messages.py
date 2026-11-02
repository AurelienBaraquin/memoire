"""Charge les exports de discussions WhatsApp (.txt).

Un document = une conversation sur une journee, sinon les chunks
n'ont aucun contexte (un "ok" tout seul ca sert a rien).
"""

import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document

# android : "12/03/2024, 14:32 - Nom: message"
# iphone  : "[12/03/2024 14:32:05] Nom: message"
LINE_RE = re.compile(
    r"^\[?(\d{1,2}/\d{1,2}/\d{2,4}),? (\d{1,2}:\d{2})(?::\d{2})?\]?(?: -)? ([^:]+): (.*)$"
)
IGNORED = {"<Médias omis>", "<Media omitted>", "image absente", "image omitted"}


def parse_date(day: str) -> str:
    fmt = "%d/%m/%Y" if len(day.split("/")[-1]) == 4 else "%d/%m/%y"
    return datetime.strptime(day, fmt).date().isoformat()


def parse_chat(text: str) -> list[tuple[str, str, str, str]]:
    """Renvoie une liste (date, heure, auteur, message)."""
    messages: list[tuple[str, str, str, str]] = []
    for line in text.splitlines():
        line = line.replace("‎", "").strip()
        match = LINE_RE.match(line)
        if match:
            day, hour, author, msg = match.groups()
            messages.append((parse_date(day), hour, author.strip(), msg))
        elif messages and line:
            # message sur plusieurs lignes
            date, hour, author, msg = messages[-1]
            messages[-1] = (date, hour, author, f"{msg}\n{line}")
    return [m for m in messages if m[3].strip() not in IGNORED]


def load_chat(path: Path) -> list[Document]:
    # "Discussion WhatsApp avec Paul.txt" -> "Paul"
    name = re.sub(r"^(Discussion WhatsApp avec|WhatsApp Chat with) ", "", path.stem)
    by_day: dict[str, list[str]] = defaultdict(list)
    for date, hour, author, msg in parse_chat(path.read_text(encoding="utf-8")):
        by_day[date].append(f"[{hour}] {author}: {msg}")
    return [
        Document(
            page_content=f"Conversation avec {name} le {date}\n" + "\n".join(lines),
            metadata={"source": "messages", "path": path.name, "title": name, "date": date},
        )
        for date, lines in sorted(by_day.items())
    ]


def load_messages(folder: Path) -> list[Document]:
    docs = []
    for path in sorted(folder.glob("*.txt")):
        docs.extend(load_chat(path))
    return docs
