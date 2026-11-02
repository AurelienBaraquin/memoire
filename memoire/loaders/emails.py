"""Charge les mails depuis des fichiers .mbox (export Gmail / Thunderbird)."""

import html
import mailbox
import re
from email.header import decode_header, make_header
from email.message import Message
from email.utils import parsedate_to_datetime
from pathlib import Path

from langchain_core.documents import Document

TAG_RE = re.compile(r"<[^>]+>")
QUOTE_RE = re.compile(r"^>.*$", re.MULTILINE)


def decode(value: str | None) -> str:
    if not value:
        return ""
    return str(make_header(decode_header(value)))


def html_to_text(raw: str) -> str:
    raw = re.sub(r"<(script|style).*?</\1>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    return html.unescape(TAG_RE.sub(" ", raw))


def get_body(msg: Message) -> str:
    """Prend la partie text/plain, sinon le html nettoye."""
    plain, rich = "", ""
    parts = msg.walk() if msg.is_multipart() else [msg]
    for part in parts:
        if part.get_content_disposition() == "attachment":
            continue
        payload = part.get_payload(decode=True)
        if not isinstance(payload, bytes):
            continue
        text = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
        if part.get_content_type() == "text/plain" and not plain:
            plain = text
        elif part.get_content_type() == "text/html" and not rich:
            rich = html_to_text(text)
    body = plain or rich
    # on vire les citations des reponses precedentes
    body = QUOTE_RE.sub("", body)
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def to_document(msg: Message, path: Path) -> Document | None:
    body = get_body(msg)
    if not body:
        return None
    subject = decode(msg["subject"]) or "(sans objet)"
    sender = decode(msg["from"])
    try:
        date = parsedate_to_datetime(msg["date"]).date().isoformat()
    except (TypeError, ValueError):
        date = ""
    return Document(
        page_content=f"Mail de {sender}\nObjet : {subject}\n\n{body}",
        metadata={"source": "emails", "path": path.name, "title": subject, "date": date},
    )


def load_emails(folder: Path) -> list[Document]:
    docs = []
    for path in sorted(folder.glob("*.mbox")):
        for msg in mailbox.mbox(path):
            doc = to_document(msg, path)
            if doc:
                docs.append(doc)
    return docs
