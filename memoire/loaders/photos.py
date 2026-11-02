"""Indexe les photos : metadonnees EXIF + description generee par un modele vision.

On ne stocke pas l'image, juste un texte qui la decrit, comme ca la
recherche marche avec les memes embeddings que le reste.
"""

import base64
import io
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from PIL import ExifTags, Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CAPTION_PROMPT = (
    "Décris cette photo en français en 2 ou 3 phrases : lieu, personnes, objets, "
    "ambiance. Ne fais pas de suppositions sur l'identité des personnes."
)

Captioner = Callable[[Path], str]


def gps_to_decimal(values: tuple[float, float, float], ref: str) -> float:
    deg, minutes, sec = (float(v) for v in values)
    result = deg + minutes / 60 + sec / 3600
    return round(-result if ref in ("S", "W") else result, 5)


def read_exif(path: Path) -> dict[str, str]:
    # get_ifd relit le fichier, il faut l'appeler avant de le fermer
    with Image.open(path) as img:
        exif = img.getexif()
        sub = exif.get_ifd(ExifTags.IFD.Exif)
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    infos: dict[str, str] = {}
    raw_date = sub.get(ExifTags.Base.DateTimeOriginal) or exif.get(ExifTags.Base.DateTime)
    if raw_date:
        try:
            infos["date"] = datetime.strptime(raw_date, "%Y:%m:%d %H:%M:%S").date().isoformat()
        except ValueError:
            pass
    if exif.get(ExifTags.Base.Model):
        infos["camera"] = str(exif[ExifTags.Base.Model]).strip()
    if gps.get(2) and gps.get(4):
        lat = gps_to_decimal(gps[2], gps.get(1, "N"))
        lon = gps_to_decimal(gps[4], gps.get(3, "E"))
        infos["gps"] = f"{lat}, {lon}"
    return infos


def make_captioner(model: BaseChatModel, max_size: int = 768) -> Captioner:
    """Captioner base sur un modele vision (llava via ollama)."""

    def caption(path: Path) -> str:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((max_size, max_size))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
        b64 = base64.b64encode(buffer.getvalue()).decode()
        message = HumanMessage(
            content=[
                {"type": "text", "text": CAPTION_PROMPT},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"},
            ]
        )
        return str(model.invoke([message]).content).strip()

    return caption


def load_photo(path: Path, folder: Path, captioner: Captioner | None) -> Document:
    infos = read_exif(path)
    date = infos.get("date") or datetime.fromtimestamp(path.stat().st_mtime).date().isoformat()
    lines = [f"Photo {path.name} prise le {date}"]
    if "gps" in infos:
        lines.append(f"Coordonnées GPS : {infos['gps']}")
    if "camera" in infos:
        lines.append(f"Appareil : {infos['camera']}")
    if captioner:
        lines.append(captioner(path))
    return Document(
        page_content="\n".join(lines),
        metadata={
            "source": "photos",
            "path": str(path.relative_to(folder)),
            "title": path.name,
            "date": date,
        },
    )


def load_photos(folder: Path, captioner: Captioner | None = None) -> list[Document]:
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS)
    return [load_photo(p, folder, captioner) for p in paths]
