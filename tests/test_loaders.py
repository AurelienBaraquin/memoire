from pathlib import Path

from PIL import Image

from memoire.loaders.emails import load_emails
from memoire.loaders.messages import load_messages, parse_chat
from memoire.loaders.obsidian import clean_links, load_obsidian, parse_frontmatter
from memoire.loaders.photos import gps_to_decimal, load_photos

FIXTURES = Path(__file__).parent / "fixtures"


def test_frontmatter():
    meta, body = parse_frontmatter("---\ndate: 2025-01-02\ntitle: 'test'\n---\ncontenu")
    assert meta == {"date": "2025-01-02", "title": "test"}
    assert body == "contenu"


def test_wikilinks():
    text, links = clean_links("voir [[Note A]] et [[Note B#partie|alias]]")
    assert text == "voir Note A et alias"
    assert links == ["Note A", "Note B"]


def test_obsidian_vault():
    docs = load_obsidian(FIXTURES / "vault")
    assert [d.metadata["title"] for d in docs] == ["Systemes", "Journal"]
    cours = docs[0]
    assert cours.metadata["date"] == "2025-10-14"
    assert cours.metadata["tags"] == "cours, epitech, revisions"
    assert "ordonnanceurs" in cours.page_content
    assert "[[" not in cours.page_content


def test_whatsapp_parse():
    msgs = parse_chat("[03/05/25 09:12:01] Paul: salut\nsuite du msg")
    assert msgs == [("2025-05-03", "09:12", "Paul", "salut\nsuite du msg")]


def test_whatsapp_one_doc_per_day():
    docs = load_messages(FIXTURES / "whatsapp")
    assert [d.metadata["date"] for d in docs] == ["2025-05-03", "2025-05-04"]
    first = docs[0]
    assert first.metadata["title"] == "Paul"
    assert "prends de l'eau\nil fait chaud" in first.page_content
    assert "Médias omis" not in first.page_content


def test_emails():
    docs = load_emails(FIXTURES / "mails")
    assert len(docs) == 2
    convoc, commande = docs
    assert convoc.metadata["title"] == "Convocation soutenance"
    assert convoc.metadata["date"] == "2025-10-20"
    assert "12 novembre" in convoc.page_content
    assert "ancien message" not in convoc.page_content
    assert "expédiée" in commande.page_content
    assert "color:red" not in commande.page_content


def test_gps_to_decimal():
    assert gps_to_decimal((43.0, 12.0, 36.0), "N") == 43.21
    assert gps_to_decimal((5.0, 30.0, 0.0), "W") == -5.5


def test_photos_exif_and_caption(tmp_path):
    img = Image.new("RGB", (32, 32), "blue")
    exif = Image.Exif()
    exif[0x0110] = "Pixel 7"  # Model
    exif.get_ifd(0x8769)[0x9003] = "2025:05:04 10:30:00"  # DateTimeOriginal
    exif.get_ifd(0x8825).update({1: "N", 2: (43.0, 12.0, 36.0), 3: "E", 4: (5.0, 26.0, 24.0)})
    img.save(tmp_path / "calanques.jpg", exif=exif)
    (tmp_path / "notes.txt").write_text("pas une photo")

    docs = load_photos(tmp_path, captioner=lambda p: "Une crique avec de l'eau turquoise.")
    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["date"] == "2025-05-04"
    assert "43.21, 5.44" in doc.page_content
    assert "Pixel 7" in doc.page_content
    assert "eau turquoise" in doc.page_content
