from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel
from langchain_core.vectorstores import InMemoryVectorStore

from memoire.config import Config, Sources
from memoire.index_graph import build_index_graph
from memoire.query_graph import NOT_FOUND, build_query_graph, parse_json

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def store():
    return InMemoryVectorStore(DeterministicFakeEmbedding(size=16))


def doc(text, source="obsidian"):
    return Document(text, metadata={"source": source, "date": "2025-05-03", "title": "t", "path": "p"})


def test_index_all_sources(store):
    config = Config(
        sources=Sources(
            obsidian=FIXTURES / "vault",
            messages=FIXTURES / "whatsapp",
            emails=FIXTURES / "mails",
        )
    )
    graph = build_index_graph(config, store)
    result = graph.invoke({})
    assert len(result["docs"]) == 6
    assert {d.metadata["source"] for d in result["docs"]} == {"obsidian", "messages", "emails"}
    assert len(store.store) == result["stored"]

    # reindexer ne doit pas creer de doublons
    graph.invoke({})
    assert len(store.store) == result["stored"]


def test_index_without_sources(store):
    with pytest.raises(ValueError):
        build_index_graph(Config(), store)


def test_parse_json():
    assert parse_json('Voici : {"sources": ["emails"]} voilà', {}) == {"sources": ["emails"]}
    assert parse_json("[1, 3]", None) == [1, 3]
    assert parse_json("pas de json", "defaut") == "defaut"


def test_query_happy_path(store):
    store.add_documents([doc("rando calanques", "messages"), doc("cours systemes")])
    llm = FakeListChatModel(
        responses=['{"sources": ["messages"], "query": "calanques"}', "[1]", "Le 3 mai [1]"]
    )
    result = build_query_graph(llm, store).invoke({"question": "quand suis-je allé aux calanques ?"})
    assert result["sources"] == ["messages"]
    assert [d.page_content for d in result["docs"]] == ["rando calanques"]
    assert result["answer"] == "Le 3 mai [1]"
    assert result["rewrites"] == 0


def test_query_rewrites_then_gives_up(store):
    store.add_documents([doc("cours systemes")])
    llm = FakeListChatModel(responses=["pas du json", "[]", "autre requete", "[]", "encore", "[]"])
    result = build_query_graph(llm, store).invoke({"question": "ma taille de chaussures ?"})
    assert result["rewrites"] == 2
    assert result["answer"] == NOT_FOUND
