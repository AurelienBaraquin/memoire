"""Graphe d'indexation.

START -> (obsidian | messages | emails | photos en parallele) -> split -> store -> END
"""

import hashlib
import operator
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from memoire.config import Config
from memoire.loaders.emails import load_emails
from memoire.loaders.messages import load_messages
from memoire.loaders.obsidian import load_obsidian
from memoire.loaders.photos import Captioner, load_photos


class IndexState(TypedDict, total=False):
    # chaque loader ajoute ses docs, d'ou le reducer
    docs: Annotated[list[Document], operator.add]
    chunks: list[Document]
    stored: int


def chunk_id(chunk: Document, i: int) -> str:
    """Id stable -> reindexer ne cree pas de doublons."""
    key = f"{chunk.metadata['source']}|{chunk.metadata['path']}|{chunk.metadata['date']}|{i}"
    return hashlib.sha1(key.encode()).hexdigest()


def build_index_graph(config: Config, store: VectorStore, captioner: Captioner | None = None):
    loaders = {
        "obsidian": load_obsidian,
        "messages": load_messages,
        "emails": load_emails,
        "photos": lambda folder: load_photos(folder, captioner),
    }
    enabled = {
        name: getattr(config.sources, name)
        for name in loaders
        if getattr(config.sources, name) is not None
    }
    if not enabled:
        raise ValueError("aucune source configurée dans [sources]")

    def make_node(name: str, folder: Path):
        def node(state: IndexState) -> IndexState:
            if not folder.exists():
                print(f"[{name}] dossier introuvable : {folder}")
                return {"docs": []}
            docs = loaders[name](folder)
            print(f"[{name}] {len(docs)} documents")
            return {"docs": docs}

        return node

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    )

    def split(state: IndexState) -> IndexState:
        chunks = []
        for doc in state["docs"]:
            for i, chunk in enumerate(splitter.split_documents([doc])):
                chunk.metadata["chunk"] = i
                chunks.append(chunk)
        return {"chunks": chunks}

    def store_chunks(state: IndexState) -> IndexState:
        chunks = state["chunks"]
        if chunks:
            ids = [chunk_id(c, c.metadata["chunk"]) for c in chunks]
            store.add_documents(chunks, ids=ids)
        return {"stored": len(chunks)}

    graph = StateGraph(IndexState)
    for name, folder in enabled.items():
        graph.add_node(name, make_node(name, folder))
        graph.add_edge(START, name)
    graph.add_node("split", split)
    graph.add_node("store", store_chunks)
    graph.add_edge(list(enabled), "split")
    graph.add_edge("split", "store")
    graph.add_edge("store", END)
    return graph.compile()
