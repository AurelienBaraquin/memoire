"""Graphe de question.

START -> route -> retrieve -> grade --(docs pertinents)--> generate -> END
                     ^           |
                     |   (rien de pertinent, max 2 fois)
                     +-- rewrite <+
"""

import json
import re
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph

SOURCES = ["obsidian", "messages", "emails", "photos"]
MAX_REWRITES = 2

ROUTE_PROMPT = """Je cherche une information dans ma mémoire personnelle. Sources possibles :
- obsidian : mes notes (cours, projets, journal, idées)
- messages : mes conversations WhatsApp
- emails : mes mails
- photos : descriptions de mes photos (lieux, dates)

Question : {question}

Réponds uniquement en JSON : {{"sources": [...], "query": "..."}}
"sources" = les sources utiles (toutes si tu hésites), "query" = la question
reformulée en mots-clés pour une recherche sémantique."""

GRADE_PROMPT = """Question : {question}

Voici des extraits trouvés dans ma mémoire :
{context}

Lesquels aident à répondre à la question ? Réponds uniquement en JSON avec
la liste de leurs numéros, ex : [1, 3]. Réponds [] si aucun."""

REWRITE_PROMPT = """La recherche "{query}" n'a rien donné d'utile pour la question :
{question}

Propose une autre formulation courte (synonymes, autre angle). Réponds juste la requête."""

ANSWER_PROMPT = """Tu es ma mémoire personnelle. Réponds à ma question en t'appuyant
uniquement sur les extraits ci-dessous (mes notes, messages, mails et photos).
Cite les extraits utilisés avec leur numéro, ex : [2]. Si la réponse n'y est pas, dis-le.

Extraits :
{context}

Question : {question}"""

NOT_FOUND = "Je n'ai rien trouvé dans ta mémoire qui réponde à cette question."


class QueryState(TypedDict, total=False):
    question: str
    query: str
    sources: list[str]
    docs: list[Document]
    rewrites: int
    answer: str


def format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        parts.append(f"[{i}] ({meta['source']}, {meta['date']}, {meta['title']})\n{doc.page_content}")
    return "\n\n".join(parts)


def parse_json(text: str, default):
    """Les petits modeles rajoutent souvent du texte autour du json."""
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not match:
        return default
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return default


def build_query_graph(llm: BaseChatModel, store: VectorStore, top_k: int = 6):
    def ask(prompt: str) -> str:
        return str(llm.invoke(prompt).content).strip()

    def route(state: QueryState) -> QueryState:
        data = parse_json(ask(ROUTE_PROMPT.format(question=state["question"])), {})
        if not isinstance(data, dict):
            data = {}
        sources = [s for s in data.get("sources", []) if s in SOURCES] or SOURCES
        query = data.get("query") or state["question"]
        return {"sources": sources, "query": str(query), "rewrites": 0}

    def retrieve(state: QueryState) -> QueryState:
        # on filtre cote python : le format des filtres change selon le vector store
        sources = state.get("sources") or SOURCES
        candidates = store.similarity_search(state.get("query") or state["question"], k=top_k * 3)
        docs = [d for d in candidates if d.metadata.get("source") in sources]
        return {"docs": docs[:top_k]}

    def grade(state: QueryState) -> QueryState:
        docs = state["docs"]
        if not docs:
            return {"docs": []}
        prompt = GRADE_PROMPT.format(question=state["question"], context=format_docs(docs))
        keep = parse_json(ask(prompt), None)
        if not isinstance(keep, list):
            return {"docs": docs}  # reponse illisible, on garde tout
        return {"docs": [docs[i - 1] for i in keep if isinstance(i, int) and 0 < i <= len(docs)]}

    def after_grade(state: QueryState) -> str:
        if state["docs"]:
            return "generate"
        if state.get("rewrites", 0) < MAX_REWRITES:
            return "rewrite"
        return "generate"

    def rewrite(state: QueryState) -> QueryState:
        prompt = REWRITE_PROMPT.format(query=state.get("query", ""), question=state["question"])
        return {
            "query": ask(prompt).strip("\"'"),
            "sources": SOURCES,  # le routing s'est peut etre trompe
            "rewrites": state.get("rewrites", 0) + 1,
        }

    def generate(state: QueryState) -> QueryState:
        if not state["docs"]:
            return {"answer": NOT_FOUND}
        prompt = ANSWER_PROMPT.format(context=format_docs(state["docs"]), question=state["question"])
        return {"answer": ask(prompt)}

    graph = StateGraph(QueryState)
    graph.add_node("route", route)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.add_edge(START, "route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade, ["generate", "rewrite"])
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    return graph.compile()
