"""Graphe de question : retrieve -> generate."""

from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph

ANSWER_PROMPT = """Tu es ma mémoire personnelle. Réponds à ma question en t'appuyant
uniquement sur les extraits ci-dessous (mes notes, messages, mails et photos).
Cite les extraits utilisés avec leur numéro, ex : [2]. Si la réponse n'y est pas, dis-le.

Extraits :
{context}

Question : {question}"""


class QueryState(TypedDict, total=False):
    question: str
    docs: list[Document]
    answer: str


def format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        parts.append(f"[{i}] ({meta['source']}, {meta['date']}, {meta['title']})\n{doc.page_content}")
    return "\n\n".join(parts)


def build_query_graph(llm: BaseChatModel, store: VectorStore, top_k: int = 6):
    def retrieve(state: QueryState) -> QueryState:
        return {"docs": store.similarity_search(state["question"], k=top_k)}

    def generate(state: QueryState) -> QueryState:
        prompt = ANSWER_PROMPT.format(context=format_docs(state["docs"]), question=state["question"])
        return {"answer": str(llm.invoke(prompt).content)}

    graph = StateGraph(QueryState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
