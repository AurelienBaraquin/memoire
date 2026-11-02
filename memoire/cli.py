import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

from memoire.config import Config, load_config
from memoire.index_graph import build_index_graph
from memoire.loaders.photos import make_captioner
from memoire.query_graph import build_query_graph


def open_store(config: Config) -> Chroma:
    embeddings = OllamaEmbeddings(model=config.models.embeddings, base_url=config.models.base_url)
    return Chroma(
        collection_name="memoire",
        embedding_function=embeddings,
        persist_directory=str(config.db_path),
    )


def cmd_index(config: Config) -> None:
    captioner = None
    if config.models.vision:
        vision = ChatOllama(model=config.models.vision, base_url=config.models.base_url)
        captioner = make_captioner(vision)
    graph = build_index_graph(config, open_store(config), captioner)
    result = graph.invoke({})
    print(f"{len(result['docs'])} documents -> {result['stored']} chunks indexés")


def cmd_ask(config: Config, question: str) -> None:
    llm = ChatOllama(model=config.models.chat, base_url=config.models.base_url, temperature=0)
    graph = build_query_graph(llm, open_store(config), config.top_k)
    result = graph.invoke({"question": question})
    print(result["answer"])
    if result["docs"]:
        print("\nSources :")
        for i, doc in enumerate(result["docs"], 1):
            meta = doc.metadata
            print(f"  [{i}] {meta['source']} - {meta['date']} - {meta['path']}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="memoire", description="RAG sur ma mémoire perso")
    parser.add_argument("-c", "--config", type=Path, default=Path("config.toml"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="indexe toutes les sources de la config")
    ask = sub.add_parser("ask", help="pose une question")
    ask.add_argument("question")
    args = parser.parse_args()

    if not args.config.exists():
        parser.error(f"{args.config} introuvable (copie config.example.toml)")
    config = load_config(args.config)
    if args.command == "index":
        cmd_index(config)
    else:
        cmd_ask(config, args.question)


if __name__ == "__main__":
    main()
