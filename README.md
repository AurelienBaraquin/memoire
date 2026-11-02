# memoire

Petit RAG perso pour interroger "ma mémoire" : mes notes Obsidian, mes conversations
WhatsApp, mes mails et mes photos. Tout tourne en local avec [Ollama](https://ollama.com),
rien ne sort de ma machine.

```
$ memoire ask "c'est quand ma soutenance ?"
Ta soutenance est le 12 novembre à 14h, salle 3 [1].

Sources :
  [1] emails - 2025-10-20 - inbox.mbox
```

## Comment ça marche

Deux graphes [LangGraph](https://github.com/langchain-ai/langgraph) :

**Indexation** — les 4 loaders tournent en parallèle, puis découpage en chunks et
stockage dans Chroma (ids stables, donc on peut réindexer sans doublons).

```
        ┌─ obsidian ─┐
START ──┼─ messages ─┼── split ── store ── END
        ├─ emails ───┤
        └─ photos ───┘
```

**Question** — le LLM choisit les sources utiles et reformule la question, on cherche,
il trie les extraits pertinents, et si rien ne colle on reformule (2 fois max).

```
START ── route ── retrieve ── grade ──> generate ── END
                     ^          │
                     └─ rewrite ┘  (rien de pertinent)
```

| Source   | Format                                  | Ce qui est indexé                          |
|----------|-----------------------------------------|--------------------------------------------|
| obsidian | vault (`.md`)                           | texte, frontmatter, tags, `[[liens]]`      |
| messages | export WhatsApp `.txt` (android/iphone) | 1 document par conversation et par jour    |
| emails   | `.mbox` (Google Takeout, Thunderbird)   | expéditeur, objet, corps sans les citations |
| photos   | `.jpg` `.png` `.webp`                   | date + GPS EXIF, description par `llava`   |

## Installation

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
ollama pull llava          # optionnel, pour décrire les photos
uv sync
cp config.example.toml config.toml   # puis mettre ses chemins
```

## Utilisation

```bash
uv run memoire index
uv run memoire ask "qu'est-ce que Paul m'a dit pour la rando ?"
```

## Tests

```bash
uv run pytest
```

Les tests utilisent des données bidon (`tests/fixtures`) et des modèles factices,
pas besoin d'Ollama.

## Limites / idées

- les photos sont lentes à indexer avec llava (~5 s par photo sur mon pc)
- pas d'indexation incrémentale, on relit tout à chaque fois
- ajouter les filtres par date ("en mai 2025") dans le routing
