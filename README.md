# 🤖 AI Agent Starter Kit — Free Edition

A clean, **free & open-source (MIT)** starting point for building AI agents with
**FastAPI + LangGraph**. Get a real tool-calling agent with conversation memory
running in minutes — then customize it into your own product.

> This is the **Free edition**. It contains the full LangGraph agent, tool
> calling, and per-session memory. Need **RAG, Docker, deployment configs, and
> production-grade swappable stores**? Grab the **[Pro edition](#-upgrade-to-pro)**.

---

## ⭐ Like it? Star the repo and upgrade to Pro

If this saved you time, a GitHub star helps a lot — and the
**[Pro edition](#-upgrade-to-pro)** adds everything you need to ship to
production.

---

## Features (Free)

| Feature | Included |
|---|---|
| FastAPI REST API + Swagger docs (`/docs`) | ✅ |
| LangGraph ReAct-style agent loop | ✅ |
| OpenAI chat integration | ✅ |
| Tool calling (calculator, current time) | ✅ |
| Per-session conversation memory (interface-based) | ✅ |
| Offline test suite (`pytest`) | ✅ |
| Type hints throughout + beginner-friendly comments | ✅ |
| RAG (ingest / embeddings / vector search) | 🔒 Pro |
| Docker + docker-compose | 🔒 Pro |
| One-click deployment configs + guide | 🔒 Pro |
| Redis / Qdrant swappable backends | 🔒 Pro |
| Commercial license + lifetime updates | 🔒 Pro |

---

## Tech Stack

- **Python 3.12+**
- **FastAPI** + **Uvicorn**
- **LangGraph** + **LangChain** — agent orchestration & tool calling
- **OpenAI API** — chat model (`gpt-4o-mini`)
- **Pydantic v2** / **pydantic-settings**
- **pytest**

---

## Project Structure

```
ai-agent-starter-kit-free/
├── app/
│   ├── main.py              # FastAPI app + endpoints (/health, /chat)
│   ├── core/
│   │   ├── config.py        # Settings loaded from env (.env)
│   │   └── logging.py       # Logging setup
│   ├── agents/
│   │   ├── graph.py         # LangGraph agent loop (agent ↔ tools)
│   │   ├── state.py         # Agent state (message history)
│   │   └── tools.py         # Tools: calculator, current_time
│   ├── memory/
│   │   └── store.py         # Memory interface + in-memory impl
│   └── schemas/
│       └── chat.py          # /chat request & response models
├── tests/
│   └── test_chat.py         # Offline tests (no API key needed)
├── requirements.txt
├── .env.example
├── README.md
└── LICENSE                  # MIT
```

---

## Quick Start

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env             # then edit .env and add OPENAI_API_KEY

# 4. Run the server (with live reload)
uvicorn app.main:app --reload

# 5. Open http://localhost:8000/docs
```

> The server starts even **without** an API key, and `/health` will respond.
> A key is only required the moment you call `/chat`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(empty)_ | **Required** for `/chat`. Your OpenAI key. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model the agent uses. |
| `APP_ENV` | `local` | Environment label. |

---

## API Examples

### Health check

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Chat with the agent

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"What time is it and calculate 15 * 23?"}'
```

```json
{
  "session_id": "demo",
  "response": "The current time is 2026-06-17T05:00:00+00:00 and 15 * 23 = 345.",
  "tool_calls": ["current_time", "calculator"]
}
```

Memory is keyed by `session_id`, so follow-up messages remember prior turns.

---

## How the Agent Works

A small **ReAct-style loop** built with LangGraph
([`app/agents/graph.py`](app/agents/graph.py)):

```
        ┌─────────┐   tool calls?   ┌────────┐
   ▶───▶│  agent  │───────yes──────▶│  tools │───┐
        └─────────┘                 └────────┘   │
             ▲                                    │
             └────────────────────────────────────┘
             │
             └── no tool calls ──▶  END
```

1. The **`agent`** node asks the LLM what to do; it answers or requests tools.
2. The **`tools`** node runs them and feeds results back.
3. The loop repeats until the model produces a final answer.

**Add your own tool** in [`app/agents/tools.py`](app/agents/tools.py):

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""   # the model reads this docstring
    return f"It's sunny in {city}."

def get_tools() -> list:
    return [calculator, current_time, get_weather]
```

---

## How Memory Works

History is stored per `session_id`
([`app/memory/store.py`](app/memory/store.py)) behind an abstract
`BaseMemoryStore` interface, with an in-memory implementation. To persist with
Redis/PostgreSQL, implement the same three methods and swap it in `app/main.py`.

> The **Pro edition** includes ready-made guidance and structure for Redis /
> PostgreSQL memory and a Qdrant / Chroma vector store.

---

## Run the tests

```bash
pip install -r requirements.txt
pytest -q
```

The suite runs fully offline — no API key required.

---

## 🚀 Upgrade to Pro

The **Pro edition** turns this starter into a production-ready service:

- 🔍 **RAG pipeline** — document ingest, OpenAI embeddings, semantic search
  (`/rag/ingest`, `/rag/search`)
- 🐳 **Docker + docker-compose** — one command to run the whole stack
- ☁️ **Deployment** — Render blueprint + Cloud Run / Railway guide (`DEPLOY.md`)
- 🔌 **Swappable backends** — clean interfaces to move to Redis / Qdrant
- 📦 **Commercial license** + **lifetime updates** + priority support

### [👉 Get the Pro edition — $79 on Gumroad](https://mashroombiz.gumroad.com/l/kcdynl)

> 30-day money-back guarantee. Commercial license + lifetime updates included.

---

## License

MIT — see [LICENSE](LICENSE). Use it freely in personal and commercial projects.
