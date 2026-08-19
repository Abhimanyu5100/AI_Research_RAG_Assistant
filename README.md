# AI Research RAG Assistant

A LangGraph-based conversational agent for **domain-specific scientific queries in AI/ML**. It combines retrieval-augmented generation over a corpus of arXiv papers with query reframing, context summarization, self-critique and external tool lookup, served through a FastAPI backend and a Streamlit chat interface.

---

## How it works

Each question flows through a stateful graph rather than a single prompt:

```
START → prepare → classify: is this AI/ML?
                    │
                    ├── no ──────────────────────────────► chatbot
                    │
                    └── yes → QueryFramer → summarizer ──► chatbot
                                                             │
                              tools (Tavily/arXiv/Wikipedia) ◄┤ tool calls
                                                             │
                                                    critique ─┴─► END
```

- **prepare** — clears per-turn state so each question starts from a clean slate.
- **classify** — off-topic questions skip retrieval entirely.
- **QueryFramer** — rewrites the question into a sharper retrieval query; the model still answers what the user actually asked.
- **summarizer** — hybrid search over Pinecone, then condenses the top passages into working context.
- **chatbot** — answers from that context, with Tavily / arXiv / Wikipedia available as tools.
- **critique** — checks the answer against the retrieved context and can request one revision.

---

## Project structure

- `src/config.py` — models, paths and token budgets, all overridable by environment variable.
- `src/embeddings.py` — the embedding model, shared by indexing and retrieval.
- `src/agent/` — the LangGraph graph (`graph.py`), Pinecone hybrid retrieval (`retriever.py`), and tools (`tools.py`).
- `src/api/main.py` — FastAPI backend with streaming responses.
- `frontend/app.py` — Streamlit chat interface.
- `scripts/` — `scrape_arxiv.py` to fetch papers, `build_index.py` to build the vector index, `healthcheck.py` to verify setup.
- `data/` — downloaded PDFs and extracted `.txt` files.

---

## Tech stack

- **LangGraph** — stateful agent graph with checkpointed conversation memory
- **LangChain + Groq** — fast inference (`openai/gpt-oss-120b`, `openai/gpt-oss-20b` by default)
- **Pinecone** — serverless hybrid index (dense + sparse, `dotproduct`)
- **Sentence-Transformers** — `intfloat/e5-large-v2` embeddings, computed locally
- **FastAPI** + **Streamlit** — backend and UI

---

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY="your_groq_api_key"
PINECONE_API_KEY="your_pinecone_api_key"
TAVILY_API_KEY="your_tavily_api_key"
```

### 3. Build the vector index

Fetch papers, then index them:

```bash
python scripts/scrape_arxiv.py --query "machine learning" --count 20
python scripts/build_index.py --rebuild
```

`--rebuild` recreates the index from scratch; without it the script leaves a populated index untouched.

### 4. Verify the setup

```bash
python scripts/healthcheck.py
```

Confirms your keys, that the configured Groq models are reachable, that the index is populated, and that retrieval returns documents with citation metadata.

### 5. Run

Two terminals:

```bash
uvicorn src.api.main:app --reload --port 8000
```

```bash
streamlit run frontend/app.py
```

Then open http://localhost:8501.

---

## Configuration

All settings live in `src/config.py` and can be overridden via `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_CHAT_MODEL` | `openai/gpt-oss-120b` | Main answering model |
| `GROQ_SUMMARY_MODEL` | `openai/gpt-oss-20b` | Summarization, routing, critique |
| `RETRIEVER_TOP_K` | `4` | Passages retrieved per query |
| `RETRIEVER_ALPHA` | `0.5` | Hybrid mix — `1.0` dense only, `0.0` sparse only |
| `E5_USE_PREFIXES` | `true` | E5 `query:` / `passage:` prefixes |
| `MAX_HISTORY_CHARS` | `8000` | Conversation kept per request |
| `MAX_OUTPUT_TOKENS` | `1500` | Cap on answer length |
| `MAX_CRITIQUE_RETRIES` | `1` | Answer revisions allowed |

Groq's model lineup changes over time. To see what your key can reach:

```bash
curl -s -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/openai/v1/models
```

---

## Retrieval

Retrieval is hybrid: dense E5 vectors alongside a BM25 sparse encoder fitted on your corpus and written to `bm25_values.json`. That file and the Pinecone index are a matched pair — `build_index.py` writes both together, so regenerate them together.

`intfloat/e5-large-v2` is trained with asymmetric `"query: "` / `"passage: "` prefixes and performs better with them, so they are enabled by default. Indexing and querying must agree: if you change `E5_USE_PREFIXES`, rebuild the index in the same setting.

---

## Sessions

Conversations are keyed by `session_id`, sent by the frontend and returned in the `X-Session-Id` header. Callers with different ids have independent histories; `POST /reset` clears one. Omit the id and each request gets a fresh session.

Token budgets in `src/config.py` bound how much history and retrieved context each request carries, which keeps requests within Groq's per-minute limits.

> Conversation state lives in LangGraph's in-process `MemorySaver`, so it resets when the server restarts. For a long-running deployment, swap in a persistent checkpointer (SQLite/Postgres) in `src/agent/graph.py`.

---

## Example queries

- "What is reinforcement learning?"
- "Compare Q-learning with policy gradient methods"
- "Latest papers on transformer-based vision models"

---

## License

MIT

---

## Acknowledgements

Thanks to LangChain, Pinecone, Groq, Tavily, and open-access scientific datasets.
