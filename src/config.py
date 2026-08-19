import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables once
load_dotenv()

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
# BM25 params fitted by scripts/build_index.py and reloaded by the retriever.
# Keep these two in sync -- a mismatch silently degrades hybrid search.
BM25_PATH = PROJECT_ROOT / "bm25_values.json"

# --- Secrets -------------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- Vector store --------------------------------------------------------
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "llama-text-embed-v2-index")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/e5-large-v2")
EMBEDDING_DIM = 1024

# E5 was trained with "query: " / "passage: " prefixes and loses accuracy
# without them. This changes the vectors, so the index must be built and
# queried in the same setting -- flip this only alongside a --rebuild.
E5_USE_PREFIXES = os.getenv("E5_USE_PREFIXES", "true").lower() in ("1", "true", "yes")

RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "4"))
RETRIEVER_ALPHA = float(os.getenv("RETRIEVER_ALPHA", "0.5"))  # 1.0 = dense only, 0.0 = sparse only

# --- LLMs ----------------------------------------------------------------
# Groq retires models regularly; keep these overridable so a decommissioned
# model is a config change rather than a code change.
# Check availability with: curl -H "Authorization: Bearer $GROQ_API_KEY" \
#     https://api.groq.com/openai/v1/models
CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
SUMMARY_MODEL = os.getenv("GROQ_SUMMARY_MODEL", "openai/gpt-oss-20b")
UTILITY_MODEL = os.getenv("GROQ_UTILITY_MODEL", "openai/gpt-oss-20b")

# --- Token budgeting -----------------------------------------------------
# Groq's free tier caps a single request at 8000 TPM, so the retrieved
# context and the running history both have to stay bounded.
# Characters, not messages: two verbose answers are enough to blow the budget
# on their own, so trimming has to measure size rather than count turns.
# Roughly 4 chars per token.
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))   # retrieved docs -> summarizer
MAX_SUMMARY_CHARS = int(os.getenv("MAX_SUMMARY_CHARS", "4000"))    # summary injected into the prompt
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "8000"))    # running conversation
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))  # hard cap on turns
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1500"))    # counts toward Groq's TPM
MAX_CRITIQUE_RETRIES = int(os.getenv("MAX_CRITIQUE_RETRIES", "1"))
