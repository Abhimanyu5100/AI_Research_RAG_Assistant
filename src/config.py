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
# The critique compares an answer against retrieved sources. With no sources it
# is an ungrounded second opinion that has been observed to reject correct
# answers, and a revision costs a full regeneration -- roughly doubling latency.
CRITIQUE_REQUIRES_CONTEXT = os.getenv(
    "CRITIQUE_REQUIRES_CONTEXT", "true").lower() in ("1", "true", "yes")
# A single tool result must never be able to consume the whole history budget.
MAX_TOOL_RESULT_CHARS = int(os.getenv("MAX_TOOL_RESULT_CHARS", "3000"))

# --- External tool reliability ------------------------------------------
# arXiv's own client retries 3 times with fixed 3s sleeps and no timeout, so a
# failing upstream costs ~15s before it gives up. These bound every tool call
# uniformly and let the agent degrade to its own knowledge instead of stalling.
TOOL_TIMEOUT_SECONDS = float(os.getenv("TOOL_TIMEOUT_SECONDS", "8"))
TOOL_MAX_RETRIES = int(os.getenv("TOOL_MAX_RETRIES", "2"))
TOOL_BACKOFF_BASE = float(os.getenv("TOOL_BACKOFF_BASE", "0.5"))
# Ceiling on a whole tool call including retries, so a hanging upstream cannot
# cost timeout x attempts before the agent moves on.
TOOL_TOTAL_DEADLINE = float(os.getenv("TOOL_TOTAL_DEADLINE", "12"))
# Models trained with built-in browsing can keep requesting lookups instead of
# answering. Cap the rounds, then force a final answer with no tools offered.
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "3"))
# Attempts to get a tool-free answer when the model keeps emitting tool calls
# that were never offered. Sampling varies, so a second ask usually lands.
TOOL_FREE_ATTEMPTS = int(os.getenv("TOOL_FREE_ATTEMPTS", "3"))

# Retrieval scores alone do not separate a relevant corpus hit from an
# unrelated one, so an explicit relevance check gates whether retrieved
# context (and its citations) are used at all.
RELEVANCE_GATE = os.getenv("RELEVANCE_GATE", "true").lower() in ("1", "true", "yes")
# Threshold on the MEAN of the retrieved scores, not the max. Measured over this
# corpus, the best single hit overlaps between on- and off-topic questions
# (on-topic 0.670-0.851, off-topic 0.463-0.725) because an unrelated question
# still has one nearest neighbour. The mean separates them: on-topic 0.663-0.827
# vs off-topic 0.460-0.607. The margin is narrow and measured on a small sample,
# so the model-side check in the summarizer stays as a second gate.
RELEVANCE_MIN_SCORE = float(os.getenv("RELEVANCE_MIN_SCORE", "0.63"))
