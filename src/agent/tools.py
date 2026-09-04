import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from langchain_core.tools import StructuredTool
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_core.tools.retriever import create_retriever_tool

from src.agent.retriever import retriever
from src.config import (
    TAVILY_API_KEY,
    TOOL_BACKOFF_BASE,
    TOOL_MAX_RETRIES,
    TOOL_TIMEOUT_SECONDS,
    TOOL_TOTAL_DEADLINE,
)

logger = logging.getLogger(__name__)

# Records which tools degraded during the current turn, so the graph can tell
# the user that search was unavailable rather than silently answering without it.
DEGRADED: dict = {}

_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")


def bounded(tool):
    """Wrap a tool with a wall-clock timeout, capped retries and backoff.

    The underlying clients set their own retry policies -- arXiv's is 3 attempts
    with fixed 3s sleeps and no timeout -- so bounding them here keeps one flaky
    upstream from stalling the whole turn. On exhaustion the tool returns a
    plain-text notice instead of raising, letting the model answer from its own
    knowledge.
    """

    def _call(query: str) -> str:
        last = None
        started = time.monotonic()
        for attempt in range(TOOL_MAX_RETRIES + 1):
            remaining = TOOL_TOTAL_DEADLINE - (time.monotonic() - started)
            if remaining <= 0:
                last = last or "deadline exceeded"
                break
            try:
                future = _EXECUTOR.submit(tool.invoke, {"query": query})
                result = future.result(timeout=min(TOOL_TIMEOUT_SECONDS, remaining))
                DEGRADED.pop(tool.name, None)
                if not result or (isinstance(result, (list, dict, str)) and len(result) == 0):
                    # Distinguish "searched, found nothing" from "search broke":
                    # a bare [] leaves the model guessing which happened.
                    return f"[{tool.name} returned no results for this query.]"
                return str(result)
            except FutureTimeout:
                last = f"timed out after {TOOL_TIMEOUT_SECONDS:g}s"
                future.cancel()
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
            if attempt < TOOL_MAX_RETRIES:
                delay = TOOL_BACKOFF_BASE * (2 ** attempt)
                if (time.monotonic() - started) + delay >= TOOL_TOTAL_DEADLINE:
                    break
                logger.warning("%s failed (%s) -- retry %d/%d in %.1fs",
                               tool.name, last, attempt + 1, TOOL_MAX_RETRIES, delay)
                time.sleep(delay)

        logger.warning("%s unavailable after %d attempts: %s",
                       tool.name, TOOL_MAX_RETRIES + 1, last)
        DEGRADED[tool.name] = last
        return (f"[{tool.name} unavailable: {last}. "
                "Answer from your own knowledge and say the lookup could not be performed.]")

    return StructuredTool.from_function(
        func=_call,
        name=tool.name,
        description=tool.description,
    )

# Exposed for graphs that want agent-driven retrieval. The default graph does
# RAG in the summarizer node instead, so this is deliberately not in `tools`:
# binding it would let the model pull the whole corpus into an already tight
# context budget.
vectorstore_tool = create_retriever_tool(
    retriever,
    name="PineconeVectorStore",
    description="Retrieves relevant research papers on AI/ML topics",
)

# The `wikipedia` package still ships its 2014 defaults: a generic user-agent
# and a plaintext http endpoint. Wikimedia now answers that with 403 and a
# plain-text policy notice, which surfaces as a JSONDecodeError rather than an
# HTTP error. Identify the client properly and use https.
# https://phabricator.wikimedia.org/T400119
try:
    import wikipedia as _wikipedia

    _wikipedia.set_user_agent(
        "AI-Research-RAG-Assistant/1.0 "
        "(https://github.com/Abhimanyu5100/Research_Rag_Assistant)"
    )
    _wikipedia.wikipedia.API_URL = "https://en.wikipedia.org/w/api.php"
except Exception as exc:  # pragma: no cover - only if the package changes shape
    logger.warning("Could not configure the Wikipedia client (%s).", exc)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)

arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

tools = [bounded(arxiv_tool), bounded(wiki_tool)]

# Tavily needs a key; without one, constructing the tool raises at import time
# and takes the whole app down. Degrade to arxiv + wikipedia instead.
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        tavily = TavilySearchResults()
        tools.insert(0, bounded(tavily))
    except Exception as exc:  # pragma: no cover - depends on optional package
        logger.warning("Tavily search unavailable (%s) -- continuing without it.", exc)
else:
    logger.warning("TAVILY_API_KEY not set -- web search disabled.")

all_tools = tools + [vectorstore_tool]
