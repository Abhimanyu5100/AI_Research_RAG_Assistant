import logging
import os

from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_core.tools.retriever import create_retriever_tool

from src.agent.retriever import retriever
from src.config import TAVILY_API_KEY

logger = logging.getLogger(__name__)

# Exposed for graphs that want agent-driven retrieval. The default graph does
# RAG in the summarizer node instead, so this is deliberately not in `tools`:
# binding it would let the model pull the whole corpus into an already tight
# context budget.
vectorstore_tool = create_retriever_tool(
    retriever,
    name="PineconeVectorStore",
    description="Retrieves relevant research papers on AI/ML topics",
)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)

arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

tools = [arxiv_tool, wiki_tool]

# Tavily needs a key; without one, constructing the tool raises at import time
# and takes the whole app down. Degrade to arxiv + wikipedia instead.
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        tavily = TavilySearchResults()
        tools.insert(0, tavily)
    except Exception as exc:  # pragma: no cover - depends on optional package
        logger.warning("Tavily search unavailable (%s) -- continuing without it.", exc)
else:
    logger.warning("TAVILY_API_KEY not set -- web search disabled.")

all_tools = tools + [vectorstore_tool]
