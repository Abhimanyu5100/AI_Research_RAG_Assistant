import logging
import os
from typing import Annotated, List, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.agent.retriever import retriever
from src.agent.tools import tools
from src.config import (
    CHAT_MODEL,
    GROQ_API_KEY,
    MAX_CONTEXT_CHARS,
    MAX_CRITIQUE_RETRIES,
    MAX_HISTORY_CHARS,
    MAX_HISTORY_MESSAGES,
    MAX_OUTPUT_TOKENS,
    MAX_SUMMARY_CHARS,
    SUMMARY_MODEL,
    UTILITY_MODEL,
)

logger = logging.getLogger(__name__)

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add it to your .env file before starting the app."
    )

CRITIQUE_MARKER = "CRITIQUE:"


def get_llm(model: str = CHAT_MODEL, **kwargs) -> ChatGroq:
    return ChatGroq(api_key=GROQ_API_KEY, model=model, **kwargs)


# Small, cheap models for the routing/summarising/critiquing side-tasks.
llm_fr_summary = get_llm(SUMMARY_MODEL)
llm_utility = get_llm(UTILITY_MODEL, temperature=0)


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    citations: List[str]
    context: str
    refined_query: str
    critique_count: int


# --- helpers -------------------------------------------------------------

def _latest_user_query(messages: List[BaseMessage]) -> str:
    """The most recent genuine user turn, skipping our own critique nudges."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and CRITIQUE_MARKER not in (msg.content or ""):
            return msg.content
    return messages[-1].content if messages else ""


def _trim_history(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Keep the tail of the conversation inside the token budget.

    Groq's free tier rejects a single request over 8000 tokens, so an
    unbounded history will eventually fail every call on the thread. Counting
    turns is not enough -- two long answers alone can exceed the limit -- so
    walk backwards accumulating size and stop at the budget.
    """
    kept: List[BaseMessage] = []
    used = 0
    for msg in reversed(messages[-MAX_HISTORY_MESSAGES:]):
        # +200 covers role tags and any serialised tool-call payload.
        size = len(msg.content or "") + 200
        if kept and used + size > MAX_HISTORY_CHARS:
            break
        kept.append(msg)
        used += size
    kept.reverse()

    # Never open on a ToolMessage whose originating tool call was trimmed away.
    while kept and isinstance(kept[0], ToolMessage):
        kept.pop(0)
    return kept


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


# --- nodes ---------------------------------------------------------------

def llm_summarizer(docs) -> str:
    if not docs:
        return ""

    context = _truncate("\n\n".join(d.page_content for d in docs), MAX_CONTEXT_CHARS)

    prompt = f"""
You are a research assistant helping summarize academic papers in the field of AI and Machine Learning.

Below is a collection of excerpts from the most relevant research papers based on a user's query. The most relevant and higher-ranked documents appear first.

Summarize these documents in a detailed, technically accurate and comprehensive manner.
Focus more on the higher-ranking documents that appear earlier in the list.
Include important definitions, key contributions and findings.
The summary should be useful for someone conducting technical research on this topic.

--- Start of documents ---

{context}

--- End of documents ---

Summary:"""

    # .content, not the AIMessage itself -- interpolating the object leaks
    # `content='...' additional_kwargs={...}` straight into the next prompt.
    return llm_summarizer_invoke(prompt)


def llm_summarizer_invoke(prompt: str) -> str:
    try:
        return llm_fr_summary.invoke(prompt).content.strip()
    except Exception as exc:
        logger.warning("Summarisation failed (%s) -- continuing without context.", exc)
        return ""


def summarizer_node(state: State) -> dict:
    query = state.get("refined_query") or _latest_user_query(state["messages"])

    try:
        documents = retriever.invoke(query)
    except Exception as exc:
        logger.warning("Retrieval failed (%s) -- answering without context.", exc)
        return {"citations": [], "context": ""}

    if not documents:
        logger.info("No documents retrieved -- answering from the model's own knowledge.")
        return {"citations": [], "context": ""}

    # Requires `source` metadata, which scripts/build_index.py now writes.
    sources = sorted({
        os.path.basename(d.metadata["source"])
        for d in documents
        if d.metadata.get("source")
    })

    summary = _truncate(llm_summarizer(documents), MAX_SUMMARY_CHARS)
    if not summary or summary.lower().strip() in ("not found.", "not found"):
        logger.info("Empty summary -- answering without context.")
        return {"citations": sources, "context": ""}

    return {"citations": sources, "context": summary}


def chatbot(state: State) -> dict:
    """Single place where the outbound prompt is assembled.

    Building it here (rather than appending to `state["messages"]`) keeps the
    retrieved context out of the persisted history, so it is not re-sent on
    every subsequent turn.
    """
    context = state.get("context")
    system = (
        "You are a research assistant specialising in AI and Machine Learning. "
        "Answer accurately and cite specifics where you can."
    )
    if context:
        system += (
            "\n\nUse the following summarized context from retrieved research "
            f"papers to inform your answer:\n\n--- Context Start ---\n{context}\n"
            "--- Context End ---\n\nIf the context does not cover the question, "
            "say so and answer from your own knowledge."
        )

    prompt = [SystemMessage(content=system)] + _trim_history(state["messages"])
    # Output tokens count toward the same per-minute budget as input.
    llm = get_llm(max_tokens=MAX_OUTPUT_TOKENS)

    try:
        return {"messages": [llm.bind_tools(tools).invoke(prompt)]}
    except Exception as exc:
        # Some models (the gpt-oss family in particular) occasionally emit a
        # call to a built-in tool name such as "search" that was never offered,
        # which Groq rejects outright. The answer itself is usually fine, so
        # retry once with no tools rather than failing the whole turn.
        if "tool_use_failed" not in str(exc) and "tool call validation" not in str(exc):
            raise
        logger.warning("Invalid tool call from the model -- retrying without tools.")
        return {"messages": [llm.invoke(prompt)]}


def prepare_node(state: State) -> dict:
    """Clear the fields that belong to a single turn.

    They are checkpointed with the conversation, so without this a follow-up
    question inherits the previous turn's retrieved context and citations, and
    the critique budget stays spent for the rest of the session.
    """
    return {"context": "", "citations": [], "refined_query": "", "critique_count": 0}


def topic_classifier_node(state: State) -> Literal["QueryFramer", "chatbot"]:
    query = _latest_user_query(state["messages"])
    prompt = (
        'Is the following question related to AI or Machine Learning? '
        f'Answer only "yes" or "no".\n\nQuestion: {query}'
    )
    try:
        response = llm_utility.invoke(prompt).content.strip().lower()
    except Exception as exc:
        logger.warning("Classifier failed (%s) -- defaulting to the RAG path.", exc)
        return "QueryFramer"
    return "QueryFramer" if response.startswith("yes") else "chatbot"


def query_framer_node(state: State) -> dict:
    """Sharpen the query for retrieval only.

    The rephrasing is kept in state rather than pushed into `messages`, so the
    model still answers the question the user actually asked.
    """
    query = _latest_user_query(state["messages"])
    prompt = (
        "Rephrase the following question to make it clearer and more specific, "
        "suitable for searching an academic paper database. Reply with the "
        f"rephrased query only.\n\nQuestion: {query}\n\nRephrased:"
    )
    try:
        refined = llm_utility.invoke(prompt).content.strip()
    except Exception as exc:
        logger.warning("Query framing failed (%s) -- using the original query.", exc)
        refined = query
    return {"refined_query": refined or query}


def critique_node(state: State) -> dict:
    """Check the answer against the retrieved context.

    Grounding the check in the context matters: asked to judge a response in
    isolation, the model flags well-supported answers as hallucinated.
    """
    attempts = state.get("critique_count", 0)
    if attempts >= MAX_CRITIQUE_RETRIES:
        return {}

    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not (last.content or "").strip():
        return {}

    context = state.get("context") or ""
    reference = (
        f"--- Reference context ---\n{_truncate(context, 3000)}\n"
        if context
        else "No reference context was retrieved; judge on general correctness only.\n"
    )
    prompt = (
        "You are checking a research answer for factual errors or unsupported "
        "claims. Minor omissions are acceptable. Reply with exactly one word: "
        "PASS or FAIL.\n\n"
        f"{reference}\n--- Answer ---\n{_truncate(last.content, 3000)}\n\nVerdict:"
    )

    try:
        result = llm_utility.invoke(prompt).content.strip().upper()
    except Exception as exc:
        logger.warning("Critique failed (%s) -- accepting the answer.", exc)
        return {}

    if result.startswith("FAIL"):
        logger.info("Answer flagged for revision (attempt %d).", attempts + 1)
        return {
            "messages": [HumanMessage(content=(
                f"{CRITIQUE_MARKER} the previous answer may contain inaccuracies "
                "or unsupported claims. Revise it, keeping only what the context "
                "and your reliable knowledge support."
            ))],
            "critique_count": attempts + 1,
        }
    return {}


def check_critique_result(state: State) -> Literal["chatbot", "__end__"]:
    last = state["messages"][-1]
    if isinstance(last, HumanMessage) and CRITIQUE_MARKER in (last.content or ""):
        if state.get("critique_count", 0) > MAX_CRITIQUE_RETRIES:
            return END
        return "chatbot"
    return END


def chatbot_or_end(state: State) -> Literal["tools", "critique"]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "critique"


# --- graph ---------------------------------------------------------------

memory = MemorySaver()
graph_builder = StateGraph(State)

graph_builder.add_node("prepare", prepare_node)
graph_builder.add_node("summarizer", summarizer_node)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_node("QueryFramer", query_framer_node)
graph_builder.add_node("critique", critique_node)

graph_builder.add_edge(START, "prepare")
graph_builder.add_conditional_edges("prepare", topic_classifier_node)
graph_builder.add_edge("QueryFramer", "summarizer")
graph_builder.add_edge("summarizer", "chatbot")
graph_builder.add_conditional_edges("chatbot", chatbot_or_end)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_conditional_edges("critique", check_critique_result)

graph = graph_builder.compile(checkpointer=memory)
