import json
import logging
import os
import re
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
from src.agent.tools import DEGRADED, tools
from src.config import (
    CHAT_MODEL,
    CRITIQUE_REQUIRES_CONTEXT,
    GROQ_API_KEY,
    MAX_CONTEXT_CHARS,
    MAX_CRITIQUE_RETRIES,
    MAX_HISTORY_CHARS,
    MAX_HISTORY_MESSAGES,
    MAX_OUTPUT_TOKENS,
    MAX_SUMMARY_CHARS,
    MAX_TOOL_RESULT_CHARS,
    MAX_TOOL_ROUNDS,
    RELEVANCE_GATE,
    RELEVANCE_MIN_SCORE,
    SUMMARY_MODEL,
    TOOL_FREE_ATTEMPTS,
    UTILITY_MODEL,
)

logger = logging.getLogger(__name__)

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add it to your .env file before starting the app."
    )

CRITIQUE_MARKER = "CRITIQUE:"
# Providers reject a tool call that was never offered. Models trained with
# built-in browsing (the gpt-oss family) can emit one even when no tools are
# bound, so this is a recoverable condition, not a hard failure.
_TOOL_ERROR_HINTS = ("tool_use_failed", "tool call validation", "tool choice is none")


def _is_tool_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(hint in text for hint in _TOOL_ERROR_HINTS)


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "rate_limit_exceeded" in text or "error code: 429" in text


def _is_provider_unreachable(exc: Exception) -> bool:
    """The LLM provider is refusing us outright (blocked IP, bad key, network).

    Distinct from a malformed reply: no later node can succeed either, so the
    turn should stop now with one clear message instead of degrading through
    every step and reporting work it never actually did.
    """
    text = str(exc).lower()
    return any(h in text for h in (
        "error code: 401", "error code: 403", "permissiondenied",
        "authenticationerror", "apiconnectionerror", "access denied",
    ))

NOT_RELEVANT = "NOT_RELEVANT"


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
    model_fallback: str
    ml_parts: List[str]
    general_parts: List[str]
    critique_count: int
    tool_rounds: int


# --- helpers -------------------------------------------------------------

def _latest_user_query(messages: List[BaseMessage]) -> str:
    """The most recent genuine user turn, skipping our own critique nudges."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and CRITIQUE_MARKER not in (msg.content or ""):
            return msg.content
    return messages[-1].content if messages else ""


def _truncate_tool_payloads(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Cap each tool result so no single payload can consume the whole budget."""
    out = []
    for m in messages:
        if isinstance(m, ToolMessage) and len(m.content or "") > MAX_TOOL_RESULT_CHARS:
            m = m.model_copy(update={
                "content": (m.content or "")[:MAX_TOOL_RESULT_CHARS] + "\n...[truncated]"
            })
        out.append(m)
    return out


def _sanitize_tool_pairs(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Keep tool calls and their results consistent.

    Providers reject an assistant turn whose tool calls have no matching
    results, and a result whose call is missing, so trimming must not leave
    either half orphaned.
    """
    offered = set()
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            offered.update(c["id"] for c in m.tool_calls)

    out = [m for m in messages
           if not (isinstance(m, ToolMessage) and m.tool_call_id not in offered)]

    answered = {m.tool_call_id for m in out if isinstance(m, ToolMessage)}
    fixed: List[BaseMessage] = []
    for m in out:
        if isinstance(m, AIMessage) and m.tool_calls:
            if any(c["id"] not in answered for c in m.tool_calls):
                if (m.content or "").strip():
                    m = AIMessage(content=m.content)  # drop the unanswered calls
                else:
                    continue                          # drop the empty stub entirely
        fixed.append(m)
    return fixed


def _trim_history(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Keep the tail of the conversation inside the token budget.

    Counting turns is not enough -- one large tool result can exceed the budget
    on its own -- so payloads are capped first, then messages are accumulated
    backwards. The most recent question is always retained: without it the model
    receives context with nothing to answer.
    """
    if not messages:
        return []

    msgs = _truncate_tool_payloads(messages)

    kept: List[BaseMessage] = []
    used = 0
    for m in reversed(msgs[-MAX_HISTORY_MESSAGES:]):
        # +200 covers role tags and any serialised tool-call payload.
        size = len(m.content or "") + 200
        if kept and used + size > MAX_HISTORY_CHARS:
            break
        kept.append(m)
        used += size
    kept.reverse()

    kept = _sanitize_tool_pairs(kept)

    if not any(isinstance(m, HumanMessage) for m in kept):
        latest = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
        if latest is not None:
            kept.insert(0, latest)
    return kept


# Models often emit typographic dashes and quotes. BM25 tokenises "multi-agent"
# and "multi‑agent" differently, so a non-breaking hyphen in a generated query
# measurably weakens sparse retrieval.
_PUNCT_FOLD = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u00a0": " ",
})


def _fold_punctuation(text: str) -> str:
    return (text or "").translate(_PUNCT_FOLD)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


# --- nodes ---------------------------------------------------------------

def llm_summarizer(docs, query: str) -> str:
    if not docs:
        return ""

    context = _truncate("\n\n".join(d.page_content for d in docs), MAX_CONTEXT_CHARS)

    relevance_rule = (
        f'If the excerpts are about a clearly different topic and contain nothing that '
        f'informs an answer, reply with exactly "{NOT_RELEVANT}" and nothing else. '
        "If any part of them is useful -- even partially -- summarise that part instead. "
        "Bibliography entries and reference lists on their own do not count as useful.\n\n"
        if RELEVANCE_GATE else ""
    )

    prompt = f"""
You are a research assistant helping summarize academic papers in the field of AI and Machine Learning.

The user asked: {query}

Below is a collection of excerpts retrieved for that question. The higher-ranked documents appear first.

{relevance_rule}Summarize these documents in a detailed, technically accurate and comprehensive manner.
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
    query = _fold_punctuation(
        state.get("refined_query") or _latest_user_query(state["messages"]))

    try:
        documents = retriever.invoke(query)
    except Exception as exc:
        logger.warning("Retrieval failed (%s) -- answering without context.", exc)
        return {"citations": [], "context": ""}

    if not documents:
        logger.info("No documents retrieved -- answering from the model's own knowledge.")
        return {"citations": [], "context": ""}

    # Drop hits the index scored as weak matches before anything else looks at
    # them: an off-topic question still returns its four nearest neighbours, and
    # summarising those invents an answer and cites papers that never made it.
    if RELEVANCE_GATE:
        scores = [d.metadata.get("score", 0.0) for d in documents]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        if mean_score < RELEVANCE_MIN_SCORE:
            logger.info("Corpus mean score %.3f below %.2f for %r -- answering without context.",
                        mean_score, RELEVANCE_MIN_SCORE, query)
            return {"citations": [], "context": ""}

    summary = _truncate(llm_summarizer(documents, query), MAX_SUMMARY_CHARS)

    # Citations are only emitted when the retrieved papers actually informed the
    # answer. Listing whatever the vector store returned would attribute claims
    # to papers that do not support them.
    if not summary or summary.strip().upper().startswith(NOT_RELEVANT):
        logger.info("Retrieved context not relevant to %r -- answering without it.", query)
        return {"citations": [], "context": ""}

    if summary.lower().strip() in ("not found.", "not found"):
        return {"citations": [], "context": ""}

    sources = sorted({
        os.path.basename(d.metadata["source"])
        for d in documents
        if d.metadata.get("source")
    })
    return {"citations": sources, "context": summary}


def chatbot(state: State) -> dict:
    """Single place where the outbound prompt is assembled.

    Building it here (rather than appending to `state["messages"]`) keeps the
    retrieved context out of the persisted history, so it is not re-sent on
    every subsequent turn.
    """
    context = state.get("context")
    # The assistant specialises in AI/ML but is not limited to it: deflecting a
    # general question with "I only answer ML questions" is a worse answer than
    # looking it up. Searching is preferred over both guessing and declining.
    system = (
        "You are a helpful research assistant. Your speciality is AI and machine "
        "learning, and you also answer general questions on any topic.\n\n"
        "When you are not confident about a fact -- anything recent, or about "
        "people, events, organisations, products, places or figures -- use the "
        "available search tools before answering. Prefer searching over guessing, "
        "and prefer searching over declining to answer. If a lookup fails or you "
        "remain unsure, say so plainly and give what you do know.\n\n"
        "Do not speculate about private matters concerning real people (health, "
        "sexuality, relationships, finances). Decline that specific part briefly, "
        "then answer whatever else was asked from publicly documented facts."
    )
    ml_parts = state.get("ml_parts") or []
    general_parts = state.get("general_parts") or []
    if ml_parts and general_parts:
        # A mixed message: each half has its own best source, and the reply
        # should cover both rather than answering whichever was classified first.
        system += (
            "\n\nThis message contains more than one kind of question. Answer all "
            "of them in a single reply, in the order asked, with a short heading "
            "for each.\n"
            + "".join(f"\n- Research question (use the paper context below): {p}" for p in ml_parts)
            + "".join(f"\n- General question (search if you are unsure): {p}" for p in general_parts)
        )

    if context:
        system += (
            "\n\nUse the following summarized context from retrieved research "
            f"papers to inform your answer:\n\n--- Context Start ---\n{context}\n"
            "--- Context End ---\n\nThe context applies to the research question(s) "
            "only. If it does not cover something, say so and answer from your own "
            "knowledge or by searching."
        )
    else:
        system += (
            "\n\nNo indexed research papers matched this question, so answer from "
            "your own knowledge or by searching. Do not mention the paper index."
        )

    prompt = [SystemMessage(content=system)] + _trim_history(state["messages"])
    # Output tokens count toward the same per-minute budget as input.
    llm = get_llm(max_tokens=MAX_OUTPUT_TOKENS)

    budget_left = state.get("tool_rounds", 0) < MAX_TOOL_ROUNDS
    if not budget_left:
        # Spent the tool budget: ask for a final answer from what is already here.
        prompt = prompt + [SystemMessage(content=(
            "You have no further lookups available. Answer now as fully as you can "
            "from what you already have, and say plainly which parts are uncertain. "
            "Do not decline simply because a lookup was unavailable."))]

    def _plain():
        """Answer with no tools offered.

        Unbinding tools is not enough on its own: a model trained with built-in
        browsing may still emit a call, which the provider rejects. Re-ask with
        an explicit prohibition before giving up.
        """
        guard = SystemMessage(content=(
            "Answer directly, in plain prose. Do NOT call, invoke or emit any "
            "tool, function or browser action -- none are available to you now. "
            "Use what you already know, and state plainly which parts are "
            "uncertain rather than declining to answer."))

        for attempt in range(TOOL_FREE_ATTEMPTS):
            try:
                return llm.invoke(prompt + [guard])
            except Exception as exc:
                if not _is_tool_error(exc):
                    raise
                logger.warning(
                    "Model emitted a tool call with none offered (attempt %d/%d).",
                    attempt + 1, TOOL_FREE_ATTEMPTS)

        # Every attempt was rejected. Say what actually happened rather than
        # blaming a lookup that may well have succeeded.
        reasons = []
        if not state.get("context"):
            reasons.append("the indexed papers do not cover it")
        if DEGRADED:
            reasons.append(f"an external lookup was unavailable ({', '.join(sorted(DEGRADED))})")
        detail = "; ".join(reasons) if reasons else "the model did not return a usable answer"
        logger.warning("Could not obtain a tool-free answer: %s", detail)
        return AIMessage(content=(
            f"I wasn't able to put together a reliable answer here -- {detail}. "
            "Rephrasing the question, or pointing me at a specific paper, would help."))

    if not budget_left:
        return {"messages": [_plain()]}

    try:
        return {"messages": [llm.bind_tools(tools).invoke(prompt)]}
    except Exception as exc:
        if _is_rate_limit(exc):
            # The daily/'per-minute quota is per model, so the smaller model is
            # usually still available. A degraded answer beats a raw 429.
            logger.warning("%s rate limited -- falling back to %s.", CHAT_MODEL, SUMMARY_MODEL)
            small = get_llm(SUMMARY_MODEL, max_tokens=MAX_OUTPUT_TOKENS)
            try:
                return {"messages": [small.bind_tools(tools).invoke(prompt)],
                        "model_fallback": SUMMARY_MODEL}
            except Exception as inner:
                if not _is_tool_error(inner):
                    raise
                return {"messages": [_plain()], "model_fallback": SUMMARY_MODEL}
        # Models trained with built-in browsing sometimes emit a call to a tool
        # that was never offered (e.g. "search", "web_browser.open_file"), which
        # the provider rejects outright. Fall back to answering without tools.
        if not _is_tool_error(exc):
            raise
        logger.warning("Invalid tool call from the model -- answering without tools.")
        return {"messages": [_plain()]}


def prepare_node(state: State) -> dict:
    """Clear the fields that belong to a single turn.

    They are checkpointed with the conversation, so without this a follow-up
    question inherits the previous turn's retrieved context and citations, and
    the critique budget stays spent for the rest of the session.
    """
    return {"context": "", "citations": [], "refined_query": "",
            "ml_parts": [], "general_parts": [], "model_fallback": "",
            "critique_count": 0, "tool_rounds": 0}


def _binary_is_ml(query: str) -> bool:
    """Fallback when the splitter's JSON cannot be parsed."""
    prompt = ('Is the following question related to AI or Machine Learning? '
              f'Answer only "yes" or "no".\n\nQuestion: {query}')
    try:
        return llm_utility.invoke(prompt).content.strip().lower().startswith("yes")
    except Exception as exc:
        if _is_provider_unreachable(exc):
            raise
        logger.warning("Classifier failed (%s) -- defaulting to the retrieval path.", exc)
        return True


def classify_node(state: State) -> dict:
    """Split a message into its AI/ML parts and its general parts.

    A single message can ask both kinds of question at once. Splitting lets the
    ML parts drive corpus retrieval while the general parts are answered by
    search, instead of forcing the whole message down one path.
    """
    query = _latest_user_query(state["messages"])
    prompt = (
        "Split the user's message into self-contained sub-questions and label each.\n"
        '"ml"      = about AI, machine learning, data science or their research literature\n'
        '"general" = anything else\n\n'
        'Reply with JSON only, no prose: {"ml": [...], "general": [...]}\n'
        "If the whole message is a single question, return it in the one list that "
        "fits and leave the other empty. Preserve the user's wording.\n\n"
        f"Message: {query}"
    )
    try:
        raw = llm_utility.invoke(prompt).content.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        ml = [str(x).strip() for x in (data.get("ml") or []) if str(x).strip()]
        general = [str(x).strip() for x in (data.get("general") or []) if str(x).strip()]
        if not ml and not general:
            raise ValueError("empty split")
    except Exception as exc:
        if _is_provider_unreachable(exc):
            raise
        logger.warning("Could not split %r (%s) -- falling back to a single label.", query, exc)
        if _binary_is_ml(query):
            ml, general = [query], []
        else:
            ml, general = [], [query]

    logger.info("Split into %d ML and %d general part(s).", len(ml), len(general))
    return {"ml_parts": ml, "general_parts": general}


def route_after_classify(state: State) -> Literal["QueryFramer", "chatbot"]:
    # Any ML part earns the retrieval path; general parts are handled by the
    # chatbot's tools on the way through.
    return "QueryFramer" if state.get("ml_parts") else "chatbot"


def query_framer_node(state: State) -> dict:
    """Sharpen the query for retrieval only.

    The rephrasing is kept in state rather than pushed into `messages`, so the
    model still answers the question the user actually asked.
    """
    # Only the ML parts should shape the retrieval query: general text in the
    # same message is noise to the vector store.
    ml_parts = state.get("ml_parts") or []
    query = " ".join(ml_parts) if ml_parts else _latest_user_query(state["messages"])
    prompt = (
        "Rephrase the following question to make it clearer and more specific, "
        "suitable for searching an academic paper database. Reply with the "
        f"rephrased query only.\n\nQuestion: {query}\n\nRephrased:"
    )
    try:
        refined = llm_utility.invoke(prompt).content.strip()
    except Exception as exc:
        if _is_provider_unreachable(exc):
            raise
        logger.warning("Query framing failed (%s) -- using the original query.", exc)
        refined = query
    return {"refined_query": _fold_punctuation(refined) or query}


def critique_node(state: State) -> dict:
    """Check the answer against the retrieved context.

    Grounding the check in the context matters: asked to judge a response in
    isolation, the model flags well-supported answers as hallucinated.
    """
    context = state.get("context") or ""
    if CRITIQUE_REQUIRES_CONTEXT and not context:
        # Nothing to check the answer against: skip rather than pay a full
        # regeneration for a guess.
        return {}

    attempts = state.get("critique_count", 0)
    if attempts >= MAX_CRITIQUE_RETRIES:
        return {}

    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not (last.content or "").strip():
        return {}

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


def count_tool_round(state: State) -> dict:
    return {"tool_rounds": state.get("tool_rounds", 0) + 1}


def chatbot_or_end(state: State) -> Literal["tools", "critique"]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None) and state.get("tool_rounds", 0) < MAX_TOOL_ROUNDS:
        return "tools"
    return "critique"


# --- graph ---------------------------------------------------------------

memory = MemorySaver()
graph_builder = StateGraph(State)

graph_builder.add_node("prepare", prepare_node)
graph_builder.add_node("summarizer", summarizer_node)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_node("count_tools", count_tool_round)
graph_builder.add_node("classify", classify_node)
graph_builder.add_node("QueryFramer", query_framer_node)
graph_builder.add_node("critique", critique_node)

graph_builder.add_edge(START, "prepare")
graph_builder.add_edge("prepare", "classify")
graph_builder.add_conditional_edges("classify", route_after_classify)
graph_builder.add_edge("QueryFramer", "summarizer")
graph_builder.add_edge("summarizer", "chatbot")
graph_builder.add_conditional_edges("chatbot", chatbot_or_end)
graph_builder.add_edge("tools", "count_tools")
graph_builder.add_edge("count_tools", "chatbot")
graph_builder.add_conditional_edges("critique", check_critique_result)

graph = graph_builder.compile(checkpointer=memory)
