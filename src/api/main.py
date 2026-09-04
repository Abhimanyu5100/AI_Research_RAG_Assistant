import json
import logging
import re
import os
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import RemoveMessage
from pydantic import BaseModel, Field

from src.agent.graph import graph
from src.agent.tools import DEGRADED
from src.config import CHAT_MODEL

# The stream is newline-delimited JSON so that reasoning steps, answer tokens,
# citations and notices can share one connection. A client buffers until it sees
# a newline, which also makes partial chunks safe to handle.
#   {"type":"reason",    "text": ...}  progress step, shown above the answer
#   {"type":"token",     "text": ...}  answer text
#   {"type":"supersede"}               discard answer streamed so far
#   {"type":"sources",   "items":[..]} citations
#   {"type":"notice",    "text": ...}  degraded capability warning
#   {"type":"error",     "text": ...}  failure

STEP_LABELS = {
    "classify": "Sorting the question",
    "QueryFramer": "Reframing the question for retrieval",
    "summarizer": "Searching the paper corpus",
    "tools": "Calling external sources",
    "chatbot": "Drafting the answer",
    "critique": "Checking the answer against sources",
}


def _event(**kw) -> str:
    return json.dumps(kw, ensure_ascii=False) + "\n"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Research RAG Assistant")

# allow_credentials=True is silently ignored by browsers when the origin is a
# wildcard, so the pairing is misleading. This API is unauthenticated and uses
# no cookies, so declare that honestly and let deployments restrict origins.
ALLOWED_ORIGINS = [o for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id"],
)


MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "4000"))


class QueryRequest(BaseModel):
    # Bounded so a single caller cannot blow the model's token budget (or the
    # bill) with one enormous prompt.
    message: str = Field(default="", max_length=MAX_MESSAGE_CHARS)
    # Conversation memory is keyed on this. A shared constant would mean every
    # caller reads and writes one another's history, and that history would
    # grow without bound until every request exceeded the token limit.
    session_id: Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "ok", "model": CHAT_MODEL}


@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty")

    session_id = request.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    async def stream_generator():
        streamed_any = False
        current_step = None
        started = time.monotonic()
        # Best-effort: this registry is process-wide, so with concurrent requests
        # a notice may reflect another caller's failed lookup.
        DEGRADED.clear()

        def elapsed() -> str:
            return f"{time.monotonic() - started:.1f}s"

        try:
            async for mode, chunk in graph.astream(
                {"messages": [("user", request.message)]},
                stream_mode=["updates", "messages"],
                config=config,
            ):
                if mode == "updates":
                    for node, payload in (chunk or {}).items():
                        label = STEP_LABELS.get(node)
                        if not label:
                            continue
                        detail = ""
                        if node == "classify" and isinstance(payload, dict):
                            ml = payload.get("ml_parts") or []
                            gen = payload.get("general_parts") or []
                            if ml and gen:
                                detail = (f": {len(ml)} research + {len(gen)} general "
                                          "part(s), handled separately")
                            elif ml:
                                detail = ": research question"
                            else:
                                detail = ": general question"
                        elif node == "QueryFramer" and isinstance(payload, dict):
                            refined = (payload.get("refined_query") or "").strip()
                            if refined:
                                detail = f": {refined}"
                        elif node == "summarizer" and isinstance(payload, dict):
                            cites = payload.get("citations") or []
                            detail = (f": {len(cites)} paper(s) used"
                                      if cites else ": no relevant papers, using general knowledge")
                        yield _event(type="reason", text=f"[{elapsed()}] {label}{detail}")
                    continue

                msg, metadata = chunk
                if metadata.get("langgraph_node") != "chatbot" or not msg.content:
                    continue

                step = metadata.get("langgraph_step")
                if streamed_any and step != current_step:
                    yield _event(type="supersede")
                current_step = step

                streamed_any = True
                yield _event(type="token", text=msg.content)

            state = await graph.aget_state(config)
            values = state.values or {}

            if DEGRADED:
                names = ", ".join(sorted(DEGRADED))
                used_corpus = bool((values or {}).get("citations"))
                fallback = ("the indexed papers and the model's own knowledge"
                            if used_corpus else "the model's own knowledge")
                yield _event(type="notice", text=(
                    f"External lookup unavailable ({names}). Answered from {fallback}."))

            fallback_model = values.get("model_fallback")
            if fallback_model:
                yield _event(type="notice", text=(
                    f"The main model was rate limited, so this was answered by "
                    f"{fallback_model}. Quality may be lower."))

            citations = values.get("citations") or []
            if citations:
                yield _event(type="sources", items=citations)

            yield _event(type="reason", text=f"[{elapsed()}] Done")

        except Exception as exc:
            logger.exception("Error while streaming response")
            text = str(exc)
            low = text.lower()
            if "error code: 403" in low or "access denied" in low:
                yield _event(type="error", text=(
                    "The model provider refused the connection (HTTP 403). This is "
                    "usually the network rather than the app: Groq blocks many VPN, "
                    "proxy and datacentre IP ranges. Try disconnecting a VPN or "
                    "switching network, then check with:\n"
                    "  curl -s -o /dev/null -w '%{http_code}' https://api.groq.com/openai/v1/models"))
            elif "error code: 401" in low or "authentication" in low:
                yield _event(type="error", text=(
                    "The model provider rejected the API key (HTTP 401). Check "
                    "GROQ_API_KEY in your .env file."))
            elif "rate_limit_exceeded" in text or "Error code: 429" in text:
                wait = re.search(r"try again in ([0-9hms.]+)", text)
                friendly = "The model provider's rate limit has been reached."
                if wait:
                    friendly += f" It should clear in about {wait.group(1)}."
                friendly += (" You can also set GROQ_CHAT_MODEL to a smaller model "
                             "in .env, or upgrade the Groq plan.")
                yield _event(type="error", text=friendly)
            else:
                yield _event(type="error",
                             text=f"The assistant hit an error: {type(exc).__name__}: {exc}")

    return StreamingResponse(
        stream_generator(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={"X-Session-Id": session_id},
    )


@app.post("/reset")
async def reset(request: QueryRequest):
    """Drop a conversation's stored history."""
    session_id = request.session_id
    if not session_id:
        return {"status": "error", "detail": "session_id is required"}
    thread = {"configurable": {"thread_id": session_id}}
    try:
        state = await graph.aget_state(thread)
        existing = (state.values or {}).get("messages", [])
        # An empty list is a no-op for the add_messages reducer -- clearing
        # history requires an explicit RemoveMessage per stored message.
        await graph.aupdate_state(
            thread,
            {
                "messages": [RemoveMessage(id=m.id) for m in existing if m.id],
                "context": "",
                "citations": [],
                "refined_query": "",
                "critique_count": 0,
            },
        )
    except Exception as exc:
        logger.warning("Reset failed for %s: %s", session_id, exc)
        return {"status": "error", "detail": str(exc)}
    return {"status": "ok", "session_id": session_id}
