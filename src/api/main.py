import logging
import os
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import RemoveMessage
from pydantic import BaseModel, Field

from src.agent.graph import graph
from src.config import CHAT_MODEL

# Emitted when the critique step supersedes a draft answer: everything streamed
# before it should be discarded by the client. Form feed will not occur in
# normal model output.
SUPERSEDE = "\x0c"

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
        try:
            async for msg, metadata in graph.astream(
                # Per-turn state is cleared by the graph's `prepare` node, so
                # callers only have to supply the new message.
                {"messages": [("user", request.message)]},
                stream_mode="messages",
                config=config,
            ):
                if metadata.get("langgraph_node") != "chatbot" or not msg.content:
                    continue

                # A second pass through `chatbot` means the critique node asked
                # for a revision, so the draft already on the wire is stale.
                step = metadata.get("langgraph_step")
                if streamed_any and step != current_step:
                    yield SUPERSEDE
                current_step = step

                streamed_any = True
                yield msg.content

            state = await graph.aget_state(config)
            citations = (state.values or {}).get("citations") or []
            if citations:
                yield "\n\n**Sources:** " + ", ".join(citations)

        except Exception as exc:
            # The 200 and headers are already on the wire by this point, so the
            # only way to report a failure is in the body. Without this the
            # client just sees the connection cut ("response ended prematurely").
            logger.exception("Error while streaming response")
            prefix = "\n\n" if streamed_any else ""
            yield f"{prefix}⚠️ The assistant hit an error: {type(exc).__name__}: {exc}"

    return StreamingResponse(
        stream_generator(),
        media_type="text/plain",
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
