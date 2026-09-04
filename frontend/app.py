import json
import os
import re
import uuid

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
API_URL = f"{API_BASE}/chat"
RESET_URL = f"{API_BASE}/reset"

st.set_page_config(page_title="RAG Assistant", page_icon="📚")
st.title("📚 LangGraph Research Chatbot")

# Each chat is a separate backend session, so switching chats switches the
# server-side conversation memory too rather than just the on-screen history.
def _new_chat() -> str:
    cid = str(uuid.uuid4())
    st.session_state.chats[cid] = {"title": "New chat", "messages": []}
    st.session_state.active = cid
    return cid


if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.active = None
if not st.session_state.chats:
    _new_chat()

with st.sidebar:
    if st.button("New chat", use_container_width=True):
        _new_chat()
        st.rerun()

    st.caption("Chats")
    for cid, chat in list(st.session_state.chats.items()):
        active = cid == st.session_state.active
        row, delete = st.columns([5, 1])
        if row.button(
            ("● " if active else "") + chat["title"][:28],
            key=f"sel-{cid}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            st.session_state.active = cid
            st.rerun()
        if delete.button("✕", key=f"del-{cid}", help="Delete this chat"):
            try:
                requests.post(RESET_URL, json={"message": "", "session_id": cid}, timeout=30)
            except requests.exceptions.RequestException:
                pass  # the chat still goes away locally
            st.session_state.chats.pop(cid, None)
            if st.session_state.active == cid:
                st.session_state.active = next(iter(st.session_state.chats), None)
            if not st.session_state.chats:
                _new_chat()
            st.rerun()

    st.divider()
    st.caption("Session id")
    st.code(st.session_state.active, language=None)

chat = st.session_state.chats[st.session_state.active]
session_id = st.session_state.active


# --- LaTeX -----------------------------------------------------------------
# Models emit \( inline \) and \[ display \] delimiters, which Streamlit's
# markdown does not recognise; it renders the raw source instead. Rewriting them
# to $ / $$ is safe mid-stream because the patterns require a closing delimiter,
# so a half-arrived equation is simply left alone until its closer turns up.
_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_INLINE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
# Models trained with built-in browsing emit internal reference markers such as
# 【0†content】. They mean nothing to the reader. Requiring the closing bracket
# keeps a half-arrived marker from being stripped mid-stream.
_CITE_ARTIFACT = re.compile(r"\u3010[^\u3011]{0,40}\u3011")


def normalize_math(text: str) -> str:
    text = _CITE_ARTIFACT.sub("", text)
    text = _DISPLAY.sub(lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n", text)
    text = _INLINE.sub(lambda m: f"${m.group(1).strip()}$", text)
    return text


def render_message(msg):
    if msg.get("reasoning"):
        with st.expander("Reasoning", expanded=False):
            st.markdown("\n".join(f"- {r}" for r in msg["reasoning"]))
    for notice in msg.get("notices", []):
        st.warning(notice, icon="⚠️")
    st.markdown(normalize_math(msg["content"]))


for message in chat["messages"]:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_message(message)
        else:
            st.markdown(message["content"])


def consume(response, reason_box, answer_box):
    """Read the NDJSON event stream, buffering until each newline.

    Reasoning renders above the answer because `reason_box` was created first.
    """
    answer, reasoning, notices, sources = "", [], [], []
    buffer = ""

    def paint_reasoning():
        with reason_box.container():
            with st.expander("Reasoning", expanded=False):
                st.markdown("\n".join(f"- {r}" for r in reasoning))

    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        if not chunk:
            continue
        # requests only decodes when the response declares a charset; decode
        # defensively so the client does not depend on that header.
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")
        buffer += chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial or malformed line: skip rather than crash

            kind = ev.get("type")
            if kind == "token":
                answer += ev.get("text", "")
                answer_box.markdown(normalize_math(answer) + " ▌")
            elif kind == "supersede":
                answer = ""  # a revision replaces the draft rather than appending
            elif kind == "reason":
                reasoning.append(ev.get("text", ""))
                paint_reasoning()
            elif kind == "notice":
                notices.append(ev.get("text", ""))
            elif kind == "sources":
                sources = ev.get("items", [])
            elif kind == "error":
                answer += f"\n\n⚠️ {ev.get('text','')}"

    if sources:
        answer += "\n\n**Sources:** " + ", ".join(sources)
    answer_box.markdown(normalize_math(answer))
    return answer, reasoning, notices


if prompt := st.chat_input("Ask a question about AI/ML research"):
    with st.chat_message("user"):
        st.markdown(prompt)
    chat["messages"].append({"role": "user", "content": prompt})
    if chat["title"] == "New chat":
        chat["title"] = prompt[:40]

    with st.chat_message("assistant"):
        reason_box = st.empty()   # created first, so reasoning sits above
        notice_box = st.empty()
        answer_box = st.empty()
        try:
            with requests.post(
                API_URL,
                json={"message": prompt, "session_id": session_id},
                stream=True,
                timeout=(10, 300),
            ) as response:
                response.raise_for_status()
                answer, reasoning, notices = consume(response, reason_box, answer_box)

            for n in notices:
                notice_box.warning(n, icon="⚠️")
            chat["messages"].append({
                "role": "assistant", "content": answer,
                "reasoning": reasoning, "notices": notices,
            })
            # The sidebar was drawn before this turn was handled, so redraw it
            # to pick up a newly derived chat title.
            st.rerun()
        except requests.exceptions.RequestException as e:
            st.error(
                f"Error calling API: {e}\n\n"
                f"Is the backend running at {API_BASE}? Start it with:\n"
                "`uvicorn src.api.main:app --port 8000`"
            )
