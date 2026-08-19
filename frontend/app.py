import os
import uuid

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
# The backend sends this when a draft answer has been revised; everything
# streamed before it must be dropped.
SUPERSEDE = "\x0c"
API_URL = f"{API_BASE}/chat"
RESET_URL = f"{API_BASE}/reset"

st.set_page_config(page_title="RAG Assistant", page_icon="📚")
st.title("📚 LangGraph Research Chatbot")

# One conversation per browser session, so concurrent users do not share
# (or overwrite) each other's history on the backend.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.caption("Session")
    st.code(st.session_state.session_id, language=None)
    if st.button("🗑️ Clear conversation"):
        try:
            requests.post(
                RESET_URL, json={"message": "", "session_id": st.session_state.session_id}, timeout=30
            )
        except requests.exceptions.RequestException:
            pass
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def render_stream(res, placeholder):
    """Stream into `placeholder`, restarting if the backend supersedes a draft."""
    buffer = ""
    for chunk in res.iter_content(chunk_size=None, decode_unicode=True):
        if not chunk:
            continue
        if SUPERSEDE in chunk:
            # Keep only what follows the final marker: the revised answer.
            buffer = chunk.split(SUPERSEDE)[-1]
        else:
            buffer += chunk
        placeholder.markdown(buffer + "▌")
    placeholder.markdown(buffer)
    return buffer


if prompt := st.chat_input("Ask a question about AI/ML research"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            with requests.post(
                API_URL,
                json={"message": prompt, "session_id": st.session_state.session_id},
                stream=True,
                timeout=(10, 300),
            ) as response:
                response.raise_for_status()
                full_response = render_stream(response, st.empty())
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except requests.exceptions.RequestException as e:
            st.error(
                f"Error calling API: {e}\n\n"
                f"Is the backend running at {API_BASE}? Start it with:\n"
                "`uvicorn src.api.main:app --port 8000`"
            )
