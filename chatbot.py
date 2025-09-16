import os
import sqlite3
from typing import List, Tuple

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


# ----------------- CONFIG -----------------
DB_PATH = os.getenv("ATTENDANCE_DB_PATH", "data/attendance.db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Configure only if key is present; otherwise defer and show a warning in UI
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


# ----------------- DATABASE -----------------
def get_attendance_rows(db_path: str) -> List[Tuple[str, str, str]]:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        # Try to read known schema: attendance(student_name, date, time, confidence)
        # If user has 'status' column, fall back to that
        try:
            cur.execute("SELECT student_name, date, time FROM attendance")
            rows = cur.fetchall()
            # Map time -> status-like string for readability
            data = [(r[0], str(r[1]), f"time:{r[2]}") for r in rows]
        except Exception:
            cur.execute("SELECT student_name, date, status FROM attendance")
            rows = cur.fetchall()
            data = [(r[0], str(r[1]), str(r[2])) for r in rows]
        conn.close()
        return data
    except Exception:
        return []


def format_rows_for_rag(rows: List[Tuple[str, str, str]]) -> List[str]:
    docs = []
    for student, dt, status in rows:
        docs.append(f"Student: {student} | Date: {dt} | Status: {status}")
    return docs


# ----------------- VECTOR STORE -----------------
@st.cache_resource(show_spinner=False)
def build_faiss_index(texts: List[str], _embedder: SentenceTransformer):
    if not texts:
        # Create a dummy index to avoid errors downstream
        dim = 384  # all-MiniLM-L6-v2 output dim
        index = faiss.IndexFlatIP(dim)
        return index, np.zeros((0, dim), dtype=np.float32)
    embeddings = _embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index, embeddings


def retrieve(query: str, texts: List[str], index, embedder: SentenceTransformer, k: int = 5) -> List[str]:
    if not texts:
        return []
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    D, I = index.search(q_emb, min(k, len(texts)))
    return [texts[i] for i in I[0] if 0 <= i < len(texts)]


# ----------------- GEMINI CHATBOT -----------------
def ask_gemini(query: str, retrieved_data: List[str]) -> str:
    # Use a default model that supports generate_content in current API versions
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        return f"LLM init failed: {e}"

    context = "\n".join(retrieved_data) if retrieved_data else "(no matching records found)"
    prompt = f"""
You are an attendance assistant. Answer using only the provided records when possible.

User query:
{query}

Relevant database records:
{context}

Be concise, accurate, and cite dates and student names from the records when relevant.
If information is missing, clearly say so.
"""
    try:
        resp = model.generate_content(prompt)
        if hasattr(resp, "text") and resp.text:
            return resp.text
        # Fallback if SDK returns candidates
        if hasattr(resp, "candidates") and resp.candidates:
            parts = getattr(resp.candidates[0], "content", None)
            if parts and hasattr(parts, "parts") and parts.parts:
                return "".join(getattr(p, "text", "") for p in parts.parts) or "(No response)"
        return "(No response)"
    except Exception as e:
        return f"LLM error: {e}"


# ----------------- STREAMLIT APP -----------------
def render_chatbot():
    # Do not set page config here to avoid conflicts when embedded in dashboard
    st.title("🤖 Smart Attendance Chatbot")
    st.caption("Ask about attendance; answers are grounded in your SQLite database.")

    if not GOOGLE_API_KEY:
        st.warning("Missing Google API key. Set GOOGLE_API_KEY env var or add st.secrets['google_api_key'].")

    embedder = load_embedder(EMBEDDING_MODEL_NAME)

    # Load and index DB on first run or when user refreshes
    if "rag_ready" not in st.session_state:
        rows = get_attendance_rows(DB_PATH)
        texts = format_rows_for_rag(rows)
        index, _ = build_faiss_index(texts, embedder)
        st.session_state.rows = rows
        st.session_state.texts = texts
        st.session_state.index = index
        st.session_state.rag_ready = True

    with st.sidebar:
        st.subheader("Data (Chatbot)")
        st.write(f"DB Path: `{DB_PATH}`")
        st.write(f"Records: {len(st.session_state.texts)}")
        if st.button("Reload Database (Chatbot)", use_container_width=True, key="reload_db_chatbot"):
            rows = get_attendance_rows(DB_PATH)
            texts = format_rows_for_rag(rows)
            index, _ = build_faiss_index(texts, embedder)
            st.session_state.rows = rows
            st.session_state.texts = texts
            st.session_state.index = index
            st.success("Database reloaded and re-indexed")

        top_k = st.slider("Top-K context (Chatbot)", 1, 10, 5, key="topk_chatbot")

    # Chat UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_query = st.chat_input("Ask about student attendance...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Retrieve and answer
        retrieved = retrieve(user_query, st.session_state.texts, st.session_state.index, embedder, k=top_k)
        answer = ask_gemini(user_query, retrieved)

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


def main():
    st.set_page_config(page_title="Attendance Chatbot", page_icon="🤖", layout="centered")
    render_chatbot()

if __name__ == "__main__":
    main()


