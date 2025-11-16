import os, re, json
from io import BytesIO
from typing import List, Dict, Tuple
import numpy as np
import streamlit as st
from openai import OpenAI

try:
    import pypdf
except ImportError:
    pypdf = None

# ------------ Config ------------
RESUME_PDF_FILENAME = "Sheela_Pothini_Resume.pdf"   # <- ensure this file is next to app.py
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
INDEX_FILE = "resume_index.json"         # persisted on server

# ------------ Key handling ------------
def get_api_key() -> str:
    # Recruiters should NOT enter a key; we read from server-side secrets or env
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    return os.environ.get("OPENAI_API_KEY", "")

# ------------ Utils ------------
def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def split_text(text: str, chunk_size=900, overlap=200) -> List[str]:
    text = clean_text(text)
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        out.append(text[start:end])
        if end == len(text): break
        start = max(0, end - overlap)
    return out

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)

def embed(client: OpenAI, texts: List[str]) -> List[List[float]]:
    if not texts: return []
    r = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in r.data]

def extract_pdf_text(file_bytes: bytes) -> str:
    if pypdf is None:
        raise RuntimeError("Install pypdf (see requirements.txt)")
    reader = pypdf.PdfReader(BytesIO(file_bytes))
    pages = []
    for pg in reader.pages:
        try:
            pages.append(pg.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def build_index_from_text(client: OpenAI, text: str) -> Dict:
    chunks = split_text(text)
    vecs = embed(client, chunks)
    idx = {"chunks": chunks, "vecs": vecs, "embed_model": EMBED_MODEL}
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(idx, f)
    return idx

def load_index_if_any() -> Dict:
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def retrieve(index: Dict, query: str, client: OpenAI, k: int = 4) -> List[str]:
    if not index: return []
    qvec = embed(client, [query])[0]
    scores: List[Tuple[float, str]] = []
    for v, c in zip(index["vecs"], index["chunks"]):
        scores.append((cosine(np.array(qvec), np.array(v)), c))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scores[:k]]

def answer(client: OpenAI, query: str, contexts: List[str], tone: str, audience: str) -> str:
    system = (
        "You are a concise, recruiter-facing résumé assistant. "
        "Answer ONLY from the provided context; if not present, say so briefly. "
        "Prefer 3–5 bullets; keep under 120 words unless asked."
    )
    ctx = "\n\n".join([f"[Context {i+1}] {c}" for i, c in enumerate(contexts)])
    user = f"AUDIENCE: {audience}\nTONE: {tone}\nRésumé Context:\n{ctx}\n\nQuestion: {query}"
    r = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2,
    )
    return r.choices[0].message.content.strip()

# ------------ App ------------
st.set_page_config(page_title="Résumé in Dialogue", page_icon="🗂️", layout="centered")
st.title("🗂️ Résumé in Dialogue")

# Tone/audience controls (visible to visitors)
col1, col2, col3 = st.columns(3)
with col1:
    tone = st.selectbox("Tone", ["professional", "executive", "friendly", "enthusiastic"], index=0)
with col2:
    audience = st.selectbox("Audience", ["General", "SWE", "PM", "Data", "AI/ML"], index=0)
with col3:
    k = st.slider("Evidence passages", 1, 8, 4)

# Show a download button for the bundled PDF
if os.path.exists(RESUME_PDF_FILENAME):
    with open(RESUME_PDF_FILENAME, "rb") as f:
        st.download_button("⬇️ Download my PDF résumé", f, file_name="Resume.pdf")

# Prepare OpenAI client (server-side key only)
api_key = get_api_key()
if not api_key:
    st.error("Server is missing OPENAI_API_KEY (set in Streamlit **Secrets** or env).")
    st.stop()
client = OpenAI(api_key=api_key)

# Build/load index automatically (no upload UI)
@st.cache_resource(show_spinner=True)
def get_or_create_index() -> Dict:
    # 1) load existing index if present
    idx = load_index_if_any()
    if idx:
        return idx
    # 2) otherwise read the bundled PDF and build once
    if not os.path.exists(RESUME_PDF_FILENAME):
        raise FileNotFoundError(f"{RESUME_PDF_FILENAME} not found. Add it next to app.py.")
    with open(RESUME_PDF_FILENAME, "rb") as f:
        pdf_bytes = f.read()
    text = extract_pdf_text(pdf_bytes)
    if not text.strip():
        raise ValueError("Résumé PDF has no extractable text. Export a text-based PDF (not scanned).")
    return build_index_from_text(client, text)

with st.spinner("Preparing résumé index…"):
    index = get_or_create_index()

# Simple legacy-safe chat UI (works across Streamlit versions)
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    who = "You" if m["role"] == "user" else "Assistant"
    st.markdown(f"**{who}:** {m['content']}")

user_col, send_col = st.columns([4,1])
with user_col:
    q = st.text_input("Ask My Résumé – Immediate Insights into My Career Achievements")
with send_col:
    send = st.button("Send", use_container_width=True)

if send and q.strip():
    question = q.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(f"**You:** {question}")

    ctx = retrieve(index, question, client, k=k)
    a = answer(client, question, ctx, tone=tone, audience=audience)

    st.session_state.messages.append({"role": "assistant", "content": a})
    st.markdown(f"**Assistant:** {a}")

    with st.expander("Sources used"):
        if ctx:
            for i, c in enumerate(ctx, 1):
                st.write(f"**Source {i}:** {c[:600]}{'…' if len(c) > 600 else ''}")
        else:
            st.write("No sources.")
