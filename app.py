

import os
import warnings
import pdfplumber
import pandas as pd
from io import StringIO
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore")


from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from pinecone import Pinecone



# PAGE CONFIG  streamlit call


st.set_page_config(
    page_title = "FinanceIQ — Bank Statement Analyzer",
    page_icon  = "💳",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)



# theme


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary   : #0a0e1a;
    --bg-card      : #111827;
    --bg-input     : #1a2235;
    --accent       : #00d4aa;
    --accent-dim   : #00d4aa22;
    --accent-2     : #4f7cff;
    --text-primary : #e8edf5;
    --text-muted   : #6b7a99;
    --border       : #1e2d45;
    --success      : #00d4aa;
    --warning      : #f59e0b;
    --error        : #ef4444;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1020 100%);
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* ── Main header ── */
.main-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}

.main-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -1px;
    margin: 0;
}

.main-header p {
    color: var(--text-muted);
    font-size: 1rem;
    margin: 0.5rem 0 0;
    font-weight: 300;
}

/* ── Status badges ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.5px;
}
.badge-green  { background:#00d4aa22; color:#00d4aa; border:1px solid #00d4aa44; }
.badge-blue   { background:#4f7cff22; color:#4f7cff; border:1px solid #4f7cff44; }
.badge-yellow { background:#f59e0b22; color:#f59e0b; border:1px solid #f59e0b44; }

/* ── Stat cards ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}

.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}

.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
}

.stat-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'Space Mono', monospace;
}

.stat-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--accent);
    margin-top: 4px;
    font-family: 'Space Mono', monospace;
}

/* ── Chat messages ── */
.msg-user {
    background: var(--accent-dim);
    border: 1px solid #00d4aa33;
    border-radius: 12px 12px 4px 12px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    margin-left: 15%;
    position: relative;
}

.msg-user::before {
    content: '👤 YOU';
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent);
    display: block;
    margin-bottom: 6px;
    letter-spacing: 1px;
}

.msg-bot {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px 12px 12px 4px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    margin-right: 15%;
}

.msg-bot::before {
    content: '🤖 FINANCEIQ';
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent-2);
    display: block;
    margin-bottom: 6px;
    letter-spacing: 1px;
}

.msg-calc {
    background: #f59e0b11;
    border: 1px solid #f59e0b33;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #f59e0b;
}

.msg-calc::before {
    content: '🔢 PYTHON CALCULATOR';
    font-size: 0.65rem;
    display: block;
    margin-bottom: 6px;
    letter-spacing: 1px;
}

/* ── Upload area ── */
.upload-zone {
    border: 2px dashed var(--border);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    background: var(--bg-input);
    transition: border-color 0.2s;
}

.upload-zone:hover {
    border-color: var(--accent);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #00b894 100%) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.5px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px #00d4aa44 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* ── Text input ── */
.stTextInput > div > div > input,
.stChatInputContainer textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Info/success boxes ── */
.stSuccess {
    background: #00d4aa11 !important;
    border: 1px solid #00d4aa33 !important;
    border-radius: 10px !important;
}

.stInfo {
    background: #4f7cff11 !important;
    border: 1px solid #4f7cff33 !important;
    border-radius: 10px !important;
}

.stWarning {
    background: #f59e0b11 !important;
    border: 1px solid #f59e0b33 !important;
    border-radius: 10px !important;
}

.stError {
    background: #ef444411 !important;
    border: 1px solid #ef444433 !important;
    border-radius: 10px !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: var(--accent) !important;
}

/* ── Sidebar labels ── */
.sidebar-section {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 1.5rem 0 0.5rem;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}

/* ── Scrollable chat area ── */
.chat-container {
    max-height: 60vh;
    overflow-y: auto;
    padding-right: 4px;
}

/* ── Calculated results box ── */
.calc-result {
    background: #0d1f2d;
    border-left: 3px solid var(--accent);
    padding: 0.8rem 1rem;
    border-radius: 0 8px 8px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--accent);
    margin: 0.5rem 0;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)



# LOAD ENVIRONMENT


load_dotenv()


def get_env(key: str) -> str:

    try:
        return st.secrets[key]

    except Exception:
        return os.getenv(key, "")


NVIDIA_EMBEDDING_API_KEY = get_env("NVIDIA_EMBEDDING_API_KEY")
NVIDIA_LLM_API_KEY       = get_env("NVIDIA_LLM_API_KEY")
PINECONE_API_KEY         = get_env("PINECONE_API_KEY")
PINECONE_INDEX_NAME      = get_env("PINECONE_INDEX_NAME")

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY



# CALCULATION KEYWORDS


CALCULATION_KEYWORDS = [
    "total", "sum", "how much", "calculate", "average", "avg",
    "count", "how many", "maximum", "minimum", "largest", "smallest",
    "highest", "lowest", "balance", "closing", "opening", "spent",
    "received", "earned", "paid", "withdrawn", "deposited"
]



#  RAG FUNCTIONS


@st.cache_resource
def get_nvidia_embeddings():

    return NVIDIAEmbeddings(
        model   = "nvidia/llama-3.2-nemoretriever-300m-embed-v1",
        api_key = NVIDIA_EMBEDDING_API_KEY,
        truncate= "NONE",
    )

@st.cache_resource
def get_qwen_llm():
    """Cached Qwen3.5 LLM client."""
    return ChatNVIDIA(
        model                 = "qwen/qwen3.5-122b-a10b",
        api_key               = NVIDIA_LLM_API_KEY,
        temperature           = 0.0,
        max_completion_tokens = 2048,
    )

@st.cache_resource
def get_pinecone_index():
    """Cached Pinecone connection."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    new_columns = []
    for i, col in enumerate(df.columns):
        if pd.isna(col) or str(col).strip() == '' or str(col) == 'nan':
            new_columns.append(f"col_{i}")
        else:
            new_columns.append(str(col).strip())
    df.columns = new_columns

    for col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace('\n', ' ', regex=False)
            .str.replace('\r', ' ', regex=False)
            .str.replace('\t', ' ', regex=False)
            .str.replace('"', "'", regex=False)
            .str.strip()
        )
    return df


def clean_metadata_value(value) -> str:
    """Convert any metadata value to a safe plain string for Pinecone."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace('\x00', '').replace('\r', ' ')
    return text[:10000]


def extract_pdf_content(uploaded_file) -> dict:
    """
    Extracts text and tables from an uploaded PDF file.

    """
    text_docs  = []
    dataframes = []

    with pdfplumber.open(uploaded_file) as pdf:
        for page_num, page in enumerate(pdf.pages):

            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_docs.append(Document(
                    page_content = page_text,
                    metadata     = {
                        "page"      : page_num + 1,
                        "type"      : "text",
                        "chunk_type": "text"
                    }
                ))

            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                df = pd.DataFrame(table[1:], columns=table[0])
                df = df.dropna(how='all').fillna('')
                dataframes.append({"df": df, "page": page_num + 1})

    return {"text_docs": text_docs, "dataframes": dataframes}


def chunk_by_transactions(extracted: dict, rows_per_chunk: int = 15) -> list:
    """Chunk text by lines, tables by complete rows."""
    all_chunks = []

    for doc in extracted["text_docs"]:
        lines = doc.page_content.split("\n")
        for i in range(0, len(lines), 20):
            chunk_text = "\n".join(lines[i: i + 20])
            if chunk_text.strip():
                all_chunks.append(Document(
                    page_content = chunk_text,
                    metadata     = {**doc.metadata, "chunk_type": "text"}
                ))

    for table_info in extracted["dataframes"]:
        df   = clean_dataframe(table_info["df"])
        page = table_info["page"]

        for start_row in range(0, len(df), rows_per_chunk):
            end_row  = min(start_row + rows_per_chunk, len(df))
            chunk_df = df.iloc[start_row:end_row]

            all_chunks.append(Document(
                page_content = chunk_df.to_string(index=False),
                metadata     = {
                    "page"      : int(page),
                    "type"      : "table",
                    "chunk_type": "transaction_rows",
                    "start_row" : int(start_row),
                    "end_row"   : int(end_row),
                    "csv_data"  : clean_metadata_value(chunk_df.to_csv(index=False)),
                    "columns"   : clean_metadata_value(", ".join(df.columns.tolist()))
                }
            ))

    return all_chunks


def index_to_pinecone(chunks: list) -> PineconeVectorStore:
    """Embed chunks with NVIDIA and store in Pinecone."""
    embeddings  = get_nvidia_embeddings()
    vectorstore = PineconeVectorStore.from_documents(
        documents  = chunks,
        embedding  = embeddings,
        index_name = PINECONE_INDEX_NAME
    )
    return vectorstore


def build_retriever(vectorstore: PineconeVectorStore):
    """Build a similarity retriever from the vector store."""
    return vectorstore.as_retriever(
        search_type   = "similarity",
        search_kwargs = {"k": 8}
    )


def run_calculator(retrieved_docs: list) -> str:
    """Run real Python math on retrieved transaction rows."""
    all_rows = []
    for doc in retrieved_docs:
        if doc.metadata.get("chunk_type") == "transaction_rows":
            csv_data = doc.metadata.get("csv_data", "")
            if csv_data:
                try:
                    chunk_df = pd.read_csv(StringIO(csv_data))
                    all_rows.append(chunk_df)
                except Exception:
                    continue

    if not all_rows:
        return ""

    combined_df = pd.concat(all_rows, ignore_index=True)

    for col in combined_df.columns:
        combined_df[col] = pd.to_numeric(
            combined_df[col].astype(str)
                .str.replace(',', '', regex=False)
                .str.replace('₹', '', regex=False)
                .str.replace('$', '', regex=False)
                .str.replace('£', '', regex=False)
                .str.strip(),
            errors='coerce'
        )

    debit_col = credit_col = balance_col = None
    for col in combined_df.columns:
        if col is None: continue
        c = str(col).lower().strip()
        if any(x in c for x in ['debit', 'withdrawal', 'dr', 'spent']):
            debit_col = col
        if any(x in c for x in ['credit', 'deposit', 'cr', 'received']):
            credit_col = col
        if 'balance' in c:
            balance_col = col

    lines = [f"Transactions Analyzed : {len(combined_df)}"]

    if debit_col:
        lines.append(f"Total Debits          : {combined_df[debit_col].dropna().sum():,.2f}")
        lines.append(f"Largest Debit         : {combined_df[debit_col].dropna().max():,.2f}")
        lines.append(f"Average Debit         : {combined_df[debit_col].dropna().mean():,.2f}")
        lines.append(f"Debit Count           : {int(combined_df[debit_col].dropna().count())}")

    if credit_col:
        lines.append(f"Total Credits         : {combined_df[credit_col].dropna().sum():,.2f}")
        lines.append(f"Largest Credit        : {combined_df[credit_col].dropna().max():,.2f}")
        lines.append(f"Average Credit        : {combined_df[credit_col].dropna().mean():,.2f}")
        lines.append(f"Credit Count          : {int(combined_df[credit_col].dropna().count())}")

    if balance_col:
        non_null = combined_df[balance_col].dropna()
        if len(non_null) > 0:
            lines.append(f"Opening Balance       : {non_null.iloc[0]:,.2f}")
            lines.append(f"Closing Balance       : {non_null.iloc[-1]:,.2f}")

    return "\n".join(lines)


def answer_question(question: str, retriever) -> tuple[str, str]:
    """
    Retrieves context, optionally calculates, then asks Qwen3.5.
    Returns (answer, calculated_result)
    """
    retrieved_docs = retriever.invoke(question)

    needs_calc = any(kw in question.lower() for kw in CALCULATION_KEYWORDS)
    calculated = run_calculator(retrieved_docs) if needs_calc else ""

    formatted_context = "\n\n".join([
        f"--- Chunk {i+1} | Page {doc.metadata.get('page','?')} ---\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs)
    ])

    prompt = ChatPromptTemplate.from_template("""
You are an expert financial analyst AI assistant analyzing a bank statement.

RULES:
1. Answer ONLY from the context and calculated results below
2. NEVER guess any number — use calculated results if provided
3. Mention exact dates, descriptions, and amounts
4. Format amounts clearly: ₹1,25,000.00 or $1,250.00
5. If not found, say: "Not found in the statement"

---STATEMENT CONTEXT---
{context}

---CALCULATED RESULTS (use these exact numbers)---
{calculated}

---QUESTION---
{question}

---ANSWER---
""")

    llm   = get_qwen_llm()
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context"   : formatted_context,
        "calculated": calculated if calculated else "No calculation needed.",
        "question"  : question
    })

    return answer, calculated



# SESSION STATE INITIALIZATION


if "chat_history"  not in st.session_state:
    st.session_state.chat_history  = []
    # Stores list of {"role": "user"/"bot", "content": "...", "calc": "..."}

if "retriever"     not in st.session_state:
    st.session_state.retriever     = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "pdf_stats"     not in st.session_state:
    st.session_state.pdf_stats     = {}



# SIDEBAR


with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding-bottom:1rem;'>
        <div style='font-family:Space Mono,monospace; font-size:1.3rem;
                    color:#00d4aa; font-weight:700;'>💳 FinanceIQ</div>
        <div style='color:#6b7a99; font-size:0.8rem;'>Bank Statement Analyzer</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">📄 Upload Statement</div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label       = "Upload your bank statement PDF",
        type        = ["pdf"],
        label_visibility = "collapsed",
        help        = "Supports any bank statement in PDF format"
    )

    if uploaded_file:
        st.markdown(f"""
        <div style='background:#00d4aa11; border:1px solid #00d4aa33;
                    border-radius:8px; padding:0.7rem; margin:0.5rem 0;
                    font-size:0.82rem;'>
            📎 <b>{uploaded_file.name}</b><br>
            <span style='color:#6b7a99;'>
                {uploaded_file.size / 1024:.1f} KB
            </span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("⚡ Process & Index PDF", use_container_width=True):
            with st.spinner("Extracting text and tables..."):
                extracted = extract_pdf_content(uploaded_file)

            with st.spinner("Creating transaction-safe chunks..."):
                chunks = chunk_by_transactions(extracted)

            with st.spinner(f"Embedding {len(chunks)} chunks with NVIDIA..."):
                vectorstore = index_to_pinecone(chunks)
                st.session_state.retriever = build_retriever(vectorstore)

            st.session_state.pdf_processed = True
            st.session_state.chat_history  = []
            st.session_state.pdf_stats     = {
                "text_sections": len(extracted["text_docs"]),
                "tables"       : len(extracted["dataframes"]),
                "chunks"       : len(chunks),
                "filename"     : uploaded_file.name
            }
            st.success("✅ Ready! Ask your questions →")
            st.rerun()


    # ── Model Info ──
    st.markdown('<div class="sidebar-section">🤖 Models</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.78rem; color:#6b7a99; line-height:1.8;'>
        <span style='color:#00d4aa;'>●</span> Embedding: nemoretriever-300m<br>
        <span style='color:#4f7cff;'>●</span> LLM: Qwen3.5-122B-A10B<br>
        <span style='color:#f59e0b;'>●</span> Vector DB: Pinecone<br>
        <span style='color:#00d4aa;'>●</span> Calculator: Python pandas
    </div>
    """, unsafe_allow_html=True)

    # ── Example Questions ──
    st.markdown('<div class="sidebar-section">💡 Example Questions</div>',
                unsafe_allow_html=True)



    # ── Clear Chat ──
    if st.session_state.chat_history:
        st.markdown('<div class="sidebar-section">⚙️ Actions</div>',
                    unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()





st.markdown("""
<div class="main-header">
    <h1>💳 FinanceIQ</h1>
    <p>AI-powered bank statement analyzer · NVIDIA Qwen3.5 · Real Python math</p>
</div>
""", unsafe_allow_html=True)


# PDF Stats
if st.session_state.pdf_processed and st.session_state.pdf_stats:
    stats = st.session_state.pdf_stats
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-label">📄 Text Sections</div>
            <div class="stat-value">{stats['text_sections']}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">📊 Tables Found</div>
            <div class="stat-value">{stats['tables']}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">🧩 Indexed Chunks</div>
            <div class="stat-value">{stats['chunks']}</div>
        </div>
    </div>
    <div style='text-align:center; margin-bottom:1rem;'>
        <span class="badge badge-green">✅ INDEXED</span>&nbsp;
        <span class="badge badge-blue">{stats['filename']}</span>
    </div>
    """, unsafe_allow_html=True)


#  Welcome screen
if not st.session_state.pdf_processed:
    st.markdown("""
    <div style='text-align:center; padding:4rem 2rem;'>
        <div style='font-size:4rem; margin-bottom:1rem;'>📂</div>
        <div style='font-family:Space Mono,monospace; font-size:1.1rem;
                    color:#e8edf5; margin-bottom:0.5rem;'>
            Upload your bank statement to get started
        </div>
        <div style='color:#6b7a99; font-size:0.9rem;'>
            Supports any bank · PDF format · All data stays private
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stat-card" style='text-align:center;'>
            <div style='font-size:2rem;'>🔍</div>
            <div style='color:#00d4aa; font-weight:600; margin:8px 0 4px;'>Smart Retrieval</div>
            <div style='color:#6b7a99; font-size:0.82rem;'>NVIDIA embeddings find relevant transactions instantly</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card" style='text-align:center;'>
            <div style='font-size:2rem;'>🔢</div>
            <div style='color:#00d4aa; font-weight:600; margin:8px 0 4px;'>Real Calculator</div>
            <div style='color:#6b7a99; font-size:0.82rem;'>Python pandas math — never guesses totals</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-card" style='text-align:center;'>
            <div style='font-size:2rem;'>🤖</div>
            <div style='color:#00d4aa; font-weight:600; margin:8px 0 4px;'>Qwen3.5 LLM</div>
            <div style='color:#6b7a99; font-size:0.82rem;'>256K context window for full statement analysis</div>
        </div>""", unsafe_allow_html=True)


# Chat Interface
if st.session_state.pdf_processed:

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user">{msg['content']}</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-bot">{msg['content']}</div>
            """, unsafe_allow_html=True)

            # Show calculator results if any
            if msg.get("calc"):
                st.markdown(f"""
                <div class="calc-result">{msg['calc']}</div>
                """, unsafe_allow_html=True)

    #Question Input
    st.markdown("<br>", unsafe_allow_html=True)

    # Check if an example question was clicked from sidebar
    prefill = st.session_state.pop("prefill_question", "")

    question = st.chat_input(
        placeholder = "Ask anything about your bank statement...",
    )

    # Handle sidebar example button click
    if prefill and not question:
        question = prefill

    if question:
        # Add user message to history
        st.session_state.chat_history.append({
            "role"   : "user",
            "content": question
        })

        # Show user message immediately
        st.markdown(f"""
        <div class="msg-user">{question}</div>
        """, unsafe_allow_html=True)

        # Get answer
        with st.spinner("🤔 Thinking..."):
            try:
                answer, calculated = answer_question(
                    question,
                    st.session_state.retriever
                )

                # Add bot response to history
                st.session_state.chat_history.append({
                    "role"   : "bot",
                    "content": answer,
                    "calc"   : calculated
                })

                # Show answer
                st.markdown(f"""
                <div class="msg-bot">{answer}</div>
                """, unsafe_allow_html=True)

                # Show calculator results if calculation was done
                if calculated:
                    st.markdown(f"""
                    <div class="calc-result">{calculated}</div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

        st.rerun()
