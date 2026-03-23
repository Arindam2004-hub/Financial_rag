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


#  Page Config 

st.set_page_config(
    page_title="FinanceIQ — Bank Statement Analyzer",
    page_icon="💳",
)


#  Load Environment 

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


# Calculation Keywords 

CALCULATION_KEYWORDS = [
    "total", "sum", "how much", "calculate", "average", "avg",
    "count", "how many", "maximum", "minimum", "largest", "smallest",
    "highest", "lowest", "balance", "closing", "opening", "spent",
    "received", "earned", "paid", "withdrawn", "deposited",
    "revenue", "profit", "loss", "income", "expense", "turnover"
]


#  Cached Resources 

@st.cache_resource
def get_nvidia_embeddings():
    return NVIDIAEmbeddings(
        model   = "nvidia/llama-3.2-nemoretriever-300m-embed-v1",
        api_key = NVIDIA_EMBEDDING_API_KEY,
        truncate= "NONE",
    )


@st.cache_resource
def get_qwen_llm():
    return ChatNVIDIA(
        model                 = "qwen/qwen3.5-122b-a10b",
        api_key               = NVIDIA_LLM_API_KEY,
        temperature           = 0.0,
        max_completion_tokens = 2048,
    )


# Helper: Clean DataFrame 

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
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace('\x00', '').replace('\r', ' ')
    return text[:10000]




def extract_pdf_content(uploaded_file) -> dict:
    """
    Extracts content three ways per page:
      1. Full raw page text  (always captured — nothing is lost)
      2. Structured tables   (when pdfplumber can parse them)
    """
    text_docs  = []
    dataframes = []

    with pdfplumber.open(uploaded_file) as pdf:
        for page_num, page in enumerate(pdf.pages):

          
            page_text = page.extract_text(layout=True)

            if not page_text or not page_text.strip():
                # Fallback without layout mode
                page_text = page.extract_text()

            if page_text and page_text.strip():
                text_docs.append(Document(
                    page_content = page_text.strip(),
                    metadata     = {
                        "page"      : page_num + 1,
                        "type"      : "text",
                        "chunk_type": "full_page_text"
                    }
                ))

           
            table_settings = {
                "vertical_strategy"  : "text",   
                "horizontal_strategy": "text",
                "snap_tolerance"     : 5,
                "join_tolerance"     : 5,
                "edge_min_length"    : 10,
            }

            try:
                tables = page.extract_tables(table_settings)
            except Exception:
                tables = page.extract_tables()   # fallback to default settings

            for table in tables:
                if not table or len(table) < 2:
                    continue

                
                header_row_idx = 0
                for idx, row in enumerate(table):
                    if any(cell and str(cell).strip() for cell in row):
                        header_row_idx = idx
                        break

                header = table[header_row_idx]
                data   = table[header_row_idx + 1:]

                if not data:
                    continue

                df = pd.DataFrame(data, columns=header)
                df = df.dropna(how='all').fillna('')
                dataframes.append({"df": df, "page": page_num + 1})

    return {"text_docs": text_docs, "dataframes": dataframes}




def chunk_by_transactions(extracted: dict, rows_per_chunk: int = 30) -> list:
    """
    Chunks text pages into 50-line windows (was 20) so section headers
    stay with their data. Tables are chunked in groups of 30 rows (was 15).
    """
    all_chunks = []

    #  Text chunks: 50 lines per chunk 
    for doc in extracted["text_docs"]:
        lines = [l for l in doc.page_content.split("\n") if l.strip()]
        chunk_size = 50   # was 20 — bigger window keeps headings with numbers

        for i in range(0, len(lines), chunk_size):
            chunk_text = "\n".join(lines[i: i + chunk_size])
            if chunk_text.strip():
                all_chunks.append(Document(
                    page_content = chunk_text,
                    metadata     = {
                        **doc.metadata,
                        "chunk_type": "text",
                        "line_start" : i,
                        "line_end"   : i + chunk_size,
                    }
                ))

    #  Table chunks 
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


#  Pinecone Indexing 

def index_to_pinecone(chunks: list) -> PineconeVectorStore:
    embeddings  = get_nvidia_embeddings()
    vectorstore = PineconeVectorStore.from_documents(
        documents  = chunks,
        embedding  = embeddings,
        index_name = PINECONE_INDEX_NAME
    )
    return vectorstore




def build_retriever(vectorstore: PineconeVectorStore):
    return vectorstore.as_retriever(
        search_type   = "similarity",
        search_kwargs = {"k": 15}   # was 8
    )


#  Calculator

def run_calculator(retrieved_docs: list) -> str:
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
        if col is None:
            continue
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
    retrieved_docs = retriever.invoke(question)

    needs_calc = any(kw in question.lower() for kw in CALCULATION_KEYWORDS)
    calculated = run_calculator(retrieved_docs) if needs_calc else ""

    formatted_context = "\n\n".join([
        f"--- Chunk {i+1} | Page {doc.metadata.get('page','?')} | Type: {doc.metadata.get('chunk_type','?')} ---\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs)
    ])

    prompt = ChatPromptTemplate.from_template("""
You are an expert financial analyst AI assistant.

RULES:
1. Read ALL chunks carefully — numbers may appear in the raw page text chunks, not just table chunks
2. If calculated results are provided, use those exact numbers
3. If numbers are found anywhere in the context, use them — do NOT say "not found" if the data is there
4. Mention exact figures, dates, and descriptions
5. Format amounts clearly (e.g. ₹1,25,000 or $1,250.00)
6. Only say "Not found in the statement" if the data is truly absent from ALL chunks below

---RETRIEVED CHUNKS (includes full page text + table rows)---
{context}

---CALCULATED RESULTS (use these exact numbers if available)---
{calculated}

---QUESTION---
{question}

---ANSWER---
""")

    llm   = get_qwen_llm()
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context"   : formatted_context,
        "calculated": calculated if calculated else "No calculation performed.",
        "question"  : question
    })

    return answer, calculated


#  Session State

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "pdf_stats" not in st.session_state:
    st.session_state.pdf_stats = {}


# Sidebar 

with st.sidebar:
    st.title("💳 FinanceIQ")
    st.caption("Bank Statement Analyzer")
    st.divider()

    st.subheader("📄 Upload Statement")

    uploaded_file = st.file_uploader(
        label="Upload your bank statement PDF",
        type=["pdf"],
        help="Supports any bank statement or annual report in PDF format"
    )

    if uploaded_file:
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")

        if st.button("⚡ Process & Index PDF", use_container_width=True):
            with st.spinner("Extracting text and tables..."):
                extracted = extract_pdf_content(uploaded_file)

            with st.spinner("Creating chunks..."):
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
            st.success("✅ Ready! Ask your questions.")
            st.rerun()

    st.divider()
    st.subheader("🤖 Models Used")
    st.write("**Embedding:** nemoretriever-300m")
    st.write("**LLM:** Qwen3.5-122B")
    st.write("**Vector DB:** Pinecone")
    st.write("**Calculator:** Python pandas")

    st.divider()
    st.subheader("💡 Example Questions")
    st.write("- What is my total spending?")
    st.write("- What is the closing balance?")
    st.write("- How many transactions were made?")
    st.write("- What is the largest debit?")
    st.write("- What is the revenue / profit for the year?")

    if st.session_state.chat_history:
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# Main Page

st.title("💳 FinanceIQ — Bank Statement Analyzer")
st.caption("Powered by NVIDIA Qwen3.5 · Real Python math · Pinecone vector search")
st.divider()

if st.session_state.pdf_processed and st.session_state.pdf_stats:
    stats = st.session_state.pdf_stats
    st.success(f"✅ File indexed: **{stats['filename']}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Text Pages",     stats["text_sections"])
    col2.metric("Tables Found",   stats["tables"])
    col3.metric("Indexed Chunks", stats["chunks"])
    st.divider()

if not st.session_state.pdf_processed:
    st.info("👈 Upload a bank statement PDF from the sidebar to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("🔍 **Smart Retrieval**")
        st.write("NVIDIA embeddings find relevant transactions instantly.")
    with col2:
        st.write("🔢 **Real Calculator**")
        st.write("Python pandas math — never guesses totals.")
    with col3:
        st.write("🤖 **Qwen3.5 LLM**")
        st.write("256K context window for full statement analysis.")

# Chat Interface 

if st.session_state.pdf_processed:
    st.subheader("💬 Ask about your statement")

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
                if msg.get("calc"):
                    st.write("**🔢 Calculated Results:**")
                    st.code(msg["calc"], language="text")

    question = st.chat_input("Ask anything about your bank statement...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        st.session_state.chat_history.append({
            "role"   : "user",
            "content": question
        })

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, calculated = answer_question(
                        question,
                        st.session_state.retriever
                    )

                    st.write(answer)

                    if calculated:
                        st.write("**🔢 Calculated Results:**")
                        st.code(calculated, language="text")

                    st.session_state.chat_history.append({
                        "role"   : "bot",
                        "content": answer,
                        "calc"   : calculated
                    })

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        st.rerun()
