import os
import tempfile
import requests
import streamlit as st
import openai
from serpapi import GoogleSearch
from google.oauth2 import service_account
from googleapiclient.discovery import build
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

# --- Configuration via Streamlit secrets ---
# Add these to Settings → Secrets:
# [openai]
# api_key = "YOUR_OPENAI_KEY"
# [serpapi]
# api_key = "YOUR_SERPAPI_KEY"
# [gcp]
# service_account = '''{...YOUR JSON...}'''
# And set Environment Variable:
# CHROMA_DB_IMPL = "duckdb+parquet"

# Load API keys and credentials
openai.api_key = st.secrets.openai.api_key
SERPAPI_KEY = st.secrets.serpapi.api_key

gcp_info = st.secrets.gcp.service_account
credentials = service_account.Credentials.from_service_account_info(
    gcp_info,
    scopes=["https://www.googleapis.com/auth/drive.readonly"]
)

drive_service = build('drive', 'v3', credentials=credentials)

# Initialize vector store (uses env var or default duckdb+parquet)
db_impl = os.getenv('CHROMA_DB_IMPL', 'duckdb+parquet')
chroma_client = chromadb.Client(Settings(
    chroma_db_impl=db_impl,
    persist_directory="./chroma_storage"
))
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai.api_key,
    model_name="text-embedding-ada-002"
)
try:
    collection = chroma_client.get_collection(name="zero1")
except Exception:
    collection = chroma_client.create_collection(
        name="zero1",
        embedding_function=embedding_fn
    )

# --- Utility functions ---

def extract_text_from_drive_file(file_id, mime_type):
    if mime_type == 'application/pdf':
        resp = requests.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
            headers={'Authorization': f'Bearer {credentials.token}'},
            stream=True
        )
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            fh.write(resp.content)
            tmp_path = fh.name
        text = []
        reader = PdfReader(tmp_path)
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                text.append(txt)
        return '\n'.join(text)
    else:
        return drive_service.files().export(
            fileId=file_id,
            mimeType='text/plain'
        ).execute().decode('utf-8')


def index_drive_docs():
    with st.spinner("Indexing Google Drive documents..."):
        token = None
        while True:
            res = drive_service.files().list(
                q="mimeType contains 'application/vnd.google-apps.document' or mimeType contains 'presentation' or mimeType='application/pdf'",
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=token
            ).execute()
            for f in res.get('files', []):
                txt = extract_text_from_drive_file(f['id'], f['mimeType'])
                if not txt:
                    continue
                emb = embedding_fn(txt)
                collection.upsert([{
                    'id': f['id'],
                    'embedding': emb,
                    'metadata': {'name': f['name'], 'source': 'drive'}
                }])
            token = res.get('nextPageToken')
            if not token:
                break
        st.success("Drive indexing complete!")


def fetch_and_index_web(query, top_k=3):
    with st.spinner(f"Fetching web results for '{query}'..."):
        client = GoogleSearch({"q": query, "api_key": SERPAPI_KEY})
        results = client.get_dict().get('organic_results', [])[:top_k]
        for r in results:
            url = r.get('link')
            if not url:
                continue
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            text = ' '.join(p.get_text() for p in soup.find_all('p'))
            if not text:
                continue
            emb = embedding_fn(text)
            collection.upsert([{
                'id': url,
                'embedding': emb,
                'metadata': {'name': r.get('title', url), 'source': url}
            }])
        st.success("Web indexing complete!")


def get_relevant_docs(query, top_k=5):
    res = collection.query(query_texts=[query], n_results=top_k)
    return res['documents'][0]


def chat_with_context(query):
    docs = get_relevant_docs(query)
    ctx = "\n\n---\n\n".join(docs)
    msgs = [
        {"role": "system", "content": "You are a Zero1 strategist. Provide actionable recommendations."},
        {"role": "user", "content": f"{query}\n\nContext:\n{ctx}"}
    ]
    resp = openai.ChatCompletion.create(model="gpt-4", messages=msgs, temperature=0.7)
    return resp.choices[0].message.content

# --- Streamlit UI ---
st.set_page_config(page_title="Zero1 RAG Assistant", layout="wide")
st.title("🔮 Zero1 RAG Assistant")

with st.sidebar:
    st.header("Indexing")
    if st.button("Index Drive Docs"):
        index_drive_docs()
    st.write("---")
    q = st.text_input("Fetch & index web:")
    if st.button("Fetch Web") and q:
        fetch_and_index_web(q)

st.write("---")
usr = st.text_input("Ask a question about Zero1:")
if st.button("Analyze") and usr:
    with st.spinner("Analyzing..."):
        ans = chat_with_context(usr)
    st.markdown("**Response:**")
    st.write(ans)

st.write("---")
st.subheader("📊 Cost & Growth Analysis")
st.info("Use pandas on numeric outputs to model X crore capex and growth.")
