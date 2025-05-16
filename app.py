import streamlit as st
import os
import tempfile
import requests
import openai
from serpapi import GoogleSearch
from google.oauth2 import service_account
from googleapiclient.discovery import build
import chromadb
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

# --- Configuration via Streamlit secrets ---
# In your Streamlit sharing settings, add secrets:
# [openai]
# api_key = "YOUR_OPENAI_KEY"
# [serpapi]
# api_key = "YOUR_SERPAPI_KEY"
# [gcp]
# service_account = '''{...YOUR JSON...}'''

# Load API keys and credentials
openai.api_key = st.secrets.openai.api_key
SERPAPI_KEY = st.secrets.serpapi.api_key

gcp_info = st.secrets.gcp.service_account
credentials = service_account.Credentials.from_service_account_info(
    gcp_info,
    scopes=["https://www.googleapis.com/auth/drive.readonly"]
)

drive_service = build('drive', 'v3', credentials=credentials)

# Initialize vector store (ChromaDB)
chroma_client = chromadb.Client()
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai.api_key,
    model_name="text-embedding-ada-002"
)
# Create or get the collection
try:
    collection = chroma_client.get_collection(name="zero1")
except Exception:
    collection = chroma_client.create_collection(
        name="zero1",
        embedding_function=embedding_fn
    )

# --- Utility functions ---

def extract_text_from_drive_file(file_id, mime_type):
    """
    Export Google Docs/Slides to text or download PDFs.
    """
    if mime_type == 'application/pdf':
        # Download PDF and extract text via PdfReader
        downloader = requests.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
            headers={'Authorization': f'Bearer {credentials.token}'},
            stream=True
        )
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            fh.write(downloader.content)
            tmp_path = fh.name
        text = []
        reader = PdfReader(tmp_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return '\n'.join(text)
    else:
        # Docs or Slides: export to plain text
        request = drive_service.files().export(fileId=file_id,
                                               mimeType='text/plain')
        return request.execute().decode('utf-8')


def index_drive_docs():
    """List, extract and index all Docs, Slides, and PDFs in Drive."""
    with st.spinner("Indexing Google Drive documents..."):
        page_token = None
        while True:
            resp = drive_service.files().list(
                q="mimeType contains 'application/vnd.google-apps.document' or mimeType contains 'presentation' or mimeType='application/pdf'",
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token
            ).execute()
            for f in resp.get('files', []):
                text = extract_text_from_drive_file(f['id'], f['mimeType'])
                if not text:
                    continue
                collection.upsert([
                    {
                        'id': f['id'],
                        'embedding': embedding_fn(text),
                        'metadata': {'name': f['name'], 'source': 'drive'}
                    }
                ])
            page_token = resp.get('nextPageToken')
            if not page_token:
                break
        st.success("Drive indexing complete!")


def fetch_and_index_web(query, top_k=3):
    """Search web via SerpAPI, extract pages, compute embeddings, and index."""
    with st.spinner(f"Fetching and indexing web results for '{query}'..."):
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
            collection.upsert([
                {
                    'id': url,
                    'embedding': emb,
                    'metadata': {'name': r.get('title', url), 'source': url}
                }
            ])
        st.success("Web indexing complete!")


def get_relevant_docs(query, top_k=5):
    """Retrieve top-k similar docs from vector store."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results['documents'][0]


def chat_with_context(query):
    docs = get_relevant_docs(query)
    context = "\n\n---\n\n".join(docs)
    messages = [
        {"role": "system", "content": "You are a Zero1 strategy assistant. Provide actionable, data-driven recommendations."},
        {"role": "user", "content": f"{query}\n\nContext:\n{context}"}
    ]
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7
    )
    return resp.choices[0].message.content

# --- Streamlit App UI ---
st.set_page_config(page_title="Zero1 Strategy RAG Assistant", layout="wide")
st.title("🔮 Zero1 Strategy RAG Assistant")

with st.sidebar:
    st.header("🚀 Indexing")
    if st.button("Index Google Drive Docs"):
        index_drive_docs()
    st.write("---")
    web_q = st.text_input("Fetch & index web for query:")
    if st.button("Fetch Web Context") and web_q:
        fetch_and_index_web(web_q)

st.write("---")
query = st.text_input("Ask your strategic question about Zero1:")
if st.button("Analyze & Respond") and query:
    with st.spinner("Analyzing..."):
        answer = chat_with_context(query)
    st.markdown("**Response:**")
    st.write(answer)

# Cost & Growth Analysis Stub
st.write("---")
st.subheader("📊 Cost & Growth Analysis")
st.info("Parse numeric outputs from the assistant and use pandas to compute target capex & growth models.")
