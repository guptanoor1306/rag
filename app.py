import os
import json
import tempfile
import requests
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from pinecone import Pinecone, ServerlessSpec
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from openai import OpenAI

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title="Zero1 RAG Assistant", layout="wide")

# --- SECRETS & CONFIG ---
OPENAI_KEY    = st.secrets["openai"]["api_key"]
SERPAPI_KEY   = st.secrets["serpapi"]["api_key"]
PINECONE_KEY  = st.secrets["pinecone"]["api_key"]
PINECONE_ENV  = st.secrets["pinecone"]["environment"]
GCP_JSON      = st.secrets["gcp"]["service_account"]
SHARED_FOLDER = st.secrets["drive"]["folder_id"]

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_KEY)

# Setup Google Drive API
gcp_info    = json.loads(GCP_JSON)
creds       = service_account.Credentials.from_service_account_info(
    gcp_info,
    scopes=["https://www.googleapis.com/auth/drive.readonly"]
)
drive_service = build("drive", "v3", credentials=creds)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_KEY, environment=PINECONE_ENV)
region, cloud = PINECONE_ENV.split("-", 1)
INDEX_NAME = "zero1"
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region)
    )
index = pc.Index(INDEX_NAME)

# Session state for chunks
def initialize_state():
    if 'indexed_chunks' not in st.session_state:
        st.session_state['indexed_chunks'] = set()
    if 'chunk_texts' not in st.session_state:
        st.session_state['chunk_texts'] = {}
initialize_state()

# --- Helper Functions ---

def get_embedding(text: str) -> list:
    resp = client.embeddings.create(model="text-embedding-ada-002", input=text)
    return resp.data[0].embedding


def extract_text_from_drive_file(file_id: str, mime: str) -> str:
    if mime == "application/pdf":
        r = requests.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
            headers={"Authorization": f"Bearer {creds.token}"},
            stream=True
        )
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            fh.write(r.content)
            path = fh.name
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    else:
        return drive_service.files().export(fileId=file_id, mimeType="text/plain").execute().decode("utf-8")


def index_drive_docs(chunk_size: int = 3000):
    mime_filter = (
        "mimeType='application/pdf' or "
        "mimeType='application/vnd.google-apps.document' or "
        "mimeType='application/vnd.google-apps.presentation'"
    )
    total_chunks = 0
    st.write("🚀 Indexing Drive folder with chunking...")
    token = None
    while True:
        resp = drive_service.files().list(
            q=f"'{SHARED_FOLDER}' in parents and ({mime_filter})",
            fields="nextPageToken, files(id,name,mimeType)",
            pageToken=token,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True
        ).execute()
        files = resp.get("files", [])
        st.write(f"🔍 Found {len(files)} files to index.")
        for f in files:
            file_id, name, mime = f['id'], f['name'], f['mimeType']
            text = extract_text_from_drive_file(file_id, mime)
            if not text:
                continue
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                chunk_id = f"{file_id}_chunk_{i//chunk_size}"
                if chunk_id in st.session_state['indexed_chunks']:
                    continue
                emb = get_embedding(chunk)
                index.upsert(vectors=[(chunk_id, emb, {})])
                st.session_state['chunk_texts'][chunk_id] = chunk
                st.session_state['indexed_chunks'].add(chunk_id)
                total_chunks += 1
        token = resp.get("nextPageToken")
        if not token:
            break
    st.write(f"✅ Drive indexing complete! {total_chunks} chunks stored.")


def fetch_and_index_web(query: str, top_k: int = 3) -> None:
    st.write(f"🌐 Fetching top {top_k} web results for '{query}'...")
    r = requests.get("https://serpapi.com/search.json", params={"q": query, "api_key": SERPAPI_KEY})
    for res in r.json().get("organic_results", [])[:top_k]:
        url, title = res.get("link"), res.get("title")
        if not url:
            continue
        html = requests.get(url, timeout=10).text
        text = " ".join(p.get_text() for p in BeautifulSoup(html, "html.parser").find_all("p"))
        if not text:
            continue
        emb = get_embedding(text)
        index.upsert(vectors=[(url, emb, {})])


def get_relevant_docs(query: str, top_k: int = 5) -> list[str]:
    emb = get_embedding(query)
    res = index.query(vector=emb, top_k=top_k, include_metadata=True)
    return [st.session_state['chunk_texts'][m['id']] for m in res['matches']]


def chat_with_context(query: str, include_web: bool, web_prompt: str) -> str:
    if include_web and web_prompt:
        fetch_and_index_web(web_prompt)
    docs = get_relevant_docs(query)
    if not docs:
        return "I’m sorry, I don’t have information on that topic."
    context = "\n\n---\n\n".join(docs)
    messages = [
        {"role": "system", "content": "You are a Zero1 strategist. Use only the provided context to answer."},
        {"role": "user", "content": f"{query}\n\nContext:\n{context}"}
    ]
    resp = client.chat.completions.create(model="gpt-4", messages=messages, temperature=0.0)
    return resp.choices[0].message.content

# --- UI ---
st.title("🔮 Zero1 RAG Assistant")
with st.sidebar:
    if st.button("Index Drive Folder"): index_drive_docs()
    st.write("---")
    include_web = st.checkbox("Include web context")
    web_prompt = st.text_input("Web prompt:") if include_web else ""

query = st.text_input("Your query:")
if st.button("Analyze") and query:
    with st.spinner("Thinking..."):
        answer = chat_with_context(query, include_web, web_prompt)
    st.markdown("**Response:**")
    st.write(answer)
