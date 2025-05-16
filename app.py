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

# --- STREAMLIT SECRETS & CONFIG ---
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


def index_drive_docs():
    """Indexes all Docs, Slides, and PDFs in the shared folder."""
    mime_filter = (
        "mimeType='application/pdf' or "
        "mimeType='application/vnd.google-apps.document' or "
        "mimeType='application/vnd.google-apps.presentation'"
    )
    token, total = None, 0
    while True:
        resp = drive_service.files().list(
            q=f"'{SHARED_FOLDER}' in parents and ({mime_filter})",
            fields="nextPageToken, files(id,name,mimeType)",
            pageToken=token,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True
        ).execute()
        for f in resp.get("files", []):
            text = extract_text_from_drive_file(f["id"], f["mimeType"])
            if text:
                emb = get_embedding(text)
                index.upsert(vectors=[(f["id"], emb, {"name": f["name"], "source": "drive"})])
                total += 1
        token = resp.get("nextPageToken")
        if not token:
            break
    return total


def fetch_and_index_web(query: str, top_k: int = 3) -> int:
    """Fetches top web results via SerpAPI and indexes them."""
    r = requests.get(
        "https://serpapi.com/search.json",
        params={"q": query, "api_key": SERPAPI_KEY}
    )
    total = 0
    for res in r.json().get("organic_results", [])[:top_k]:
        url = res.get("link")
        if not url:
            continue
        html = requests.get(url, timeout=10).text
        text = " ".join(p.get_text() for p in BeautifulSoup(html, "html.parser").find_all("p"))
        if text:
            emb = get_embedding(text)
            index.upsert(vectors=[(url, emb, {"name": res.get("title", url), "source": url})])
            total += 1
    return total


def get_relevant_docs(query: str, top_k: int = 5) -> list[str]:
    emb = get_embedding(query)
    res = index.query(vector=emb, top_k=top_k, include_metadata=True)
    return [f"{m['metadata']['source']}: {m['metadata'].get('name','')}" for m in res["matches"]]


def chat_with_context(query: str) -> str:
    docs = get_relevant_docs(query)
    prompt = query + "\n\nContext:" + "\n".join(docs)
    resp = client.chat.completions.create(model="gpt-4", messages=[
        {"role": "user", "content": prompt}
    ], temperature=0.7)
    return resp.choices[0].message.content

# --- UI ---
st.title("🔮 Zero1 RAG Assistant")
query = st.text_input("Enter your strategic query:")
include_web = st.checkbox("Include web context in this query")
if st.button("Analyze") and query:
    with st.spinner("Indexing Drive documents..."):
        drive_count = index_drive_docs()
    if drive_count:
        st.success(f"Indexed {drive_count} Drive files.")
    else:
        st.info("No Drive files found or indexed.")
    if include_web:
        with st.spinner("Fetching and indexing web context..."):
            web_count = fetch_and_index_web(query)
        if web_count:
            st.success(f"Indexed {web_count} web pages.")
        else:
            st.info("No web pages indexed.")
    with st.spinner("Generating response..."):
        answer = chat_with_context(query)
    st.markdown("**Response:**")
    st.write(answer)
