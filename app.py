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
    """Real-time indexing logs and upserts for Drive files."""
    mime_filter = (
        "mimeType='application/pdf' or "
        "mimeType='application/vnd.google-apps.document' or "
        "mimeType='application/vnd.google-apps.presentation'"
    )
    token, total = None, 0
    st.write("🚀 **Starting Drive folder indexing...**")
    while True:
        resp = drive_service.files().list(
            q=f"'{SHARED_FOLDER}' in parents and ({mime_filter})",
            fields="nextPageToken, files(id,name,mimeType)",
            pageToken=token,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True
        ).execute()
        files = resp.get("files", [])
        st.write(f"🔍 Found **{len(files)}** files in folder {SHARED_FOLDER}:")
        for f in files:
            name, mime = f['name'], f['mimeType']
            st.write(f" • **{name}** (`{mime}`)")
            txt = extract_text_from_drive_file(f['id'], mime)
            if not txt:
                st.write(f"   ⚠️ No text extracted for {name}")
                continue
            emb = get_embedding(txt)
            index.upsert(vectors=[(f['id'], emb, {"name": name, "source": "drive"})])
            st.write(f"   ✅ Upserted vector for {name}")
            total += 1
        token = resp.get("nextPageToken")
        if not token:
            break
    st.write(f"✅ **Drive indexing complete!** Total upserts: **{total}**")
    stats = index.describe_index_stats()
    st.write(f"📦 Pinecone now has **{stats.get('total_vector_count',0)}** vectors.")
    return total


def fetch_and_index_web(query: str, top_k: int = 3) -> int:
    """Fetches top web results via SerpAPI and indexes them with logs."""
    st.write(f"🌐 **Fetching & indexing top {top_k} web results for** `{query}`")
    r = requests.get(
        "https://serpapi.com/search.json",
        params={"q": query, "api_key": SERPAPI_KEY}
    )
    total = 0
    for res in r.json().get("organic_results", [])[:top_k]:
        url = res.get("link")
        if not url:
            continue
        st.write(f" • Fetching **{res.get('title',url)}**")
        html = requests.get(url, timeout=10).text
        text = " ".join(p.get_text() for p in BeautifulSoup(html, "html.parser").find_all("p"))
        if not text:
            st.write(f"   ⚠️ No text from {url}")
            continue
        emb = get_embedding(text)
        index.upsert(vectors=[(url, emb, {"name": res.get('title',url), "source": url})])
        st.write(f"   ✅ Upserted web page: {res.get('title',url)}")
        total += 1
    st.write(f"✅ **Web indexing complete!** Total upserts: **{total}**")
    return total


def get_relevant_docs(query: str, top_k: int = 5) -> list[str]:
    emb = get_embedding(query)
    res = index.query(vector=emb, top_k=top_k, include_metadata=True)
    return [f"{m['metadata']['source']}: {m['metadata'].get('name','')}" for m in res["matches"]]


def chat_with_context(query: str, include_web: bool, web_prompt: str) -> str:
    # Re-index Drive docs first for up-to-date context
    index_drive_docs()
    if include_web and web_prompt:
        fetch_and_index_web(web_prompt)
    docs = get_relevant_docs(query)
    context_sections = [f"[Drive] {d}" for d in docs]
    if include_web and web_prompt:
        context_sections += [f"[Web] {web_prompt}"]
    context = "\n\n---\n\n".join(context_sections)
    prompt = query + "\n\nContext:\n" + context
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return resp.choices[0].message.content

# --- UI ---
st.title("🔮 Zero1 RAG Assistant")
with st.sidebar:
    st.header("Actions")
    if st.button("Index Drive Folder"):
        index_drive_docs()
    st.write("---")
    include_web = st.checkbox("Include web context in query")
    if include_web:
        web_prompt = st.text_input("Enter web search prompt:")
    else:
        web_prompt = ""

st.write("---")
query = st.text_input("Enter your strategic query:")
if st.button("Analyze") and query:
    with st.spinner("Processing..."):
        answer = chat_with_context(query, include_web, web_prompt)
    st.markdown("**Response:**")
    st.write(answer)
