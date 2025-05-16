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

# Debug GCP project and service account
try:
    gcp_info = json.loads(GCP_JSON)
    st.write("⚙️ Using GCP project:", gcp_info.get("project_id"))
    st.write("⚙️ Service account:", gcp_info.get("client_email"))
except Exception as e:
    st.error(f"Error parsing GCP credentials: {e}")

# Setup Google Drive API
credentials = service_account.Credentials.from_service_account_info(
    gcp_info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
)
drive_service = build("drive", "v3", credentials=credentials)

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
            headers={"Authorization": f"Bearer {credentials.token}"},
            stream=True
        )
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            fh.write(r.content)
            path = fh.name
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    else:
        data = drive_service.files().export(fileId=file_id, mimeType="text/plain").execute()
        return data.decode("utf-8")


def index_drive_docs():
    st.write("🚀 **Starting Drive folder indexing...**")
    total = 0
    token = None
    while True:
        resp = drive_service.files().list(
            q=f"'{SHARED_FOLDER}' in parents",
            fields="nextPageToken, files(id,name,mimeType)",
            pageToken=token,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True
        ).execute()

        files = resp.get("files", [])
        st.write(f"🔍 Found **{len(files)}** files in folder {SHARED_FOLDER}:")
        for f in files:
            st.write(f" • **{f['name']}** (`{f['mimeType']}`)")
            txt = extract_text_from_drive_file(f["id"], f["mimeType"])
            if not txt:
                st.write(f"   ⚠️ No text extracted for {f['name']}")
                continue
            emb = get_embedding(txt)
            index.upsert(vectors=[(f["id"], emb, {"name": f["name"], "source": "drive"})])
            st.write(f"   ✅ Upserted vector for {f['name']}")
            total += 1

        token = resp.get("nextPageToken")
        if not token:
            break

    st.write(f"✅ **Indexing complete!** Upserted **{total}** vectors.")
    stats = index.describe_index_stats()
    st.write(f"📦 Pinecone now has **{stats.get('total_vector_count',0)}** vectors.")


def fetch_and_index_web(query: str, top_k: int = 3):
    st.write(f"🌐 **Fetching & indexing top {top_k} web results for** `{query}`")
    r = requests.get(
        "https://serpapi.com/search.json",
        params={"q": query, "api_key": SERPAPI_KEY}
    )
    data = r.json()
    for res in data.get("organic_results", [])[:top_k]:
        url = res.get("link")
        if not url:
            continue
        html = requests.get(url, timeout=10).text
        text = " ".join(
            p.get_text() for p in BeautifulSoup(html, "html.parser").find_all("p")
        )
        if not text:
            continue
        emb = get_embedding(text)
        meta = {"name": res.get("title", url), "source": url}
        index.upsert(vectors=[(url, emb, meta)])
        st.write(f"   🔹 Indexed web page: **{res.get('title',url)}**")
    st.write("✅ **Web indexing complete!**")


def get_relevant_docs(query: str, top_k: int = 5) -> list[str]:
    emb = get_embedding(query)
    res = index.query(vector=emb, top_k=top_k, include_metadata=True)
    return [
        f"{m['metadata']['source']}: {m['metadata'].get('name','')}"
        for m in res["matches"]
    ]


def chat_with_context(query: str) -> str:
    docs = get_relevant_docs(query)
    ctx = "\n\n---\n\n".join(docs)
    msgs = [
        {"role": "system", "content": "You are a Zero1 strategy assistant."},
        {"role": "user",   "content": f"{query}\n\nContext:\n{ctx}"}
    ]
    resp = client.chat.completions.create(model="gpt-4", messages=msgs, temperature=0.7)
    return resp.choices[0].message.content

# --- UI ---
with st.sidebar:
    st.header("Actions")
    if st.button("Index Drive Folder"):
        index_drive_docs()
    st.write("---")
    web_q = st.text_input("Fetch & index web:")
    if st.button("Fetch Web") and web_q:
        fetch_and_index_web(web_q)

st.write("---")
user_q = st.text_input("Ask your strategic question:")
if st.button("Analyze") and user_q:
    with st.spinner("🤖 Thinking..."):
        ans = chat_with_context(user_q)
    st.markdown("**Response:**")
    st.write(ans)

st.write("---")
st.subheader("📊 Cost & Growth Analysis")
st.info("Use pandas on the assistant’s outputs to model capex & growth.")
