# app.py

import os
import json
import tempfile
import requests
import streamlit as st
from serpapi import GoogleSearch
from google.oauth2 import service_account
from googleapiclient.discovery import build
from pinecone import Pinecone, ServerlessSpec
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from openai import OpenAI

# --- STREAMLIT SECRETS & CONFIG ---
# Secrets (see above) are stored in st.secrets
OPENAI_KEY    = st.secrets["openai"]["api_key"]
SERPAPI_KEY   = st.secrets["serpapi"]["api_key"]
PINECONE_KEY  = st.secrets["pinecone"]["api_key"]
PINECONE_ENV  = st.secrets["pinecone"]["environment"]
GCP_JSON      = st.secrets["gcp"]["service_account"]
SHARED_FOLDER = st.secrets["drive"]["folder_id"]

# Initialize OpenAI v1 client
client = OpenAI(api_key=OPENAI_KEY)

# Initialize Google Drive service
gcp_info = json.loads(GCP_JSON)
creds    = service_account.Credentials.from_service_account_info(
    gcp_info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
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

# --- HELPERS ---

def get_embedding(text: str) -> list:
    resp = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return resp["data"][0]["embedding"]

def extract_text_from_drive_file(fid: str, mime: str) -> str:
    if mime == "application/pdf":
        r = requests.get(
            f"https://www.googleapis.com/drive/v3/files/{fid}?alt=media",
            headers={"Authorization": f"Bearer {creds.token}"},
            stream=True
        )
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            fh.write(r.content)
            path = fh.name
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    else:
        data = drive_service.files().export(
            fileId=fid, mimeType="text/plain"
        ).execute()
        return data.decode("utf-8")

def index_drive_docs():
    """Index only the files in the shared folder."""
    with st.spinner("Indexing Drive folder…"):
        token = None
        while True:
            resp = drive_service.files().list(
                q=(
                    f"'{SHARED_FOLDER}' in parents and "
                    "(mimeType='application/pdf' or "
                    "mimeType contains 'document' or "
                    "mimeType contains 'presentation')"
                ),
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=token
            ).execute()
            for f in resp.get("files", []):
                txt = extract_text_from_drive_file(f["id"], f["mimeType"])
                if not txt:
                    continue
                emb  = get_embedding(txt)
                meta = {"name": f["name"], "source": "drive"}
                index.upsert(vectors=[(f["id"], emb, meta)])
            token = resp.get("nextPageToken")
            if not token:
                break
        st.success("Drive folder indexed!")

def fetch_and_index_web(query: str, top_k: int = 3):
    with st.spinner(f"Fetching web for '{query}'…"):
        results = (
            GoogleSearch({"q": query, "api_key": SERPAPI_KEY})
            .get_dict()
            .get("organic_results", [])[:top_k]
        )
        for r in results:
            url = r.get("link")
            if not url:
                continue
            html = requests.get(url, timeout=10).text
            text = " ".join(
                p.get_text() for p in BeautifulSoup(html, "html.parser").find_all("p")
            )
            if not text:
                continue
            emb  = get_embedding(text)
            meta = {"name": r.get("title", url), "source": url}
            index.upsert(vectors=[(url, emb, meta)])
        st.success("Web pages indexed!")

def get_relevant_docs(query: str, top_k: int = 5) -> list[str]:
    emb = get_embedding(query)
    res = index.query(vector=emb, top_k=top_k, include_metadata=True)
    return [
        f"{m['metadata']['source']}: {m['metadata'].get('name','')}"
        for m in res["matches"]
    ]

def chat_with_context(query: str) -> str:
    docs = get_relevant_docs(query)
    ctx  = "\n\n---\n\n".join(docs)
    messages = [
        {"role": "system", "content": "You are a Zero1 strategy assistant."},
        {"role": "user",   "content": f"{query}\n\nContext:\n{ctx}"}
    ]
    resp = client.chat.completions.create(
        model="gpt-4", messages=messages, temperature=0.7
    )
    return resp.choices[0].message.content

# --- STREAMLIT UI ---

st.set_page_config(page_title="Zero1 RAG Assistant", layout="wide")
st.title("🔮 Zero1 RAG Assistant")

with st.sidebar:
    st.header("Indexing")
    if st.button("Index Drive Folder"):
        index_drive_docs()
    st.write("---")
    web_q = st.text_input("Fetch & index web:")
    if st.button("Fetch Web") and web_q:
        fetch_and_index_web(web_q)

st.write("---")
user_q = st.text_input("Ask your strategic question:")
if st.button("Analyze") and user_q:
    with st.spinner("Analyzing…"):
        ans = chat_with_context(user_q)
    st.markdown("**Response:**")
    st.write(ans)

st.write("---")
st.subheader("📊 Cost & Growth Analysis")
st.info("Parse the assistant’s numeric outputs with pandas to model X-crore capex & growth.")
