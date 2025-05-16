# app.py

import os
import tempfile
import requests
import streamlit as st
import openai
from serpapi import GoogleSearch
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from pinecone import Pinecone, ServerlessSpec
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

# --- Configuration via Streamlit secrets ---
# In Settings → Secrets:
# [openai]
# api_key = "YOUR_OPENAI_API_KEY"
# [serpapi]
# api_key = "YOUR_SERPAPI_KEY"
# [gcp]
# service_account = '''{ ...SERVICE_ACCOUNT_JSON... }'''
# [pinecone]
# api_key     = "YOUR_PINECONE_API_KEY"
# environment = "YOUR_PINECONE_ENVIRONMENT"

# Load OpenAI & SerpAPI keys
openai.api_key = st.secrets["openai"]["api_key"]
SERPAPI_KEY   = st.secrets["serpapi"]["api_key"]

# Parse and load GCP service account for Drive API
gcp_info_json = st.secrets["gcp"]["service_account"]
gcp_info = json.loads(gcp_info_json)
credentials = service_account.Credentials.from_service_account_info(
    gcp_info,
    scopes=["https://www.googleapis.com/auth/drive.readonly"]
)
drive_service = build("drive", "v3", credentials=credentials)

# Initialize Pinecone client
pc = Pinecone(
    api_key=st.secrets["pinecone"]["api_key"],
    environment=st.secrets["pinecone"]["environment"]
)
env = st.secrets["pinecone"]["environment"]  # e.g. "us-east1-gcp"
region, cloud = env.split("-", 1)
index_name = "zero1"
existing = pc.list_indexes().names()
if index_name not in existing:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region)
    )
index = pc.Index(index_name)

# Helper: get embeddings
def get_embedding(text: str) -> list:
    resp = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return resp["data"][0]["embedding"]

# Extract text from Drive files
def extract_text_from_drive_file(file_id, mime_type):
    if mime_type == "application/pdf":
        resp = requests.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
            headers={"Authorization": f"Bearer {credentials.token}"},
            stream=True
        )
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            fh.write(resp.content)
            tmp_path = fh.name
        reader = PdfReader(tmp_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        data = drive_service.files().export(
            fileId=file_id,
            mimeType="text/plain"
        ).execute()
        return data.decode("utf-8")

# Index all Docs, Slides, and PDFs from Drive into Pinecone
def index_drive_docs():
    with st.spinner("Indexing Google Drive…"):
        token = None
        while True:
            SHARED_FOLDER_ID = "⟨1eU8cfYvpHtNyiy9hcAF1JREoj59Fe6or?usp=sharing⟩"

            res = drive_service.files().list(
                q=(
                    f"'{SHARED_FOLDER_ID}' in parents and "
                    "(mimeType contains 'application/vnd.google-apps.document' "
                    "or mimeType contains 'presentation' "
                    "or mimeType='application/pdf')"
                ),
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=token
            ).execute()

            for f in res.get("files", []):
                txt = extract_text_from_drive_file(f["id"], f["mimeType"])
                if not txt:
                    continue
                emb = get_embedding(txt)
                meta = {"name": f["name"], "source": "drive"}
                index.upsert(vectors=[(f["id"], emb, meta)])
            token = res.get("nextPageToken")
            if not token:
                break
        st.success("Drive indexed!")

# Fetch & index top-k web results via SerpAPI
def fetch_and_index_web(query, top_k=3):
    with st.spinner(f"Fetching web for '{query}'…"):
        results = GoogleSearch({"q": query, "api_key": SERPAPI_KEY}) \
            .get_dict().get("organic_results", [])[:top_k]
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
            emb = get_embedding(text)
            meta = {"name": r.get("title", url), "source": url}
            index.upsert(vectors=[(url, emb, meta)])
        st.success("Web indexed!")

# Retrieve nearest docs from Pinecone
def get_relevant_docs(query: str, top_k: int = 5) -> list:
    emb = get_embedding(query)
    res = index.query(vector=emb, top_k=top_k, include_metadata=True)
    return [
        f"{m['metadata']['source']}: {m['metadata'].get('name','')}"
        for m in res["matches"]
    ]

# Chat assistant
def chat_with_context(query: str) -> str:
    docs = get_relevant_docs(query)
    context = "\n\n---\n\n".join(docs)
    messages = [
        {"role": "system", "content": "You are a Zero1 strategy assistant."},
        {"role": "user", "content": f"{query}\n\nContext:\n{context}"}
    ]
    resp = openai.ChatCompletion.create(
        model="gpt-4", messages=messages, temperature=0.7
    )
    return resp.choices[0].message.content

# Streamlit UI
st.set_page_config(page_title="Zero1 RAG Assistant", layout="wide")
st.title("🔮 Zero1 RAG Assistant")

with st.sidebar:
    st.header("Indexing")
    if st.button("Index Drive Docs"):
        index_drive_docs()
    st.write("---")
    web_q = st.text_input("Fetch & index web:")
    if st.button("Fetch Web") and web_q:
        fetch_and_index_web(web_q)

st.write("---")
user_query = st.text_input("Ask your strategic question:")
if st.button("Analyze") and user_query:
    with st.spinner("Analyzing…"):
        answer = chat_with_context(user_query)
    st.markdown("**Response:**")
    st.write(answer)

st.write("---")
st.subheader("📊 Cost & Growth Analysis")
st.info("Use pandas on the assistant’s outputs to compute capex & growth.")
