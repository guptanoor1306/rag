import os
import tempfile
import requests
import streamlit as st
import openai
from serpapi import GoogleSearch
from google.oauth2 import service_account
from googleapiclient.discovery import build
import pinecone
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

# --- Configuration via Streamlit secrets ---
# In Settings → Secrets:
# [openai]
# api_key = "YOUR_OPENAI_API_KEY"
# [serpapi]
# api_key = "YOUR_SERPAPI_KEY"
# [gcp]
# service_account = '''{...SERVICE_ACCOUNT_JSON...}'''
# [pinecone]
# api_key = "YOUR_PINECONE_API_KEY"
# environment = "YOUR_PINECONE_ENVIRONMENT"

# Load credentials
openai.api_key = st.secrets.openai.api_key
SERPAPI_KEY   = st.secrets.serpapi.api_key

gcp_info      = st.secrets.gcp.service_account
credentials   = service_account.Credentials.from_service_account_info(
    gcp_info,
    scopes=["https://www.googleapis.com/auth/drive.readonly"]
)
drive_service = build('drive', 'v3', credentials=credentials)

# Initialize Pinecone
pinecone.init(
    api_key=st.secrets.pinecone.api_key,
    environment=st.secrets.pinecone.environment
)
index_name = "zero1"
# Create index if it doesn't exist
if index_name not in pinecone.list_indexes():
    # 1536 is the dimensionality of text-embedding-ada-002
    pinecone.create_index(index_name, dimension=1536, metric="cosine")
index = pinecone.Index(index_name)

# Utility: get embedding
def get_embedding(text: str) -> list:
    resp = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return resp['data'][0]['embedding']

# Extract text from Google Drive files
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
        for page in PdfReader(tmp_path).pages:
            txt = page.extract_text() or ""
            text.append(txt)
        return "\n".join(text)
    else:
        return drive_service.files().export(
            fileId=file_id,
            mimeType='text/plain'
        ).execute().decode('utf-8')

# Index all Docs, Slides, and PDFs from Google Drive
def index_drive_docs():
    with st.spinner("Indexing Google Drive..."):
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
                emb = get_embedding(txt)
                meta = {"name": f['name'], "source": "drive"}
                index.upsert(vectors=[(f['id'], emb, meta)])
            token = res.get('nextPageToken')
            if not token:
                break
        st.success("Drive indexing complete!")

# Fetch and index top-k web results via SerpAPI
def fetch_and_index_web(query, top_k=3):
    with st.spinner(f"Fetching web for '{query}'..."):
        client = GoogleSearch({"q": query, "api_key": SERPAPI_KEY})
        for r in client.get_dict().get('organic_results', [])[:top_k]:
            url = r.get('link')
            if not url:
                continue
            html = requests.get(url, timeout=10).text
            text = " ".join(p.get_text() for p in BeautifulSoup(html, 'html.parser').find_all('p'))
            if not text:
                continue
            emb = get_embedding(text)
            meta = {"name": r.get('title', url), "source": url}
            index.upsert(vectors=[(url, emb, meta)])
        st.success("Web indexing complete!")

# Retrieve top-k contexts from Pinecone
def get_relevant_docs(query: str, top_k: int = 5) -> list:
    emb = get_embedding(query)
    res = index.query(vector=emb, top_k=top_k, include_metadata=True)
    return [match['metadata']['source'] + ": " + match['metadata'].get('name','') for match in res['matches']]

# Chat with context
def chat_with_context(query: str) -> str:
    docs = get_relevant_docs(query)
    context = "\n\n---\n\n".join(docs)
    messages = [
        {"role": "system", "content": "You are a Zero1 strategy assistant."},
        {"role": "user", "content": f"{query}\n\nContext:\n{context}"}
    ]
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7
    )
    return resp.choices[0].message.content

# --- Streamlit UI ---
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
    with st.spinner("Analyzing..."):
        answer = chat_with_context(user_query)
    st.markdown("**Response:**")
    st.write(answer)

st.write("---")
st.subheader("📊 Cost & Growth Analysis")
st.info("Use pandas on the assistant’s numeric outputs to model X crore capex and 100× growth.")
