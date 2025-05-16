Zero1 RAG App: Minimal Cost Setup & Configuration

1. Minimizing Costs

Local Vector Store: Run ChromaDB locally (no hosting fees) instead of Pinecone/Qdrant paid tiers.

Embedding Model Choice: Use text-embedding-ada-002—the most cost-efficient embedding with good quality.

Batch Indexing: Index Drive docs and web pages in batches (e.g., nightly) to reduce API calls.

Limit Web Fetches: Restrict SerpAPI queries per session (e.g., top-3 results) or schedule periodic crawls.

Streamlit Free Tier: Deploy on Streamlit Cloud’s free tier; avoid heavy concurrency to stay within quotas.

Caching: Leverage st.experimental_singleton or st.cache_data for Drive lists and web pages.

API Call Optimization: Combine text before embedding (up to 2048 tokens) to reduce embedding calls.
