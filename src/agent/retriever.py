import logging
import os

from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from src.config import (
    BM25_PATH,
    EMBEDDING_DIM,
    INDEX_NAME,
    PINECONE_API_KEY,
    RETRIEVER_ALPHA,
    RETRIEVER_TOP_K,
)
from src.embeddings import get_embeddings

logger = logging.getLogger(__name__)

if not PINECONE_API_KEY:
    raise RuntimeError(
        "PINECONE_API_KEY is not set. Add it to your .env file before starting the app."
    )
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to the index, creating an empty one only if it is genuinely missing.
if INDEX_NAME not in pc.list_indexes().names():
    logger.warning("Index %r not found -- creating an empty one. Run "
                   "scripts/build_index.py to populate it.", INDEX_NAME)
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)

# Load the BM25 parameters that were fitted on this corpus. Falling back to
# BM25Encoder().default() would load generic MSMARCO statistics that do not
# match what was indexed, so warn loudly rather than degrading in silence.
if BM25_PATH.exists():
    bm25_encoder = BM25Encoder().load(str(BM25_PATH))
else:
    logger.warning(
        "No fitted BM25 parameters at %s -- falling back to generic defaults. "
        "Sparse retrieval will not match the indexed corpus. "
        "Run scripts/build_index.py to generate them.",
        BM25_PATH,
    )
    bm25_encoder = BM25Encoder().default()

embeddings = get_embeddings()

retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings,
    sparse_encoder=bm25_encoder,
    index=index,
    top_k=RETRIEVER_TOP_K,
    alpha=RETRIEVER_ALPHA,
)
