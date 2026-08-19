"""Build the Pinecone hybrid index from the extracted paper text in data/.

Usage:
    python scripts/build_index.py            # refuse to touch an existing index
    python scripts/build_index.py --rebuild  # delete and recreate it
"""
import argparse
import os
import sys

from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone_text.sparse import BM25Encoder

from src.config import (
    BM25_PATH,
    DATA_DIR,
    EMBEDDING_DIM,
    INDEX_NAME,
    PINECONE_API_KEY,
    RETRIEVER_ALPHA,
    RETRIEVER_TOP_K,
)
from src.embeddings import get_embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the existing index first. Destroys all indexed vectors.",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()

    if not PINECONE_API_KEY:
        sys.exit("PINECONE_API_KEY is not set. Add it to your .env file.")
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

    print(f"Loading documents from {DATA_DIR} ...")
    loader = DirectoryLoader(str(DATA_DIR), glob="pdf_*.txt", loader_cls=TextLoader)
    docs = loader.load()
    if not docs:
        sys.exit(f"No pdf_*.txt files found in {DATA_DIR}. Run scripts/scrape_arxiv.py first.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    documents = splitter.split_documents(docs)
    print(f"{len(docs)} documents -> {len(documents)} chunks")

    texts = [d.page_content for d in documents]
    # Carry `source` through to Pinecone -- this is what lets the agent cite
    # which paper an answer came from. Without it every citation list is empty.
    metadatas = [{"source": d.metadata.get("source", "unknown")} for d in documents]

    pc = Pinecone(api_key=PINECONE_API_KEY)
    exists = INDEX_NAME in pc.list_indexes().names()

    if exists and not args.rebuild:
        stats = pc.Index(INDEX_NAME).describe_index_stats()
        sys.exit(
            f"Index {INDEX_NAME!r} already exists with "
            f"{stats.get('total_vector_count', 0)} vectors.\n"
            "Re-run with --rebuild to delete and recreate it."
        )

    print("Fitting BM25 on the corpus ...")
    bm25_encoder = BM25Encoder().default()
    bm25_encoder.fit(texts)
    bm25_encoder.dump(str(BM25_PATH))
    print(f"Wrote BM25 parameters to {BM25_PATH}")

    if exists:
        print(f"Deleting existing index: {INDEX_NAME}")
        pc.delete_index(INDEX_NAME)

    print("Creating hybrid index (dotproduct is required for hybrid search) ...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    index = pc.Index(INDEX_NAME)
    retriever = PineconeHybridSearchRetriever(
        embeddings=get_embeddings(),
        sparse_encoder=bm25_encoder,
        index=index,
        top_k=RETRIEVER_TOP_K,
        alpha=RETRIEVER_ALPHA,
    )

    print("Uploading vectors in batches ...")
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Uploading"):
        retriever.add_texts(
            texts[i : i + args.batch_size],
            metadatas=metadatas[i : i + args.batch_size],
        )

    print(index.describe_index_stats())


if __name__ == "__main__":
    main()
