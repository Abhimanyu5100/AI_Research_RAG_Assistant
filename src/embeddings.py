"""Single source of truth for the embedding model.

Both scripts/build_index.py and src/agent/retriever.py go through here, so the
vectors written to Pinecone are always produced the same way as the vectors
used to query it.
"""
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL_NAME, E5_USE_PREFIXES


class E5Embeddings(HuggingFaceEmbeddings):
    """E5 expects asymmetric prefixes: "query: " for queries, "passage: " for
    indexed text. Without them the model is measurably worse at retrieval."""

    use_prefixes: bool = False

    def embed_query(self, text: str) -> List[float]:
        if self.use_prefixes:
            text = f"query: {text}"
        return super().embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.use_prefixes:
            texts = [f"passage: {t}" for t in texts]
        return super().embed_documents(texts)


def get_embeddings() -> E5Embeddings:
    return E5Embeddings(
        model_name=EMBEDDING_MODEL_NAME,
        use_prefixes=E5_USE_PREFIXES,
    )
