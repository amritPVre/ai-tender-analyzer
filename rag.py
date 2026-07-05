"""RAG pipeline: chunking, embedding, FAISS indexing, retrieval."""

from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import CHUNK_CHARS, OVERLAP_CHARS, RETRIEVE_K

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


class RAGPipeline:
    """Local RAG pipeline without LangChain."""

    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: list[str] = []
        self.document_text: str = ""

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(EMBED_MODEL_NAME)
        return self.model

    @staticmethod
    def chunk_text(text: str, chunk_size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
        """Sliding-window character-based chunking with overlap."""
        text = text.strip()
        if not text:
            return []

        chunks = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + chunk_size, length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= length:
                break
            start += chunk_size - overlap

        return chunks

    def build(self, document_text: str) -> int:
        """Chunk, embed, and index document. Returns chunk count."""
        self.document_text = document_text
        self.chunks = self.chunk_text(document_text)

        if not self.chunks:
            self.index = None
            return 0

        model = self._get_model()
        embeddings = model.encode(self.chunks, show_progress_bar=False, convert_to_numpy=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        return len(self.chunks)

    def retrieve(self, query: str, k: int = RETRIEVE_K) -> list[str]:
        """Retrieve top-k semantically relevant chunks."""
        if not self.chunks or self.index is None:
            return []

        model = self._get_model()
        query_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
        k = min(k, len(self.chunks))
        _, indices = self.index.search(query_emb, k)

        return [self.chunks[i] for i in indices[0] if i >= 0]

    @staticmethod
    def assemble_context(chunks: list[str]) -> str:
        """Assemble retrieved chunks into a context block."""
        if not chunks:
            return "No relevant context found in the document."
        return "\n\n---\n\n".join(chunks)
