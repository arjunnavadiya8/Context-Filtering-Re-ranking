"""
reranker.py — core pipeline
Retrieve top-10 → cross-encoder re-rank → best 3 → structured prompt → Groq LLM
"""

import os, time, torch
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Chunk:
    content: str
    score: float
    rank: int


# ── Prompt Templates ──────────────────────────────────────────────────────────

# BEFORE: flat dump — no relevance signal, LLM fills gaps with hallucinations
NAIVE_PROMPT = PromptTemplate.from_template(
    "Answer using only the context below.\n\n"
    "Context: {context}\n\n"
    "Question: {question}\nAnswer:"
)

# AFTER: ranked chunks + strict grounding — hallucination drops significantly
RERANKED_PROMPT = PromptTemplate.from_template(
    "You are a factual assistant. Answer ONLY from the context below.\n"
    "If the answer is not present, say exactly: 'Not found in context.'\n"
    "Do NOT use outside knowledge. Do NOT guess.\n\n"
    "[Chunk 1 — Most Relevant]\n{chunk_1}\n\n"
    "[Chunk 2]\n{chunk_2}\n\n"
    "[Chunk 3]\n{chunk_3}\n\n"
    "Question: {question}\nAnswer:"
)


# ── Re-ranker ─────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    cross-encoder/ms-marco-MiniLM-L-6-v2
    Joint (query, doc) scoring — far more precise than bi-encoder similarity.
    22MB, runs on CPU in ~100ms for 10 docs.
    """
    def __init__(self):
        print("[Reranker] Loading cross-encoder/ms-marco-MiniLM-L-6-v2 ...")
        self.model = CrossEncoder(RERANK_MODEL, device=DEVICE)
        print("[Reranker] Ready.")

    def rerank(self, query: str, docs: list[Document], top_n: int = 3) -> list[Chunk]:
        pairs  = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs).tolist()
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [
            Chunk(content=doc.page_content, score=round(float(s), 4), rank=i + 1)
            for i, (s, doc) in enumerate(ranked[:top_n])
        ]


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore
        self.reranker    = CrossEncoderReranker()
        self.llm         = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)

    def query(self, question: str, use_reranking: bool = True) -> dict:
        t0   = time.time()
        docs = self.vectorstore.similarity_search(question, k=10)   # Step 1: top-10

        if use_reranking:
            chunks = self.reranker.rerank(question, docs, top_n=3)  # Step 2: re-rank
            pad    = [Chunk("No additional context.", 0.0, i) for i in range(3)]
            top3   = (chunks + pad)[:3]                             # Step 3: best 3
            answer = (RERANKED_PROMPT | self.llm | StrOutputParser()).invoke({  # Step 4: structured prompt
                "question": question,
                "chunk_1":  top3[0].content,
                "chunk_2":  top3[1].content,
                "chunk_3":  top3[2].content,
            })
        else:
            chunks = [Chunk(d.page_content, round(1.0 - i * 0.1, 2), i + 1) for i, d in enumerate(docs[:3])]
            answer = (NAIVE_PROMPT | self.llm | StrOutputParser()).invoke({
                "context":  "\n\n".join(c.content for c in chunks),
                "question": question,
            })

        return {
            "answer":     answer,
            "reranked":   use_reranking,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "chunks":     [{"rank": c.rank, "score": c.score, "content": c.content} for c in chunks],
        }


# ── Factory ───────────────────────────────────────────────────────────────────

def build_pipeline(docs: list[Document]) -> RAGPipeline:
    print("[Pipeline] Building FAISS index ...")
    embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": DEVICE})
    vectorstore = FAISS.from_documents(docs, embeddings)
    print(f"[Pipeline] Indexed {len(docs)} documents. Ready.")
    return RAGPipeline(vectorstore)