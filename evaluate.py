# """
# evaluate.py — hallucination analysis report
# Run: python evaluate.py

# NOTE: Set GROQ_API_KEY as an environment variable before running:
#   Windows PowerShell: $env:GROQ_API_KEY = "gsk_..."
#   Windows CMD:        set GROQ_API_KEY=gsk_...
#   Linux/Mac:          export GROQ_API_KEY=gsk_...
# """

# import os, json
# from langchain_groq import ChatGroq
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.documents import Document
# from reranker import build_pipeline

# GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# if not GROQ_API_KEY:
#     raise EnvironmentError(
#         "\n\nGROQ_API_KEY not set!\n"
#         "PowerShell: $env:GROQ_API_KEY = 'gsk_your_key_here'\n"
#         "Get a free key at: https://console.groq.com\n"
#     )

# # ── Judge prompt ──────────────────────────────────────────────────────────────

# JUDGE = PromptTemplate.from_template(
#     "Score how faithful the ANSWER is to the CONTEXT.\n"
#     "1.0=fully grounded | 0.5=partial | 0.0=hallucinated\n"
#     'Reply ONLY with JSON: {{"score": <0.0-1.0>, "hallucinated_claims": ["..."]}}\n\n'
#     "CONTEXT: {context}\nANSWER: {answer}\nJSON:"
# )

# # ── 5 benchmark queries (AI/ML domain matches the corpus) ─────────────────────

# QUERIES = [
#     "What is deep learning and how does it work?",
#     "How do Convolutional Neural Networks work?",
#     "What is the Transformer architecture and why is it important?",
#     "What causes overfitting and how can it be prevented?",
#     "What is the difference between deep learning and machine learning?",
# ]

# # Same docs as main.py — mix of relevant, partially relevant, and noise
# DOCS = [
#     Document(page_content="Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to learn hierarchical representations of data. Each layer learns increasingly abstract features."),
#     Document(page_content="Convolutional Neural Networks (CNNs) are deep learning models designed for image recognition. They use convolutional layers to automatically detect spatial features like edges, textures, and shapes."),
#     Document(page_content="Recurrent Neural Networks (RNNs) and LSTMs are deep learning architectures designed for sequential data like text and time series. They maintain hidden state across time steps to capture temporal dependencies."),
#     Document(page_content="Deep learning requires large amounts of labeled training data and significant compute (GPUs/TPUs). Training involves forward pass, loss calculation, and backpropagation to update weights via gradient descent."),
#     Document(page_content="Transformers are a deep learning architecture introduced in 'Attention Is All You Need' (2017). They use self-attention mechanisms and have become the foundation for LLMs like GPT and BERT."),
#     Document(page_content="Machine learning is a branch of AI where models learn patterns from data rather than being explicitly programmed. Key categories are supervised, unsupervised, and reinforcement learning."),
#     Document(page_content="Supervised learning trains models on labeled input-output pairs. Common algorithms include linear regression, decision trees, SVMs, and neural networks."),
#     Document(page_content="Overfitting occurs when a model learns training data too well and fails to generalize. Regularization techniques like dropout, L1/L2, and early stopping help prevent it."),
#     Document(page_content="The Python programming language was created by Guido van Rossum in 1991. It is known for its readable syntax and is widely used in data science and web development."),
#     Document(page_content="SQL is a query language for relational databases. Common operations include SELECT, INSERT, UPDATE, DELETE, and JOIN across tables."),
# ]


# # ── Helpers ───────────────────────────────────────────────────────────────────

# def score_answer(context: str, answer: str, llm) -> tuple[float, list[str]]:
#     raw = (JUDGE | llm | StrOutputParser()).invoke({"context": context, "answer": answer}).strip()
#     try:
#         if "```" in raw:
#             raw = raw.split("```")[1].replace("json", "").strip()
#         data = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
#         return float(data.get("score", 0.5)), data.get("hallucinated_claims", [])
#     except Exception:
#         return 0.5, ["[parse error — defaulted to 0.5]"]


# # ── Main ──────────────────────────────────────────────────────────────────────

# def main():
#     pipeline = build_pipeline(DOCS)
#     judge    = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=GROQ_API_KEY)
#     results  = []

#     print("\n" + "=" * 60)
#     print("Running evaluation on 5 queries...")
#     print("=" * 60 + "\n")

#     for i, q in enumerate(QUERIES, 1):
#         print(f"[{i}/5] {q}")
#         naive    = pipeline.query(q, use_reranking=False)
#         reranked = pipeline.query(q, use_reranking=True)

#         naive_ctx    = " | ".join(c["content"] for c in naive["chunks"])
#         reranked_ctx = " | ".join(c["content"] for c in reranked["chunks"])

#         ns, nh = score_answer(naive_ctx,    naive["answer"],    judge)
#         rs, rh = score_answer(reranked_ctx, reranked["answer"], judge)

#         results.append({
#             "query":    q,
#             "naive":    {"answer": naive["answer"],    "score": ns, "hallucinations": nh,
#                          "top_chunk": naive["chunks"][0]["content"][:80] + "..."},
#             "reranked": {"answer": reranked["answer"], "score": rs, "hallucinations": rh,
#                          "top_chunk": reranked["chunks"][0]["content"][:80] + "..."},
#         })
#         print(f"       Naive faithfulness={ns:.2f}  |  Reranked faithfulness={rs:.2f}  |  Δ={rs - ns:+.2f}\n")

#     # ── Hallucination Report ──────────────────────────────────────────────────
#     avg_n = sum(r["naive"]["score"]    for r in results) / len(results)
#     avg_r = sum(r["reranked"]["score"] for r in results) / len(results)
#     hr_n  = 1 - avg_n
#     hr_r  = 1 - avg_r
#     redux = (hr_n - hr_r) / max(hr_n, 1e-6) * 100

#     print("=" * 60)
#     print("HALLUCINATION ANALYSIS REPORT")
#     print("=" * 60)
#     print(f"  Baseline hallucination rate       : {hr_n:.1%}")
#     print(f"  Post-reranking hallucination rate : {hr_r:.1%}")
#     print(f"  Reduction                         : {redux:.1f}%")
#     print(f"  Method : G-Eval faithfulness (Groq llama-3.1-8b as judge)")
#     print(f"  Formula: hallucination_rate = 1 − mean(faithfulness_score)")

#     print("\n" + "-" * 60)
#     print("BEFORE vs AFTER — 5 QUERIES")
#     print("-" * 60)
#     for r in results:
#         print(f"\nQ: {r['query']}")
#         print(f"  Naive    top chunk : {r['naive']['top_chunk']}")
#         print(f"  Reranked top chunk : {r['reranked']['top_chunk']}")
#         print(f"  NAIVE    [{r['naive']['score']:.2f}] : {r['naive']['answer'][:180]}...")
#         print(f"  RERANKED [{r['reranked']['score']:.2f}] : {r['reranked']['answer'][:180]}...")
#         if r["naive"]["hallucinations"] and r["naive"]["hallucinations"] != ["[parse error — defaulted to 0.5]"]:
#             print(f"  Hallucinated claims: {r['naive']['hallucinations']}")


# if __name__ == "__main__":
#     main()


"""
evaluate.py — hallucination analysis report
Run:
  1. pip install langchain-groq
  2. $env:GROQ_API_KEY = "gsk_your_key"
  3. python evaluate.py
"""
"""
evaluate.py — hallucination analysis report
Run:
  1. pip install langchain-groq
  2. $env:GROQ_API_KEY = "gsk_your_key"
  3. python evaluate.py
"""

import os, json, sys

# ── Dependency check with helpful message ─────────────────────────────────────
try:
    from langchain_groq import ChatGroq
except ModuleNotFoundError:
    sys.exit(
        "\n[ERROR] langchain-groq not installed.\n"
        "Fix: pip install langchain-groq\n"
        "Then re-run: python evaluate.py\n"
    )

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from reranker import build_pipeline

# ── API key check ─────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    sys.exit(
        "\n[ERROR] GROQ_API_KEY not set.\n"
        "PowerShell: $env:GROQ_API_KEY = 'gsk_your_key_here'\n"
        "Get free key: https://console.groq.com\n"
    )

# ── Judge prompt ──────────────────────────────────────────────────────────────

JUDGE = PromptTemplate.from_template(
    "Score how faithful the ANSWER is to the CONTEXT.\n"
    "1.0=fully grounded | 0.5=partial | 0.0=hallucinated\n"
    'Reply ONLY with JSON: {{"score": <0.0-1.0>, "hallucinated_claims": ["..."]}}\n\n'
    "CONTEXT: {context}\nANSWER: {answer}\nJSON:"
)

# ── 5 benchmark queries ───────────────────────────────────────────────────────

QUERIES = [
    "What is deep learning and how does it work?",
    "How do Convolutional Neural Networks work?",
    "What is the Transformer architecture and why is it important?",
    "What causes overfitting and how can it be prevented?",
    "What is the difference between deep learning and machine learning?",
]

DOCS = [
    Document(page_content="Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn hierarchical representations of data. Each layer learns increasingly abstract features."),
    Document(page_content="Convolutional Neural Networks (CNNs) are deep learning models designed for image recognition. They use convolutional layers to automatically detect spatial features like edges, textures, and shapes."),
    Document(page_content="Recurrent Neural Networks (RNNs) and LSTMs are deep learning architectures for sequential data like text and time series. They maintain hidden state across time steps."),
    Document(page_content="Deep learning requires large labeled datasets and GPUs/TPUs. Training uses forward pass, loss calculation, and backpropagation to update weights via gradient descent."),
    Document(page_content="Transformers were introduced in 'Attention Is All You Need' (2017). They use self-attention and are the foundation for LLMs like GPT and BERT."),
    Document(page_content="Machine learning is a branch of AI where models learn patterns from data rather than explicit programming. Key types: supervised, unsupervised, reinforcement learning."),
    Document(page_content="Supervised learning trains on labeled input-output pairs. Common algorithms: linear regression, decision trees, SVMs, neural networks."),
    Document(page_content="Overfitting occurs when a model memorizes training data and fails to generalize. Solutions: dropout, L1/L2 regularization, early stopping, more data."),
    Document(page_content="Python was created by Guido van Rossum in 1991. It is widely used in data science, web development, and automation due to its readable syntax."),
    Document(page_content="SQL is a query language for relational databases. Common operations: SELECT, INSERT, UPDATE, DELETE, and JOIN across tables."),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def score_answer(context: str, answer: str, llm) -> tuple[float, list[str]]:
    raw = (JUDGE | llm | StrOutputParser()).invoke({"context": context, "answer": answer}).strip()
    try:
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        data = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        return float(data.get("score", 0.5)), data.get("hallucinated_claims", [])
    except Exception:
        return 0.5, ["[parse error]"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pipeline = build_pipeline(DOCS)
    judge    = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=GROQ_API_KEY)
    results  = []

    print("\n" + "=" * 60)
    print("Running evaluation on 5 queries ...")
    print("=" * 60 + "\n")

    for i, q in enumerate(QUERIES, 1):
        print(f"[{i}/5] {q}")
        naive    = pipeline.query(q, use_reranking=False)
        reranked = pipeline.query(q, use_reranking=True)

        naive_ctx    = " | ".join(c["content"] for c in naive["chunks"])
        reranked_ctx = " | ".join(c["content"] for c in reranked["chunks"])

        ns, nh = score_answer(naive_ctx,    naive["answer"],    judge)
        rs, rh = score_answer(reranked_ctx, reranked["answer"], judge)

        results.append({
            "query":    q,
            "naive":    {"answer": naive["answer"],    "score": ns, "hallucinations": nh,
                         "top_chunk": naive["chunks"][0]["content"][:80] + "..."},
            "reranked": {"answer": reranked["answer"], "score": rs, "hallucinations": rh,
                         "top_chunk": reranked["chunks"][0]["content"][:80] + "..."},
        })
        print(f"       Naive={ns:.2f}  Reranked={rs:.2f}  Δ={rs - ns:+.2f}\n")

    # ── Report ────────────────────────────────────────────────────────────────
    avg_n = sum(r["naive"]["score"]    for r in results) / len(results)
    avg_r = sum(r["reranked"]["score"] for r in results) / len(results)
    hr_n  = 1 - avg_n
    hr_r  = 1 - avg_r
    redux = (hr_n - hr_r) / max(hr_n, 1e-6) * 100

    print("=" * 60)
    print("HALLUCINATION ANALYSIS REPORT")
    print("=" * 60)
    print(f"  Baseline hallucination rate       : {hr_n:.1%}")
    print(f"  Post-reranking hallucination rate : {hr_r:.1%}")
    print(f"  Reduction                         : {redux:.1f}%")
    print(f"  Method  : G-Eval faithfulness (Groq llama-3.1-8b as judge)")
    print(f"  Formula : hallucination_rate = 1 - mean(faithfulness_score)")

    print("\n" + "-" * 60)
    print("BEFORE vs AFTER — 5 QUERIES")
    print("-" * 60)
    for r in results:
        print(f"\nQ: {r['query']}")
        print(f"  Naive    top chunk : {r['naive']['top_chunk']}")
        print(f"  Reranked top chunk : {r['reranked']['top_chunk']}")
        print(f"  NAIVE    [{r['naive']['score']:.2f}] : {r['naive']['answer'][:180]}...")
        print(f"  RERANKED [{r['reranked']['score']:.2f}] : {r['reranked']['answer'][:180]}...")
        if r["naive"]["hallucinations"] and "[parse error]" not in r["naive"]["hallucinations"]:
            print(f"  Hallucinated      : {r['naive']['hallucinations']}")


if __name__ == "__main__":
    main()