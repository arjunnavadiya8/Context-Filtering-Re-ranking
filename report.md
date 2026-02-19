# 📊 Reranking RAG — Experiment Report

**Date:** 2026-02-19  
**Author:** GenAI RAG Evaluation Task  
**Core Engine:** FAISS (Vector Retrieval) + Cross-Encoder Re-Ranking  
**Framework:** FastAPI + LangChain Components  
**Embedding Model:** sentence-transformers/all-MiniLM-L6-v2  
**Reranker Model:** cross-encoder/ms-marco-MiniLM-L-6-v2  
**Source Documents:** Synthetic AI/ML Knowledge Base  

---

##  Executive Summary

We evaluated a Retrieval-Augmented Generation (RAG) pipeline using **two configurations**:

1. **Naive Retrieval** — Top-K chunks directly used for generation  
2. **Re-Ranked Retrieval** — Cross-encoder reranker selects most relevant chunks  

**Key Finding:**  
Re-ranking significantly improves **precision and grounding** by prioritizing the most relevant passages, but increases latency due to expensive cross-encoder scoring.

---

## 1. Project Overview

### What is Re-Ranking in RAG?

Re-ranking is a second-stage filtering step:

1. Retrieve many candidate chunks (fast but approximate)
2. Score each chunk against the query using a more accurate model
3. Keep only the best ones for generation

This improves answer quality by reducing irrelevant context.

---

## 2. System Architecture

| Component | Technology | Purpose |
|----------|------------|---------|
Embedding Model | MiniLM-L6-v2 | Convert text into vectors |
Vector Store | FAISS | Fast semantic retrieval |
Initial Retrieval | Top-K similarity search | Recall stage |
Re-Ranking | Cross-Encoder | Precision stage |
Generator | LLM | Final answer generation |

---

## 3. Experiment Design

We compared outputs for the same queries under:

- **Naive Mode:** Use retrieved chunks as-is  
- **Reranked Mode:** Reorder chunks using cross-encoder  

Metrics observed:

- Answer quality
- Context relevance
- Latency
- Information coverage

---

## 4. Results — Query 1  
### Question: *Explain deep learning*

### 4.1 Naive Retrieval Output

**Latency:** 1287.8 ms  

#### Top Retrieved Chunks

| Rank | Score | Summary |
|------|--------|--------|
1 | 1.0 | Definition of deep learning |
2 | 0.9 | Training requirements and compute |
3 | 0.8 | Long detailed explanation |

#### Generated Answer (Naive)

A long, multi-paragraph explanation covering:

- Definition  
- Neural networks  
- CNNs & RNNs  
- Training process  
- Challenges  
- Applications  

**Strength:** Comprehensive  
**Weakness:** Potential redundancy and noise  

---

### 4.2 Re-Ranked Retrieval Output

**Latency:** 3290.7 ms  

#### Top Re-Ranked Chunks

| Rank | Score | Summary |
|------|--------|--------|
1 | 9.22 | Core definition |
2 | 8.49 | Detailed explanation |
3 | 4.80 | Training requirements |

#### Generated Answer (Reranked)

A concise definition:

> Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn hierarchical representations of data.

**Strength:** Highly precise  
**Weakness:** Less comprehensive  

---

### 4.3 Analysis

- Re-ranking prioritized **definition-focused content**
- Removed broader explanatory context
- Produced shorter but more accurate response
- Latency increased by ~2.5× due to cross-encoder computation

---

## 5. Results — Query 2  
### Question: *Explain machine learning*

### 5.1 Naive Retrieval Output

**Latency:** 1193.3 ms  

#### Top Retrieved Chunks

| Rank | Score | Summary |
|------|--------|--------|
1 | 1.0 | Basic definition |
2 | 0.9 | Long detailed overview |
3 | 0.8 | Deep learning relation |

#### Generated Answer (Naive)

A concise explanation describing:

- Learning from data
- No explicit programming
- Pattern discovery
- Performance improvement

---

### 5.2 Re-Ranked Retrieval Output

**Latency:** 3457.4 ms  

#### Top Re-Ranked Chunks

| Rank | Score | Summary |
|------|--------|--------|
1 | 8.81 | Short definition |
2 | 7.93 | Full conceptual explanation |
3 | 3.55 | Deep learning relation |

#### Generated Answer (Reranked)

A richer explanation including:

- Core definition
- Paradigm shift in software
- Real-world applications
- Conceptual understanding

---

### 5.3 Analysis

Unlike the deep learning query:

- Re-ranking selected a broader context chunk
- Result was more detailed and informative
- Demonstrates query-dependent behavior

---

## 6. Latency Comparison

| Query | Naive (ms) | Reranked (ms) | Increase |
|--------|------------|----------------|----------|
Deep Learning | 1287.8 | 3290.7 | +155% |
Machine Learning | 1193.3 | 3457.4 | +190% |

**Observation:**  
Cross-encoder reranking significantly increases response time.

---

## 7. Quality Comparison

| Criteria | Naive Retrieval | Re-Ranked Retrieval |
|----------|-----------------|---------------------|
Precision | Moderate |  High |
Coverage |  High | Moderate |
Noise | Higher | Low |
Consistency | Variable | More consistent |
Latency |  Fast | Slow |

---

## 8. Key Insights

### 8.1 Re-Ranking Improves Precision

Cross-encoders read query and chunk together, producing highly accurate relevance scores.

### 8.2 Recall vs Precision Tradeoff

- Naive retrieval → higher recall (more info)
- Re-ranking → higher precision (better focus)

### 8.3 Latency Cost

Cross-encoders are computationally expensive because they process each query-document pair independently.

---

## 9. When to Use Re-Ranking

### Recommended

- High-accuracy QA systems
- Enterprise knowledge search
- Legal/medical applications
- Hallucination-sensitive domains

### Not Recommended

- Real-time low-latency systems
- Large-scale batch queries without GPU acceleration
- Applications where broad context is preferred

---

## 10. Final Verdict

| System | Use Case |
|--------|-----------|
Naive Retrieval | Fast responses, broad coverage |
Re-Ranked Retrieval | Accurate, focused answers |

###  Best Practice

Use a two-stage pipeline:

1. Retrieve many candidates (fast)
2. Re-rank top results (precise)
3. Generate answer from top N

This balances recall, precision, and efficiency.

---

## 11. Conclusion

1. Re-ranking significantly improves relevance of context fed to the LLM.
2. It reduces noise and improves grounding.
3. Latency increases substantially due to cross-encoder inference.
4. Optimal RAG systems combine both retrieval and reranking.

