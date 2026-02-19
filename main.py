"""
main.py — FastAPI server
Run: uvicorn main:app --reload
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.documents import Document
from reranker import build_pipeline, RAGPipeline

# ── Docs corpus — AI/ML topic so queries like "deep learning" show real reranking difference ──
# WHY SAME ANSWER BEFORE: old docs were about Paris — zero relevance to AI queries,
# so bi-encoder and cross-encoder both scored all chunks near 0 → same top-3 → same answer.
# Now docs are about AI/ML — reranker can clearly separate relevant from irrelevant chunks.

DOCS = [
    # Deep Learning — directly relevant
    Document(page_content="Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to learn hierarchical representations of data. Each layer learns increasingly abstract features."),
    Document(page_content="Convolutional Neural Networks (CNNs) are deep learning models designed for image recognition. They use convolutional layers to automatically detect spatial features like edges, textures, and shapes."),
    Document(page_content="Recurrent Neural Networks (RNNs) and LSTMs are deep learning architectures designed for sequential data like text and time series. They maintain hidden state across time steps to capture temporal dependencies."),
    Document(page_content="Deep learning requires large amounts of labeled training data and significant compute (GPUs/TPUs). Training involves forward pass, loss calculation, and backpropagation to update weights via gradient descent."),
    Document(page_content="Transformers are a deep learning architecture introduced in 'Attention Is All You Need' (2017). They use self-attention mechanisms and have become the foundation for LLMs like GPT and BERT."),
    

    # Machine Learning — partially relevant
    Document(page_content="Machine learning is a branch of AI where models learn patterns from data rather than being explicitly programmed. Key categories are supervised, unsupervised, and reinforcement learning."),
    Document(page_content="Supervised learning trains models on labeled input-output pairs. Common algorithms include linear regression, decision trees, SVMs, and neural networks."),
    Document(page_content="Overfitting occurs when a model learns training data too well and fails to generalize. Regularization techniques like dropout, L1/L2, and early stopping help prevent it."),

    # Unrelated — tests reranker's ability to filter noise
    Document(page_content="The Python programming language was created by Guido van Rossum in 1991. It is known for its readable syntax and is widely used in data science and web development."),
    Document(page_content="SQL is a query language for relational databases. Common operations include SELECT, INSERT, UPDATE, DELETE, and JOIN across tables."),

    Document(page_content="""
Machine learning (ML) is a core subfield of artificial intelligence that focuses on enabling computers to learn patterns from data and improve their performance on a task without being explicitly programmed with task-specific rules. Instead of hand-crafting logic, developers provide algorithms with data, allowing systems to discover relationships, trends, and structures automatically. This paradigm shift has transformed software engineering, enabling applications that were previously infeasible, such as real-time language translation, recommendation systems, autonomous vehicles, and predictive healthcare.

At its foundation, machine learning relies on statistical principles. Models attempt to approximate an unknown function that maps inputs to outputs. During training, the model adjusts internal parameters to minimize a loss function, which measures the difference between predicted and actual values. The optimization process typically uses gradient-based methods, especially variants of gradient descent, to iteratively improve performance.

Machine learning methods are broadly categorized into supervised, unsupervised, and reinforcement learning. In supervised learning, models are trained on labeled datasets where each input is paired with the correct output. Classification tasks predict discrete labels, such as identifying whether an email is spam, while regression tasks predict continuous values, such as housing prices or temperature forecasts. Common algorithms include linear and logistic regression, decision trees, support vector machines, k-nearest neighbors, and neural networks.

Unsupervised learning operates on unlabeled data and aims to uncover hidden structure. Clustering algorithms group similar data points, while dimensionality reduction techniques compress data into lower-dimensional representations while preserving essential information. These methods are useful for exploratory data analysis, anomaly detection, and feature extraction.

Reinforcement learning differs fundamentally by involving an agent that interacts with an environment. The agent learns a policy — a strategy for choosing actions — based on rewards received from the environment. Over time, it learns to maximize cumulative reward. This paradigm has been used to achieve superhuman performance in games, robotics control, and resource optimization problems.

A central challenge in machine learning is generalization. Models must perform well not only on training data but also on unseen examples. Overfitting occurs when a model memorizes noise or specific patterns in the training data, while underfitting occurs when the model is too simple to capture underlying relationships. Techniques such as regularization, cross-validation, early stopping, and data augmentation help balance this trade-off.

Modern machine learning workflows involve data collection, cleaning, feature engineering, model selection, training, evaluation, and deployment. With the rise of cloud computing and specialized hardware, ML systems can now process massive datasets and operate at global scale. As a result, machine learning has become a foundational technology driving innovation across nearly every industry.
"""),

Document(page_content="""
Deep learning is an advanced branch of machine learning that uses neural networks with many layers to learn complex representations of data. These networks are inspired loosely by the structure of the human brain, consisting of interconnected units called neurons. Each neuron performs a weighted sum of its inputs followed by a nonlinear activation function, enabling the network to approximate highly complex functions.

Traditional machine learning methods often rely on manual feature engineering, where domain experts design input features. Deep learning eliminates much of this manual effort by learning hierarchical features directly from raw data. Lower layers detect simple patterns such as edges in images, while deeper layers capture abstract concepts like objects or faces.

Convolutional Neural Networks (CNNs) are specialized for processing grid-like data such as images. They use convolutional filters to detect spatial patterns and pooling layers to reduce dimensionality while preserving important information. CNNs have achieved remarkable success in computer vision tasks, including image classification, object detection, medical imaging, and autonomous driving.

Recurrent Neural Networks (RNNs) are designed for sequential data such as time series, speech, and text. They maintain hidden states that capture information from previous steps. Variants like Long Short-Term Memory (LSTM) networks address issues such as vanishing gradients, enabling learning over long sequences.

Training deep networks requires large datasets and significant computational power. Backpropagation computes gradients of the loss function with respect to each parameter, allowing optimization algorithms to update weights efficiently. Modern optimizers like Adam accelerate convergence and improve stability.

Despite their power, deep learning models face challenges. They can be computationally expensive, difficult to interpret, and vulnerable to adversarial attacks. Moreover, they often require careful tuning of hyperparameters and large amounts of labeled data. Transfer learning — reusing pretrained models on new tasks — has become a powerful technique to overcome data limitations.

Deep learning has driven breakthroughs in speech recognition, computer vision, natural language understanding, and game playing. Its ability to model nonlinear relationships and extract features automatically makes it one of the most transformative technologies of the 21st century.
"""),

Document(page_content="""
Natural Language Processing (NLP) is a field of artificial intelligence that focuses on enabling computers to understand, interpret, and generate human language. Human language is highly complex, containing ambiguity, context, idioms, and cultural nuances, making NLP one of the most challenging areas of AI.

Early NLP systems relied on rule-based approaches, using handcrafted grammars and dictionaries. These systems struggled with scalability and flexibility. Statistical methods later emerged, modeling language using probability distributions derived from large corpora. Techniques such as n-gram models and TF-IDF representations enabled basic text analysis but lacked deep semantic understanding.

The introduction of neural networks transformed NLP. Word embeddings such as Word2Vec and GloVe represented words as dense vectors, capturing semantic relationships. Contextual embeddings, notably BERT, further improved performance by considering the surrounding words in a sentence, allowing models to distinguish meanings based on context.

Transformer architectures revolutionized NLP by replacing recurrence with self-attention mechanisms. Self-attention allows models to weigh the importance of each word relative to others in a sequence, capturing long-range dependencies efficiently. Transformers scale well to large datasets and form the backbone of modern language models.

NLP tasks include text classification, sentiment analysis, machine translation, question answering, summarization, named entity recognition, and dialogue systems. Applications range from search engines and virtual assistants to legal document analysis and biomedical research.

Challenges remain, including handling low-resource languages, understanding sarcasm and humor, and mitigating biases present in training data. Ethical considerations are also critical, as NLP systems influence public discourse and decision-making.

As models grow larger and more capable, NLP is moving toward systems that can reason, maintain context across long conversations, and interact naturally with humans. This progress is reshaping how people access information and communicate with technology.
"""),

Document(page_content="""
Generative AI refers to artificial intelligence systems capable of creating new content that resembles data they were trained on. Unlike traditional predictive models that classify or estimate values, generative models learn the underlying probability distribution of data and sample from it to produce novel outputs.

Large Language Models (LLMs) are among the most prominent generative AI systems. Built on transformer architectures, they are trained on massive text corpora using self-supervised learning. By predicting the next token in a sequence, they learn grammar, facts, reasoning patterns, and stylistic nuances. LLMs can perform a wide range of tasks without task-specific training, including conversation, summarization, translation, coding, and creative writing.

Generative AI also includes models for images, audio, and video. Generative Adversarial Networks (GANs) consist of two competing networks — a generator and a discriminator — that learn to produce realistic outputs. Diffusion models generate images by gradually transforming random noise into structured visuals through iterative denoising.

Applications of generative AI span many industries. In healthcare, it aids drug discovery and medical imaging. In entertainment, it creates art, music, and visual effects. In software development, it assists with code generation and debugging. Businesses use generative models for marketing content, customer support automation, and data augmentation.

However, generative AI introduces significant challenges. Models can produce hallucinations — outputs that sound plausible but are incorrect. They may also reflect biases present in training data or generate harmful content if not properly controlled. Ensuring safety, fairness, and transparency is an active area of research.

Future developments aim to improve reasoning, factual accuracy, multimodal capabilities, and alignment with human values. Generative AI is increasingly viewed as a collaborative tool that augments human creativity and productivity rather than replacing it. As the technology matures, it is expected to become deeply integrated into everyday workflows and digital experiences.
"""),

]


# ── Schemas ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    use_reranking: bool = True

class CompareRequest(BaseModel):
    question: str


# ── App ───────────────────────────────────────────────────────────────────────

pipeline: RAGPipeline | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = build_pipeline(DOCS)
    yield

app = FastAPI(title="RAG Re-ranking API", version="1.0.0", lifespan=lifespan)


@app.post("/query")
def query(req: AskRequest):
    if not pipeline:
        raise HTTPException(503, "Pipeline not ready")
    return pipeline.query(req.question, req.use_reranking)


@app.post("/compare")
def compare(req: CompareRequest):
    """Before vs after — naive top-3 vs re-ranked top-3 for the same question."""
    if not pipeline:
        raise HTTPException(503, "Pipeline not ready")
    naive    = pipeline.query(req.question, use_reranking=False)
    reranked = pipeline.query(req.question, use_reranking=True)
    return {
        "question":              req.question,
        "naive_answer":          naive["answer"],
        "naive_chunks":          naive["chunks"],
        "naive_latency_ms":      naive["latency_ms"],
        "reranked_answer":       reranked["answer"],
        "reranked_chunks":       reranked["chunks"],
        "reranked_latency_ms":   reranked["latency_ms"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "ready": pipeline is not None}