# RAG Re-ranking Pipeline

A high-performance RAG (Retrieval-Augmented Generation) pipeline that reduces hallucinations by implementing a multi-stage retrieval and re-ranking process.

## Architecture

The pipeline follows a sophisticated 4-stage process to ensure maximum relevance:
1.  **Retrieval (Top-10)**: Uses FAISS and HuggingFace embeddings to retrieve the 10 most similar chunks.
2.  **Re-ranking (Top-3)**: Implements a Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`) to score query-document pairs. This is far more precise than simple vector similarity.
3.  **Refinement**: Selects the best 3 chunks and pads with "No additional context" if necessary.
4.  **Generation**: Passes the ranked chunks into a strict, structured prompt delivered via Groq (Llama 3.1).

## Project Structure

- `reranker.py`: Core logic for the RAG pipeline and Cross-Encoder re-ranker.
- `main.py`: FastAPI server providing REST endpoints for queries and comparisons.
- `evaluate.py`: Benchmarking tool to calculate hallucination rates using G-Eval (LLM-as-a-judge).
- `requirements.txt`: Project dependencies.
- `.env`: Environment variables (API keys and model configurations).

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Environment**:
    Create a `.env` file with your Groq API key:
    ```env
    GROQ_API_KEY=your_gsk_key_here
    GROQ_MODEL=llama-3.1-8b-instant
    ```

## Usage

### Running the API Server
```bash
uvicorn main:app --reload
```
The server exposes:
- `POST /query`: Standard RAG query.
- `POST /compare`: Returns both Naive and Re-ranked results for evaluation.
- `GET /health`: System status check.

### Running Evaluation
To generate a hallucination analysis report:
```bash
$env:GROQ_API_KEY = "your_key"
python evaluate.py
```

## Performance Benefits
As seen in `report.txt`, re-ranking significantly improves faithfulness scores by ensuring the most relevant context is placed at the top of the prompt, reducing the likelihood of the LLM "hallucinating" from noise or irrelevant chunks.
