# RAG Pipeline with LangChain and ChromaDB

A Retrieval-Augmented Generation (RAG) pipeline that answers questions from PDF documents. It loads PDFs, splits them into chunks, embeds them with Sentence-Transformers, stores them in ChromaDB, retrieves the most relevant chunks using cosine similarity, and passes them to an LLM to generate answers grounded in the documents.

## Pipeline

```
PDFs -> Documents -> Chunks -> Embeddings -> ChromaDB
                                                |
Question -> Query embedding -> Cosine-similarity retrieval (top-3) -> LLM -> Answer
```

## Configuration

| Setting | Value |
|---|---|
| Documents | 2 PDFs (32 pages total) |
| Chunking | RecursiveCharacterTextSplitter, chunk size 500, overlap 50 |
| Chunks created | 321 |
| Embedding model | all-MiniLM-L6-v2 (384 dimensions) |
| Vector store | ChromaDB (persistent) |
| Retrieval | Cosine similarity, top-3 chunks |
| LLMs | Groq (qwen/qwen3.8-27b) and a GPT model via an OpenAI-compatible API |

## Tech Stack

Python, LangChain, ChromaDB, Sentence-Transformers, scikit-learn, PyPDF, PyMuPDF, Groq

## Data

The PDFs are not included in this repo. The sample run indexed two papers:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) (Gao et al., 2023)

The notebook also shows how `TextLoader` reads a plain text file, using a short `Python.txt` description of the Python language.

To run it yourself, create a `data/pdfs/` folder and add your own PDFs.

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

## Run

1. Open `RAG_pipeline.ipynb` in Jupyter.
2. Add your PDFs to `data/pdfs/`.
3. Click **Kernel, then Restart Kernel and Run All Cells**.
4. When prompted, paste your LLM API keys. They are entered at runtime and never stored in the notebook.

If you rerun the notebook, delete the `data/vector_store` folder first so chunks aren't stored twice.

## Example Questions

- **"What is an encoder-decoder?"**: the answer describes the encoder and decoder stacks from the Transformer paper (6 layers each, with self-attention and feed-forward sub-layers), using the retrieved chunks.
<img width="1445" height="522" alt="sample_ans png" src="https://github.com/user-attachments/assets/88b6e500-e96e-4586-9b22-31c1b006e563" />

- **"What is RAG?"**: the answer describes RAG as a framework that has evolved through different paradigms, based on the retrieved context.
<img width="1722" height="410" alt="sample_ans2 png" src="https://github.com/user-attachments/assets/46ce83ed-f869-420e-a925-b0b4e56b3970" />

## Limitations and Next Steps

- No evaluation set yet to measure answer quality
- Add a similarity threshold so the system says "not found" when nothing relevant is retrieved
- Add a Streamlit interface
- Improve retrieval quality (larger or smarter chunking, re-ranking)

## Author

**Daman Dhillon **
[LinkedIn](https://www.linkedin.com/in/daman-jeet-kaur-dhillon-324191312/) | [GitHub]
