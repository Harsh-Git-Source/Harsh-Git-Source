# 📚 DocuMind AI — Multimodal RAG for Document Intelligence

An end-to-end **Multimodal Retrieval-Augmented Generation (RAG)** application for extracting, summarizing, retrieving and reasoning over **PDF text, tables and images**.

The project follows the architecture described in the academic project report while providing a clean, modular and runnable application with a Streamlit UI.

---

## 🚀 Problem Statement

Real-world business documents are not text-only. Important information can appear in:

- paragraphs
- tables
- charts
- scanned figures
- images
- headers and lists

Traditional text-only RAG pipelines can miss visual and structured information.

**DocuMind AI** addresses this by extracting multiple modalities from PDFs, creating retrieval-optimized summaries, indexing those summaries, and using multimodal reasoning to answer user questions.

---

## 🧠 Architecture

```text
                         PDF Documents
                               |
                               v
                    Unstructured partition_pdf
                               |
              +----------------+----------------+
              |                |                |
             Text            Tables           Images
              |                |                |
              v                v                v
           Groq LLM         Groq LLM       Gemini Vision
           Summary          Summary          Summary
              |                |                |
              +----------------+----------------+
                               |
                               v
                all-MiniLM-L6-v2 Embeddings
                               |
                               v
                         ChromaDB
                               |
                               v
                   MultiVectorRetriever
                               |
                         User Question
                               |
                               v
                     Relevant Parent Data
                               |
                 +-------------+-------------+
                 |                           |
             Text / Tables                 Images
                 |                           |
                 +-------------+-------------+
                               |
                               v
                     Gemini Multimodal LLM
                               |
                               v
                         Final Answer
```

---

## ✨ Features

- 📄 Multi-PDF upload
- 🧩 Layout-aware PDF element extraction
- 📝 Text summarization
- 📊 Table summarization
- 🖼️ Image understanding
- 🔎 Semantic retrieval
- 🗂️ ChromaDB persistence
- 🧠 LangChain MultiVectorRetriever
- 🤖 Multimodal answer generation
- 📌 Source/page/modality tracking
- 💬 Chat-style interface
- 🔐 Environment-variable based API keys
- 🧪 Basic test suite
- 📦 GitHub-ready project structure

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF parsing | Unstructured |
| Text/table summarization | Groq LLM |
| Vision | Google Gemini |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB |
| Retrieval | LangChain MultiVectorRetriever |
| Language | Python |

---

## 📁 Project Structure

```text
multimodal-rag-document-intelligence/
│
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
│
├── config/
│   └── config.yaml
│
├── src/
│   ├── ingestion/
│   │   └── pdf_parser.py
│   ├── processing/
│   │   └── summarizer.py
│   ├── embeddings/
│   │   └── embedding_model.py
│   ├── retrieval/
│   │   └── multivector_retriever.py
│   ├── llm/
│   │   ├── groq_client.py
│   │   └── gemini_client.py
│   └── pipeline/
│       └── rag_pipeline.py
│
├── data/
├── docs/
├── notebooks/
├── tests/
├── results/
└── screenshots/
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd multimodal-rag-document-intelligence
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> PDF parsing with Unstructured may require additional system packages depending on the PDF strategy and operating system.

### 4. Configure API keys

Copy:

```bash
cp .env.example .env
```

Then add:

```env
GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key
```

Never commit `.env`.

---

## ▶️ Run the application

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in your terminal.

---

## 🔍 Example Questions

After uploading a document, try questions such as:

```text
What are the major findings in this report?

Summarize the financial information in the tables.

What does the chart on page 4 show?

Compare the values mentioned in the retrieved tables.

What are the key recommendations?

Which section contains the information about revenue?
```

---

## 🔐 API Model Configuration

Model IDs can be changed through `.env`:

```env
GROQ_MODEL=llama-3.3-70b-versatile
GEMINI_MODEL=gemini-3.8-flash
```

The academic report used Groq-hosted LLaMA 4 and Gemini Pro Vision. The repository preserves their architectural roles but exposes model IDs through configuration so supported current models can be selected without rewriting the pipeline.

---

## 📌 Project Methodology

The project follows these stages:

1. PDF ingestion
2. Layout-aware element extraction
3. Modality classification
4. Text/table summarization
5. Image summarization
6. Summary embedding
7. Vector storage
8. Multi-vector retrieval
9. Multimodal context assembly
10. Final answer generation

---

## 🎯 Why MultiVectorRetriever?

Instead of embedding large raw content directly, the system indexes compact summaries.

```text
Summary
   ↓
Embedding
   ↓
Vector Search
   ↓
Parent ID
   ↓
Original Content
```

This allows retrieval to match concise semantic representations while preserving richer original content for answer generation.

---

## 📊 Evaluation Philosophy

This is a **system-level multimodal RAG project**, not a conventional supervised classification problem.

Useful evaluation dimensions include:

- retrieval relevance
- answer grounding
- source attribution
- modality coverage
- numerical fidelity
- response latency
- qualitative answer quality

The academic report describes system-level efficiency improvements rather than a classification accuracy benchmark.

---

## ⚠️ Security

Never commit:

- `.env`
- API keys
- private PDFs
- extracted confidential images
- local Chroma databases

The `.gitignore` is configured to keep runtime document data and secrets out of Git.

---

## 👨‍💻 Future Improvements

- Hybrid BM25 + vector retrieval
- Reranking
- OCR-specific processing
- Table-aware retrieval
- Document-level access control
- Evaluation datasets and automated RAG metrics
- Streaming responses
- Production API using FastAPI
- Docker deployment
- Cloud-hosted Chroma
- Conversation memory

---

## 📄 Academic Project Context

The repository is based on the academic project **“Multimodal RAG Model for Document Summarization and Extraction.”**

The report describes PDF parsing with Unstructured, modality-specific summarization, MiniLM embeddings, ChromaDB, MultiVectorRetriever and multimodal reasoning.
