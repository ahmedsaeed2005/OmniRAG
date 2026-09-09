# 🚀 OmniRAG

### Advanced Hybrid & Multimodal Retrieval-Augmented Generation System

OmniRAG is an advanced **Retrieval-Augmented Generation (RAG)** system designed to provide accurate, context-aware answers from documents by combining multiple retrieval and understanding techniques.

The system goes beyond traditional semantic search by combining:

**Semantic Search + BM25 + Hybrid Retrieval + Reranking + OCR + Vision + Gemini + LangGraph**

---

## 🎯 Project Overview

Traditional RAG systems often depend mainly on vector similarity search. While semantic search is powerful, it may miss important exact keywords, technical terms, numbers, and specific phrases.

OmniRAG addresses this limitation by combining multiple retrieval strategies into a unified pipeline.

The system can:

* 📄 Process PDF documents
* 🔎 Perform semantic retrieval
* 🔤 Perform keyword-based retrieval using BM25
* 🔀 Combine semantic and keyword results using hybrid retrieval
* 🎯 Rerank retrieved documents
* 🖼️ Extract and analyze images
* 🔍 Extract text from images using OCR
* 🤖 Generate answers using Google Gemini
* 🔄 Orchestrate the workflow using LangGraph
* 🖥️ Provide an interactive Streamlit interface

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │      User Query     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Query Parser     │
                    └──────────┬──────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │ Semantic Search  │      │   BM25 Search    │
        │     FAISS        │      │ Keyword Search   │
        └────────┬─────────┘      └────────┬─────────┘
                 │                         │
                 └────────────┬────────────┘
                              ▼
                    ┌─────────────────────┐
                    │  Hybrid Retrieval  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Reranker        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Retrieved Context  │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
             ┌──────────────┐     ┌──────────────┐
             │ OCR / Vision │     │ Text Context │
             └──────┬───────┘     └──────┬───────┘
                    │                    │
                    └─────────┬──────────┘
                              ▼
                    ┌─────────────────────┐
                    │   Google Gemini     │
                    │   LLM Generation    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Final Answer   │
                    └─────────────────────┘
```

---

# 🔥 Key Features

## 1. Hybrid Retrieval

OmniRAG combines two complementary retrieval approaches:

| Method           | Focus    | Advantage                       |
| ---------------- | -------- | ------------------------------- |
| Semantic Search  | Meaning  | Understands semantic similarity |
| BM25             | Keywords | Excellent for exact terms       |
| Hybrid Retrieval | Both     | More robust retrieval           |

### Why Hybrid Retrieval?

For example, if the user asks:

> "What is the definition of Hadoop?"

Semantic search can understand the meaning of the question, while BM25 can specifically match the keyword **Hadoop**.

Combining both approaches improves the probability of retrieving the most relevant context.

---

# 🧠 Semantic Search

The semantic retrieval component converts documents and queries into vector embeddings.

OmniRAG uses:

```text
Sentence Transformers
        ↓
Text Embeddings
        ↓
FAISS Vector Store
        ↓
Similarity Search
```

This allows the system to retrieve documents based on **meaning rather than exact word matching**.

---

# 🔤 BM25 Keyword Search

BM25 provides lexical retrieval based on keyword matching.

This is particularly useful when queries contain:

* Technical terms
* Names
* Numbers
* Specific terminology
* Exact phrases

BM25 complements semantic search by capturing information that vector similarity may overlook.

---

# 🔀 Hybrid Retrieval

The Hybrid Retrieval module combines:

```text
Semantic Results
       +
BM25 Results
       ↓
Hybrid Results
```

This creates a more robust retrieval layer than relying on a single retrieval strategy.

---

# 🎯 Reranking

After retrieving candidate documents, OmniRAG applies a reranking stage.

```text
Query
  ↓
Initial Retrieval
  ↓
Candidate Documents
  ↓
Reranking
  ↓
Most Relevant Context
```

The reranking stage improves the quality of the final context sent to the LLM.

This reduces the amount of irrelevant information passed to the generation stage.

---

# 📄 PDF Processing

OmniRAG supports PDF document ingestion.

The processing pipeline includes:

```text
PDF
 ↓
Document Parsing
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
Vector Store
```

The system divides documents into smaller chunks before indexing them.

This allows the retrieval system to find specific pieces of information instead of processing the entire document at once.

---

# 🖼️ Multimodal Understanding

OmniRAG is designed to work with more than plain text.

The system can process visual information contained inside documents.

The multimodal pipeline includes:

```text
PDF
 ↓
Image Extraction
 ↓
OCR / Vision Analysis
 ↓
Extracted Information
 ↓
RAG Context
 ↓
Gemini
```

This makes the system more suitable for documents containing:

* Images
* Screenshots
* Scanned pages
* Diagrams
* Visual information

---

# 🔍 OCR

OmniRAG integrates OCR capabilities using **Tesseract OCR**.

OCR allows the system to extract text from images and scanned documents.

```text
Image
 ↓
Tesseract OCR
 ↓
Extracted Text
 ↓
Chunking
 ↓
Retrieval
```

This helps the RAG pipeline work with information that is not directly available as machine-readable text.

---

# 👁️ Vision Analysis

In addition to OCR, OmniRAG contains a vision processing layer for analyzing extracted document images.

This allows visual information to become part of the overall RAG pipeline.

---

# 🤖 Google Gemini

Google Gemini is used as the generation component of the RAG system.

The final pipeline is:

```text
User Query
     ↓
Retrieval
     ↓
Relevant Context
     ↓
Gemini
     ↓
Generated Answer
```

Instead of asking the LLM to answer only from its general knowledge, OmniRAG provides retrieved context from the user's documents.

---

# 🔄 LangGraph Workflow

LangGraph is used to orchestrate the RAG workflow.

The workflow manages different stages of the system and connects them together.

Conceptually:

```text
START
  ↓
Query Processing
  ↓
Retrieval
  ↓
Reranking
  ↓
Context Preparation
  ↓
LLM Generation
  ↓
END
```

This provides a structured workflow that can be extended with additional agents, tools, or decision-making nodes.

---

# 🖥️ Streamlit Interface

OmniRAG includes a Streamlit application that provides an interactive interface for interacting with the RAG system.

The interface allows users to work with documents and submit questions to the system.

Run the application using:

```bash
streamlit run app/streamlit_app.py
```

---

# 📁 Project Structure

```text
OmniRAG/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── sample.pdf
│   ├── sample1.pdf
│   └── test_page.png
│
├── demo/
│   └── end.mp4
│
├── src/
│   │
│   ├── cache/
│   │   └── cache_manager.py
│   │
│   ├── chunking/
│   │   └── chunker.py
│   │
│   ├── citations/
│   │
│   ├── embeddings/
│   │   └── embedder.py
│   │
│   ├── graph/
│   │   └── workflow.py
│   │
│   ├── ingestion/
│   │   ├── document_parser.py
│   │   ├── ocr.py
│   │   └── pdf_loader.py
│   │
│   ├── llm/
│   │   └── gemini.py
│   │
│   ├── retrieval/
│   │   ├── hybrid.py
│   │   ├── keyword.py
│   │   ├── query_parser.py
│   │   ├── reranker.py
│   │   ├── retrieval_pipeline.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   │
│   ├── vision/
│   │   └── image_analyzer.py
│   │
│   ├── config.py
│   └── rag.py
│
├── tests/
│   ├── test_chunker.py
│   ├── test_document_parser.py
│   ├── test_embeddings.py
│   ├── test_gemini.py
│   ├── test_graph.py
│   ├── test_hybrid.py
│   ├── test_image_extraction.py
│   ├── test_keyword.py
│   ├── test_multimodal_rag.py
│   ├── test_ocr.py
│   ├── test_pdf_loader.py
│   ├── test_query_parser.py
│   ├── test_rag.py
│   ├── test_reranker.py
│   ├── test_retrieval_pipeline.py
│   ├── test_retriever.py
│   ├── test_vision.py
│   └── test_vision_parser.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# ⚙️ Technologies

| Technology            | Purpose                     |
| --------------------- | --------------------------- |
| Python                | Core Programming Language   |
| LangChain             | LLM / RAG Components        |
| LangGraph             | Workflow Orchestration      |
| Google Gemini         | LLM Generation              |
| FAISS                 | Vector Search               |
| Sentence Transformers | Embeddings                  |
| BM25                  | Keyword Retrieval           |
| Tesseract OCR         | Text Extraction from Images |
| PyPDF                 | PDF Processing              |
| Streamlit             | User Interface              |

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/ahmedsaeed2005/OmniRAG.git
cd OmniRAG
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

⚠️ **Important:** Never upload your `.env` file or API key to GitHub.

The `.gitignore` file already excludes `.env`.

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

Then open the local Streamlit URL shown in the terminal.

---

# 🧪 Testing

The project contains multiple test modules covering the major components.

Examples:

```bash
python tests/test_chunker.py
```

```bash
python tests/test_embeddings.py
```

```bash
python tests/test_hybrid.py
```

```bash
python tests/test_rag.py
```

```bash
python tests/test_multimodal_rag.py
```

---

# 🎥 Demo

A short demonstration of the OmniRAG system is available here:

### ▶️ [Watch the OmniRAG Demo](demo/end.mp4)

The demo shows the system running through the Streamlit interface.

---

# 🔬 Retrieval Pipeline

The core retrieval pipeline can be summarized as:

```text
                    User Query
                        │
                        ▼
                 Query Processing
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
       Semantic Search          BM25
             │                     │
             └──────────┬──────────┘
                        ▼
                 Hybrid Retrieval
                        │
                        ▼
                    Reranker
                        │
                        ▼
               Relevant Context
                        │
                        ▼
                     Gemini
                        │
                        ▼
                  Final Answer
```

The main advantage of this architecture is that each retrieval technique solves a different problem.

**Semantic Search** focuses on meaning.

**BM25** focuses on exact lexical matches.

**Hybrid Retrieval** combines both.

**Reranking** improves the final ordering of retrieved results.

**Gemini** uses the resulting context to generate the final response.

---

# 💡 Why OmniRAG?

Traditional RAG:

```text
Query
 ↓
Vector Search
 ↓
LLM
```

OmniRAG:

```text
Query
 ↓
Query Processing
 ↓
Semantic Search + BM25
 ↓
Hybrid Retrieval
 ↓
Reranking
 ↓
OCR / Vision
 ↓
Context Preparation
 ↓
Gemini
 ↓
Answer
```

The goal is to improve retrieval quality, support multimodal documents, and create a more flexible RAG architecture.

---

# 🚀 Future Improvements

Planned improvements include:

* 🔹 Advanced query rewriting
* 🔹 Agentic RAG
* 🔹 Automatic retrieval evaluation
* 🔹 Better multimodal reasoning
* 🔹 More advanced reranking models
* 🔹 Conversation memory
* 🔹 Streaming responses
* 🔹 Multiple document formats
* 🔹 Production deployment
* 🔹 Observability and tracing
* 🔹 Retrieval evaluation metrics

---

# 🎓 Project Goals

OmniRAG was developed to explore and implement modern RAG architectures and understand the complete pipeline from document ingestion to final answer generation.

The project focuses on:

* Information Retrieval
* NLP
* LLMs
* Vector Databases
* Hybrid Search
* Multimodal AI
* OCR
* Reranking
* Agentic Workflows

---

# 👨‍💻 Author

**Ahmed Saeed**

Computer Science Student
Interested in:

* Artificial Intelligence
* Machine Learning
* NLP
* Computer Vision
* Generative AI
* RAG Systems

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub.

---

### 📌 Repository

https://github.com/ahmedsaeed2005/OmniRAG
