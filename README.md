# 📄 Hybrid RAG Document Assistant

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-orange)
![Groq](https://img.shields.io/badge/LLM-Groq-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

An **AI-powered document question-answering system** that allows users to upload a PDF and ask questions about its content using a **Hybrid Retrieval-Augmented Generation (RAG)** architecture.

The system combines **semantic vector search (FAISS)** and **keyword search (BM25)** to retrieve relevant information and uses **Llama-3.1 via Groq API** to generate accurate answers.

---

# 🚀 Live Demo

🌐 **Web Application**

https://hybrid-rag-assistant-shivam.streamlit.app

📦 **GitHub Repository**

https://github.com/shivam-9s/hybrid-rag-assistant

---

# 🧠 Project Motivation

Large Language Models often **hallucinate** when answering questions without reliable context.

This project solves the problem using **Retrieval-Augmented Generation (RAG)**.

Instead of relying only on the model’s internal knowledge, the system:

1. Retrieves relevant document sections
2. Provides that context to the LLM
3. Generates accurate answers grounded in the document

This approach significantly **improves reliability and accuracy**.

---

# 🏗 System Architecture

```
User
 │
 ▼
Streamlit Web App
 │
 ▼
PDF Upload
 │
 ▼
Document Loader (PyPDFLoader)
 │
 ▼
Text Chunking
 │
 ▼
Embeddings Generation
(Sentence Transformers)
 │
 ▼
Hybrid Retrieval
 ├── FAISS (Vector Search)
 └── BM25 (Keyword Search)
 │
 ▼
Context Retrieval
 │
 ▼
LLM (Llama-3.1 via Groq)
 │
 ▼
Generated Answer
 │
 ▼
Displayed to User
```

---

# ⚙️ Key Features

✅ Upload any PDF document
✅ Hybrid Retrieval (Vector + Keyword Search)
✅ Fast responses using Groq LLM API
✅ Accurate context-based answers
✅ Interactive chat interface
✅ Real-time document processing
✅ Web deployment with Streamlit

---

# 📊 Example Workflow

### 1️⃣ Upload PDF

Users upload a document such as lecture notes, research papers, or reports.

### 2️⃣ Document Processing

The system splits the document into smaller chunks.

### 3️⃣ Embedding Creation

Each chunk is converted into vector embeddings using **MiniLM**.

### 4️⃣ Hybrid Retrieval

Relevant chunks are retrieved using:

• FAISS similarity search
• BM25 keyword matching

### 5️⃣ LLM Generation

The retrieved context is sent to **Llama-3.1**, which generates an answer.

---

# 🧩 Tech Stack

### Frontend

* Streamlit

### Backend

* Python

### AI / NLP

* LangChain
* Sentence Transformers
* FAISS Vector Database
* BM25 Retrieval

### LLM

* Groq API
* Llama-3.1-8B-Instant

### Deployment

* Streamlit Cloud
* GitHub

---

# 📚 Libraries Used

```
streamlit
langchain
langchain-community
langchain-core
langchain-groq
sentence-transformers
faiss-cpu
pypdf
rank-bm25
```

---

# 📂 Project Structure

```
hybrid-rag-assistant
│
├── streamlit_app.py        # Streamlit UI
├── rag_chain.py            # RAG pipeline
├── retriever.py            # Hybrid retrieval
├── ingest.py               # Document ingestion
├── app.py                  # CLI version
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
```

---

# ⚡ Installation (Run Locally)

### Clone the repository

```bash
git clone https://github.com/shivam-9s/hybrid-rag-assistant.git
cd hybrid-rag-assistant
```

---

### Create virtual environment

```
python -m venv venv
```

---

### Activate environment

Windows

```
venv\Scripts\activate
```

Mac/Linux

```
source venv/bin/activate
```

---

### Install dependencies

```
pip install -r requirements.txt
```

---

### Set Groq API Key

Windows

```
$env:GROQ_API_KEY="your_api_key"
```

Mac/Linux

```
export GROQ_API_KEY="your_api_key"
```

---

### Run the application

```
streamlit run streamlit_app.py
```

Open in browser:

```
http://localhost:8501
```

---

# 🔐 Environment Variables

```
GROQ_API_KEY=your_groq_api_key
```

---

# 📈 Future Improvements

• Multi-document support
• Chat memory and conversation history
• Document citation highlighting
• Source reference links
• Vector database persistence
• Docker containerization
• Authentication system

---

# 📸 Application Preview

Upload PDF → Ask Question → Get AI Answer

Example:

```
User: What is Machine Learning?

AI: Machine learning is a method where computers learn patterns from data and use those patterns to make predictions without being explicitly programmed.
```

---

# 👨‍💻 Author

**Shivam Kumar**

GitHub
https://github.com/shivam-9s

---

# ⭐ Support

If you found this project useful, please **star the repository** ⭐
