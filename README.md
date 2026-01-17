# BuildRAG

**BuildRAG** is a local, production-ready **Retrieval-Augmented Generation (RAG)** workspace built with **Streamlit**, **Ollama**, and **Chroma**.

It lets you upload documents, index them locally, and chat with an LLM that answers questions **grounded in your own data**, with sources.

## ✨ Features
- 📂 Upload & manage PDF / TXT / MD documents
- 🧠 Local vector database with Chroma
- 🔍 Multiple retrieval modes (similarity, MMR, score threshold)
- 💬 Chat interface with source citations
- 🛠️ Debug mode to inspect retrieved chunks
- 🔒 Fully local (Ollama-powered LLM & embeddings)

## 🏗️ Tech Stack
- **UI:** Streamlit  
- **LLM:** Ollama (e.g. llama3.1)  
- **Embeddings:** nomic-embed-text  
- **Vector DB:** Chroma  

## 🚀 Run
```bash
ollama serve
streamlit run app.py