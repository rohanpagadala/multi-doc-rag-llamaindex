
# Multi-Document RAG with LlamaIndex + Gemini

A lightweight Retrieval-Augmented Generation (RAG) system that allows querying across multiple documents using LlamaIndex, Google Gemini (via GoogleGenAI), and HuggingFace embeddings.

# 🚀 Features
Multi-document ingestion from local folder (docs/)
Vector search using BAAI/bge-small-en-v1.5 embeddings
LLM-powered answering using Gemini 2.5 Flash
Simple Python interface (no UI layer required)
Fast semantic retrieval with configurable top-k results

# 🧠 Architecture
Documents → Chunking → Embeddings → Vector Index → Retrieval → Gemini LLM → Answer

# 📦 Tech Stack
LlamaIndex
Google Generative AI (Gemini)
HuggingFace Transformers (Embeddings)
Python Dotenv

⚙️ Installation
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
# 🔑 Environment Variables

Create a .env file:

GOOGLE_API_KEY=your_google_api_key_here
📂 Add Documents
Place your files inside the docs/ folder.
Supported formats (via LlamaIndex SimpleDirectoryReader):
PDF
TXT
DOCX
MD

# ▶️ Usage
Run your Python script:
python app.py

Then use:
from app import ask_question

print(ask_question("What is this document about?"))

# 🧪 Example
ask_question("Summarize all documents related to the system design notes")
# 🔧 Key Configurations

Inside app.py:

Settings.chunk_size = 512
Settings.chunk_overlap = 50
similarity_top_k = 4

# You can tune these for:

1)better context retention
2)faster retrieval
3)improved accuracy
