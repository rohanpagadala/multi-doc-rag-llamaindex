from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)

from llama_index.llms.google_genai import GoogleGenAI

from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding
)

load_dotenv()

Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash"
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu"
)

Settings.chunk_size = 512
Settings.chunk_overlap = 50

documents = SimpleDirectoryReader(
    "docs"
).load_data()

index = VectorStoreIndex.from_documents(
    documents
)

query_engine = index.as_query_engine(
    similarity_top_k=4
)

def ask_question(query):
    response = query_engine.query(query)
    return str(response)