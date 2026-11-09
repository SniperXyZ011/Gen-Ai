from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

documents = [
    "This is the first document for embedding.",
    "This is the second document for embedding.",
    "This is the third document for embedding."
]

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimension=32, chunk_size=1000, max_retries=3, request_timeout=30)

try:
    embedding_vectors = embedding.embed_documents(documents)
    print("Embedding vectors:", embedding_vectors)
except Exception as error:
    print(f"Embedding request failed: {type(error).__name__}: {error}")
    