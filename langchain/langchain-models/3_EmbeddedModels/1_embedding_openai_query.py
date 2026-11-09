from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimension=32, chunk_size=1000, max_retries=3, request_timeout=30)
embedding_text = "This is a test text for embedding."
try:
    embedding_vector = embedding.embed_query(embedding_text)
    print("Embedding vector:", embedding_vector)
except Exception as error:
    print(f"Embedding request failed: {type(error).__name__}: {error}")
    