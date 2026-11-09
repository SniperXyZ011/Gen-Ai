from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=r"D:\Gen-ai\hf_cache",
    model_kwargs={"device": "cpu"},
)

try:
    text = "Hello I am Shyam."
    embedding_vector = embedding.embed_query(text)
    print("Embedding dimensions:", len(embedding_vector))
    print("Embedding vector:", embedding_vector)
except Exception as error:
    print(f"Embedding request failed: {type(error).__name__}: {error}")