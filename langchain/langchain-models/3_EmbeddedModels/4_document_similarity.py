from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=r"D:\Gen-ai\hf_cache",
    model_kwargs={"device": "cpu"},
)

documents = [
    "Virat Kohli is a famous cricketer known for his aggressive batting style and leadership skills.",
    "MS Dhoni is a legendary cricketer and former captain of the Indian cricket team, known for his calm demeanor and finishing abilities.",
    "Rohit Sharma is a talented cricketer and the current captain of the Indian cricket team, known for his elegant batting technique and ability to score big centuries.",
    "Sachin Tendulkar is a cricketing legend and one of the greatest batsmen of all time, known for his impeccable technique and numerous records in international cricket.",
    "Jasprit Bumrah is a skilled fast bowler and a key player in the Indian cricket team, known for his unique bowling action and ability to take crucial wickets in high-pressure situations."
]

query = "Tell me about Sachin Tendulkar's cricketing achievements and records."

document_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], document_embeddings)[0]

idx, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(f"Most similar document: {documents[idx]}")
print(f"Similarity score: {score}")