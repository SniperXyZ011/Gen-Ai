from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="ibm-granite/granite-4.2-30b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

try:
    response = model.invoke("Write a poem about the sea.")
    print(response)
except Exception as error:
    print(f"Request failed: {type(error).__name__}: {error}")

