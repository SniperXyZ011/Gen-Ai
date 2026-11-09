from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

print("Sending request...", flush=True)

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    timeout=30,
    max_retries=0,
)

try:
    response = model.invoke("What is the capital of France?")
    print(response.content[0]['text'])
except Exception as error:
    print(f"Request failed: {type(error).__name__}: {error}")
