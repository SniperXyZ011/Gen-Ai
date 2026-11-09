from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = model.invoke(user_input)
    print("AI: ", response.content[0]['text'])