from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature=0.7, max_completion_tokens=1000) 
"temperature is a parameter that controls the randomness of the model's output. A higher temperature (e.g., 1.0) will result in more random outputs, while a lower temperature (e.g., 0.2) will make the output more focused and deterministic."
result = model.invoke("Write a poem about the sea.")   
print(result.content)