from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()

model = ChatAnthropic(model="claude-v1", temperature=0.7, max_completion_tokens=1000)
response = model.invoke("Write a poem about the sea.")
print(response.content)