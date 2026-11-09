from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ["HF_HOME"] = "D/Gen-ai/hf-cache"
# when we run this code, it will first download the model from Hugging Face and cache it in the specified directory. If the model is already cached, it will load it from there instead of downloading it again. And then it will use the model to generate a response to the prompt "Write a poem about the sea." It will be very hardware intensive, so make sure you have a good GPU and enough memory to run it. If you don't have a GPU, you can try using a smaller model or running it on a cloud service that provides GPU access.

llm = HuggingFacePipeline.from_model_id(
    model_id="ibm-granite/granite-4.2-30b",
    task="text-generation",
    pipeline_kwargs={"max_length": 1000, "temperature": 0.7}
)

model = ChatHuggingFace(llm=llm)

try:
    response = model.invoke("Write a poem about the sea.")
    print(response)
except Exception as error:
    print(f"Request failed: {type(error).__name__}: {error}")