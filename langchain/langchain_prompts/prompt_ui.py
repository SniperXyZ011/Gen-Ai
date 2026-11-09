from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts.loading import load_prompt
# from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv  
import streamlit as st

load_dotenv()

model = ChatGoogleGenerativeAI( model="gemini-3.5-flash-lite",
    timeout=30,
    max_retries=0,
    temperature=0)

st.header("Research tool")

# user_input = st.text_input("Enter your prompt: ")
paper_input = st.selectbox("Select a paper to summarize:", ["Select...", "Attention is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis", "AlphaFold: Using AI for scientific discovery"])

style_input = st.selectbox("Select a style for the summary:", ["Select...", "Concise", "Detailed", "Technical", "Layman"])

length_input = st.selectbox("Select a length for the summary:", ["Select...", "Short (1-2 paragraphs)", "Medium (3-4 paragraphs)", "Long (5+ paragraphs)"])

#prompt template
# template = PromptTemplate(
#     template="""
# Please summarize the following research paper titled "{paper_input}" with the following specifications:
# Explaination Style: {style_input}
# Explaination Length: {length_input}
# 1. Mathematical Details:
#     - Include relevant equations, formulas if present in the paper.
#     - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
# 2. Analogies and Examples:
#     - Provide analogies to explain complex concepts in a more relatable manner.
#     - Include real-world examples to illustrate the applications of the research findings.

# If a certain information is not present in the paper, please respond with "Insufficient information available". Ensure that the summary is accurate and faithful to the content of the paper.

# """
# ,
# input_variables=["paper_input", "style_input", "length_input"]
# )

template = load_prompt("template.json")

#fill placeholders
prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

if st.button("Summarize"):
    response = model.invoke(prompt)
    st.write(response.content[0]['text'])