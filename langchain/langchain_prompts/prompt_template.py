#prompt template
from langchain_core.prompts import PromptTemplate


template = PromptTemplate(
    template="""
Please summarize the following research paper titled "{paper_input}" with the following specifications:
Explaination Style: {style_input}
Explaination Length: {length_input}
1. Mathematical Details:
    - Include relevant equations, formulas if present in the paper.
    - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
2. Analogies and Examples:
    - Provide analogies to explain complex concepts in a more relatable manner.
    - Include real-world examples to illustrate the applications of the research findings.

If a certain information is not present in the paper, please respond with "Insufficient information available". Ensure that the summary is accurate and faithful to the content of the paper.

"""
,
input_variables=["paper_input", "style_input", "length_input"]
)


template.save("template.json")