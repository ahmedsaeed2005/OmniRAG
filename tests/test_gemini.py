from src.llm.gemini import create_llm


llm = create_llm()

response = llm.invoke(
    "What is Big Data? Answer in one short sentence."
)

print(response.content)