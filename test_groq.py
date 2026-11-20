
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import SecretStr

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is required.")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=SecretStr(api_key),
    temperature=0
)

response = llm.invoke("Reply with exactly: Groq API working")

print(response.content)
