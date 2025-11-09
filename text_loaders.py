import os
import google.generativeai as genai
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from google import genai 

load_dotenv()

# Set your free Gemini API key from .env
# api_key = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=api_key)

# Load text
loader = TextLoader(r"pandas.txt", encoding="utf-8")
docs = loader.load()
print(f"Loaded {len(docs)} documents")

# Prepare prompt
text_to_summarize = docs[0].page_content
prompt_template = 'Write a summary of the text:\n{text}'
prompt = prompt_template.format(text=text_to_summarize)

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
)
# Parse and print result
parser = StrOutputParser()
summary = parser.parse(response.text)
print(summary)
