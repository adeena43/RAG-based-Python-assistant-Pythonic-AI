import os
import google.generativeai as genai
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from google import genai 

load_dotenv()

# Set your free Gemini API key from .env
# api_key = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=api_key)

# Load text
url = 'https://en.wikipedia.org/wiki/Giant_panda'
loader = WebBaseLoader(url)
docs = loader.load()
print(f"Loaded {len(docs)} documents")
print(docs[0].page_content)


text_to_summarize = docs[0].page_content
prompt_template = 'Write the definition of panda in 10 words only:\n{text}'
prompt = prompt_template.format(text=text_to_summarize)

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
)

parser = StrOutputParser()
summary = parser.parse(response.text)
print(summary)
