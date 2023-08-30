import requests 

# Get the webpage
page = requests.get("https://wesionary.team/career")
data = page.text

with open("data.html", "w", encoding="utf-8") as f:
    f.write(data)

# Build document from data
from langchain.document_loaders import BSHTMLLoader

loader = BSHTMLLoader("data.html")
document = loader.load()

# Search data from this document 
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
prompt = PromptTemplate.from_template("""
        Given data from careers page on the wesionaryTeam here; {data}. 
        Give me list of team members and their designation?
    """)
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run({"data": document[0].page_content[1000:3000]}))