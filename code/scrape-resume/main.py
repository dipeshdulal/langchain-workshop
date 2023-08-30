from langchain.document_loaders import PyPDFLoader

# document source
loader = PyPDFLoader("scrape-resume/main.pdf")
docs = loader.load()

# llm
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
prompt = PromptTemplate.from_template("Given resume here; {resume}. Assume you are a skilled hr manager. Tell be why we need to hire this person. What are their strengths and weaknesses?")
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run({"resume": docs[0].page_content}))