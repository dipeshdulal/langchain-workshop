import requests 

# Get the webpage
page = requests.get("https://en.wesionary.team/career")
data = page.text

with open("data.html", "w") as f:
    f.write(data)

# load splitters and load, split document 
from langchain.document_loaders import BSHTMLLoader
from langchain.text_splitter import CharacterTextSplitter

loader = BSHTMLLoader("data.html")
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=10)
docs = loader.load_and_split(splitter)
print(len(docs))

# embed one document and check result 
from langchain.embeddings.openai import OpenAIEmbeddings
import os
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
# print(embeddings.embed_documents([docs[0].page_content[0:100]]))

# vector stores 
from langchain.vectorstores import Chroma

db = Chroma.from_documents(docs, embeddings)
query = "analyze and return team members name and designation"
docs = db.similarity_search(query, k=3)

# query from returned documents. 
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
prompt = PromptTemplate.from_template("""Given documents; 
    {document}
                                      
    You need to analyze and return team members name and designation.
    Do not try to make up answer on your own. If you don't know it, just say "I don't know".
    Answer:""")
chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
print(chain.run({"document": [d.page_content for d in docs]}))