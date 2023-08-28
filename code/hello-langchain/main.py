from langchain.llms import OpenAI
import os

llm = OpenAI(openai_api_key=os.environ['OPENAI_API_KEY'])

text = "Do you understand what a hello world program mean for a programmer?"
print(llm.predict(text=text))

# Prompt templates
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("What is a good software development compmany name that works on {language}?")
print(prompt.format(language="Python"))

# output parser
from langchain.schema import BaseOutputParser

class CommaSeparatedListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(", ")

# combine prompt, output and model into a single model, LLMChain
from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

template = """You are a helpful assistant who generates comma separated list of company names for a given industry."""
system_prompt = SystemMessagePromptTemplate.from_template(template)

human_prompt_template = """What is a good software development company name that works on {language}?"""
human_prompt = HumanMessagePromptTemplate.from_template(human_prompt_template)
chat_prompt = ChatPromptTemplate.from_messages([template, human_prompt])

chain = LLMChain(
    llm=llm,
    prompt=chat_prompt,
    output_parser=CommaSeparatedListOutputParser()
)
print(chain.run("golang"))