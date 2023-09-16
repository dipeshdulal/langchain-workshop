from langchain import PromptTemplate, LLMChain
from langchain.chains import SequentialChain
from langchain import OpenAI
import os


llm = OpenAI(openai_api_key=os.environ['OPENAI_API_KEY'])


first_prompt = PromptTemplate.from_template("""
    Give me 5 of the {things}.                                        
""")

final_prompt = PromptTemplate.from_template("""
    Give me description of the items given. 
    {items}
""")

chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="items", verbose=True)
chain_two = LLMChain(llm=llm, prompt=final_prompt, verbose=True)

combined_chain = SequentialChain(
    chains=[chain_one, chain_two],
    input_variables=["things"],
    verbose=True
)

print(combined_chain.run("programming languages"))