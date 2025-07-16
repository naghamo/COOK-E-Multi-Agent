from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from tokens.tokens_count import update_total_tokens
from dotenv import load_dotenv
import os

load_dotenv()

AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
DEPLOYMENT_NAME = "team10-gpt4o"
AZURE_OPENAI_ENDPOINT = "https://096290-oai.openai.azure.com"
API_VERSION = "2023-05-15"



chat = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version=API_VERSION,
    openai_api_type="azure",
    temperature=0,
)

context_template =  """
You are an expert cooking assistant.
"""

prompt_template = ChatPromptTemplate.from_template(context_template)

def generate_context(user_input):
    formatted_prompt = prompt_template.format(user_input=user_input)
    messages = [HumanMessage(content=formatted_prompt)]
    with get_openai_callback() as cb:
        response = chat(messages=messages)
        print("Prompt tokens:", cb.prompt_tokens)
        print("Completion tokens:", cb.completion_tokens)
        print("Total tokens (this run):", cb.total_tokens)
        update_total_tokens(cb.total_tokens, filename="../tokens/total_tokens_Nagham.txt")
    return response.content

if __name__ == "__main__":
    inp = "Vegan pizza for 5 under 40₪ with pickup, no mushrooms"
    print(generate_context(inp))