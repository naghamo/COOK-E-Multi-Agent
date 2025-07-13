from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from tokens_count import update_total_tokens
from dotenv import load_dotenv
import os

load_dotenv()

AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
DEPLOYMENT_NAME = os.environ["DEPLOYMENT_NAME"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
API_VERSION = os.environ["API_VERSION"]


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
Given a user's free-text request, extract as much information as possible into a single valid JSON object, using the following fields:

- food_name: string. (Required. If not found, set to null and include a short error.)
- people: integer. (Default to 1 if not specified.)
- delivery: string. ("delivery", "pickup", or null; default to "delivery" if not specified.)
- special_requests: string or null.
- budget: string or null.
- raw_text: the original user input.
- extra_fields: dictionary for any other constraints or preferences (e.g., allergies, brands, supermarkets, dietary restrictions, tools, timing, etc.), using snake_case for keys.
- error: string, only if the input is ambiguous, not a food request, or if food_name is missing.

Instructions:
- If a field is missing, set to null (unless a default is specified).
- If food_name is missing, set error and do not proceed further.
- If the input is not a food request or is ambiguous, set error with a short message.
- Output ONLY valid JSON with the fields above. No extra text.

User Request: {user_input}
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
        update_total_tokens(cb.total_tokens, filename="../data/total_tokens_Nagham.txt")
    return response.content

if __name__ == "__main__":
    inp = "Vegan pizza for 5 under 40₪ with pickup, no mushrooms"
    print(generate_context(inp))
