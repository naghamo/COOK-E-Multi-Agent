"""Agent 1: LLM Context Parser
--------------------------------------
This agent parses user input to extract structured context for further processing in the pipeline."""
import json
import re
import warnings
import warnings
import importlib

from langchain_community.callbacks.manager import get_openai_callback

# from langchain_community.chat_models import AzureChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from pydantic import SecretStr

from tokens.tokens_count import update_total_tokens
from dotenv import load_dotenv
import os
# load environment variables
load_dotenv()

AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
DEPLOYMENT_NAME = "team10-gpt4o"
AZURE_OPENAI_ENDPOINT = "https://096290-oai.openai.azure.com"
API_VERSION = "2023-05-15"

# Initialize the AzureChatOpenAI model
chat = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT_NAME,
    api_key=SecretStr(AZURE_OPENAI_API_KEY),
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=API_VERSION,
    openai_api_type="azure",
    temperature=0,
)
# Define the context template for the LLM
context_template =  """
You are an expert cooking assistant.
Given a user's free-text request, extract as much information as possible into a single valid JSON object, using the following fields:

- food_name: string. (Required. If not found, set to null and include a short error.)
- people: integer. (Default to 1 if not specified.)
- delivery: string. ("delivery", "pickup", or null; default to "delivery" if not specified.)
- special_requests: string or null.
- budget: int or null.
- raw_text: the original user input.
- extra_fields: dictionary for any other constraints or preferences (e.g., allergies, brands, supermarkets, dietary restrictions, tools, timing, etc.), using snake_case for keys.
- error: string, only if the input is ambiguous, not a food request, or if food_name is missing.

Instructions:
- If a field is missing, set to null (unless a default is specified).
- If food_name is missing, set error and do not proceed further.
- If the input is not a food request or is ambiguous, set error with a short message.
- Output ONLY valid JSON with the fields above. No extra text.
- Spelling-correction rule: If the user's food name (or a clearly intended dish/ingredient) is misspelled or non-standard, set food_name to the corrected, standardized name. Save the original under extra_fields.original_food_name. For other corrected terms (e.g., ingredients, brands), include a mapping under extra_fields.corrected_terms.
User Request: {user_input}
"""

# Create a ChatPromptTemplate from the context template
prompt_template = ChatPromptTemplate.from_template(context_template)
def extract_json_from_llm(text):
    """Extracts JSON content from the LLM response text."""
    text = text.strip()
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text
# Function to generate context from user input
def parse_context(user_input, tokens_filename="../tokens/total_tokens.txt"):
    """Parses user input to extract structured context using the LLM."""
    formatted_prompt = prompt_template.format(user_input=user_input)
    messages = [HumanMessage(content=formatted_prompt)]
    with get_openai_callback() as cb:
        response = chat(messages=messages)

        update_total_tokens(cb.total_tokens, filename=tokens_filename)# Log the total tokens used
    # Extract JSON from the response content
    json_content = extract_json_from_llm(response.content)
    # Parse the JSON content
    try:
        parsed_content = json.loads(json_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from response: {e}")

    return parsed_content


# if __name__ == "__main__":
#     # Example user input
#     inp = "play"
#     print(parse_context(inp))
