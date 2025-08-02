"""
Agent 7: Inventory Filter
------------------------
This agent receives:
- The ingredient confirmation list from the previous agent (as a JSON list of dicts, or DataFrame)
- The user's confirmation/approval (as an updated/edited JSON list or confirmation letter)
It outputs only the ingredients still needed to buy, with quantity and unit.
Logs LLM token usage for audit.
"""

import pandas as pd
import json
from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from tokens.tokens_count import update_total_tokens
from dotenv import load_dotenv
import os

# --- ENV SETUP ---

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

filter_prompt = """
You are an assistant that receives:
- first_conf_req: a JSON list of dictionaries representing all ingredients and quantities the previous agent decided were needed to buy (fields: name, requested_quantity, requested_unit, to_buy_min, to_buy_unit, explanation)
- user_confirmation: a JSON list or string that includes the user’s edits or confirmations (they may change the quantity, set some to zero, or remove some items)

Return a JSON list of only those ingredients where to_buy_min > 0 after taking into account the user’s edits, for each giving:
- name
- to_buy_min
- to_buy_unit

Only output the JSON list. No explanations, no extra text.

first_conf_req:
{first_conf_req}

user_confirmation:
{user_confirmation}
"""

filter_template = ChatPromptTemplate.from_template(filter_prompt)

def run_inventory_filter_agent(first_conf_req, user_confirmation):
    """
    Filters the list of ingredients to buy based on user confirmation.
    Args:
        first_conf_req (list or DataFrame): Confirmation list of all potential buys.
        user_confirmation (list or string): User’s edited confirmation (usually a JSON list).
        tokens_log_file (str): Where to log token counts.
    Returns:
        List of dicts (name, to_buy_min, to_buy_unit) for only those actually to buy.
    """
    # Convert input to string for prompt
    if isinstance(first_conf_req, list):
        conf_str = json.dumps(first_conf_req, ensure_ascii=False)
    elif isinstance(first_conf_req, pd.DataFrame):
        conf_str = first_conf_req.to_json(orient="records", force_ascii=False)
    else:
        conf_str = str(first_conf_req)
    # User confirmation to string
    if isinstance(user_confirmation, (list, dict)):
        user_conf_str = json.dumps(user_confirmation, ensure_ascii=False)
    else:
        user_conf_str = str(user_confirmation)

    prompt = filter_template.format_messages(
        first_conf_req=conf_str,
        user_confirmation=user_conf_str
    )
    with get_openai_callback() as cb:
        response = chat(messages=prompt)
    # Token counting
    update_total_tokens(cb.total_tokens, filename="../tokens/total_tokens_Nagham.txt")
    # Parse LLM output
    raw_content = response.content.strip()
    # Remove markdown fences if present
    import re
    match = re.search(r"```json\s*(.*?)\s*```", raw_content, re.DOTALL | re.IGNORECASE)
    json_str = match.group(1).strip() if match else raw_content
    try:
        buy_list = json.loads(json_str)
    except Exception as e:
        print("Error parsing LLM output:", e)
        print("LLM output was:", raw_content)
        buy_list = []
    return buy_list

# Example usage
if __name__ == "__main__":
    # Example output from previous agent:
    first_conf_req = [
        {"name": "olive oil", "requested_quantity": 50, "requested_unit": "ml", "to_buy_min": 50, "to_buy_unit": "ml", "explanation": "not at home"},
        {"name": "garlic", "requested_quantity": 2, "requested_unit": "units", "to_buy_min": 0, "to_buy_unit": "units", "explanation": "enough at home"},
        {"name": "pasta", "requested_quantity": 200, "requested_unit": "grams", "to_buy_min": 0, "to_buy_unit": "grams", "explanation": "enough at home"},
        {"name": "salt", "requested_quantity": 1, "requested_unit": "teaspoons", "to_buy_min": 1, "to_buy_unit": "teaspoons", "explanation": "need a bit more"},
    ]
    # Example: user deletes garlic, sets salt to 0, but confirms olive oil as is:
    user_confirmation = [
        {"name": "olive oil", "to_buy_min": 50, "to_buy_unit": "ml"},
        {"name": "garlic", "to_buy_min": 0, "to_buy_unit": "units"},
        {"name": "pasta", "to_buy_min": 0, "to_buy_unit": "grams"},
        {"name": "salt", "to_buy_min": 0, "to_buy_unit": "teaspoons"}
    ]

    buy_list = run_inventory_filter_agent(first_conf_req, user_confirmation)
    print("Ingredients still needed to buy:", buy_list)
