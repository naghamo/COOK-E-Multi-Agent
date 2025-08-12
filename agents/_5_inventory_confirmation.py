"""
Agent 5: Inventory Confirmation
-------------------------------
This agent receives:
- The recipe ingredient list
- The matched inventory (as from matcher agent)
- The structured user input
- Today's date

It returns, for each recipe ingredient, whether and why it needs to be bought, as a JSON list with explanations.
"""
import json
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from langchain_core._api import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
import pandas as pd
from langchain_community.callbacks.manager import get_openai_callback

from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from datetime import datetime
from dotenv import load_dotenv
import os
from tokens.tokens_count import update_total_tokens
from pydantic import SecretStr

# --- ENV SETUP ---

# Load API keys and deployment settings from .env
load_dotenv()
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
DEPLOYMENT_NAME = "team10-gpt4o"
AZURE_OPENAI_ENDPOINT = "https://096290-oai.openai.azure.com"
API_VERSION = "2023-05-15"

# Initialize Azure OpenAI LLM for confirmation
chat = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT_NAME,
    api_key=SecretStr(AZURE_OPENAI_API_KEY),
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=API_VERSION,
    openai_api_type="azure",
    temperature=0,
)

# --- PROMPT ---

confirmation_prompt = """
Given:
- df_recipe: list of ingredients with quantity and unit.
- df_inventory: list of home items with quantity, unit, and expiry.
- parsed_user_input: includes anything the user said they already have.
- today: current date (in YYYY-MM-DD format).

Rules:
- When comparing expiry dates, always compare the **full date** (year, month, and day) to today.
- If ingredient is missing, expired (expiry date is **on or before** today's full date), expiring soon (within 3 days from today's full date), or has insufficient quantity → mark for purchase.
- If expiring soon but available in quantity, still buy full amount.
- Match ingredients loosely (e.g., "onion" ≈ "red onion").
- **Always use the unit as given in the recipe for all output and for all comparisons.**
- If the inventory for a matched ingredient is in a different unit, **convert or scale the inventory quantity to the recipe unit before comparing**.
    - Example: If recipe asks for "100 grams" but inventory has "0.1 kilograms", convert 0.1 kg = 100 grams, then compare to the recipe's required 100 grams.
    - Another example: If the recipe asks for "2 cups" and inventory has "500 milliliters", convert 500 ml ≈ 2.11 cups (1 cup ≈ 236.6 ml).
    - If a conversion is not possible, assume the inventory quantity cannot be used and treat as missing.
- **If the recipe quantity is null or not given, treat it as 1 (one unit of that ingredient). Only check if the ingredient exists and is not expired or expiring soon.**
- Output all quantities using the same unit as the recipe (requested_unit).
- For **universal/common household items** (e.g., water, tap water, salt, pepper, oil), if the recipe requires them and the inventory does not explicitly contain them, **assume they are available at home unless the user states otherwise.** In such cases, explain: "common household item, assumed available."

Return a JSON list of dictionaries with:
- name: ingredient name (from recipe)
- requested_quantity: quantity from recipe (use 1 if not given)
- requested_unit: unit from recipe (use "unit" if not present)
- to_buy_min: amount to buy (0 if not needed)
- to_buy_unit: same as requested_unit
- explanation: reason for purchase (e.g., "expired", "expiring in 2 days", "not enough after converting units", "enough at home", "common household item, assumed available")
- exp: write if you checked the expiry date, how many days remain, and if not applicable, state which date you checked.

**Only return the JSON list. Nothing else.**

df_recipe:
{df_recipe}

df_inventory:
{df_inventory}

parsed_user_input:
{parsed_user_input}

today: {today}
"""



# Create a ChatPromptTemplate from the confirmation prompt
confirmation_template = ChatPromptTemplate.from_template(confirmation_prompt)
# parse the JSON part out
def extract_json_list_from_codeblock(codeblock_str):
    """Extracts a JSON list from a code block string."""
    match = re.search(r'```json\s*(\[[\s\S]*?\])\s*```', codeblock_str)
    if not match:
        raise ValueError("No JSON list found in code block!")
    json_str = match.group(1)
    return json.loads(json_str)
def run_confirmation_agent(df_recipe, matched_inventory, parsed_user_input,tokens_filename="../tokens/total_tokens.txt"):
    """
    For each recipe ingredient, determines if it needs to be bought (based on home inventory & rules).
    Args:
        df_recipe (DataFrame): Recipe ingredient list.
        matched_inventory (list or DataFrame): Inventory items relevant for the recipe.
        parsed_user_input (dict): Parsed/structured user input.

    Returns:
        LLM-generated JSON string of the decision list.
    """
    # today's date for expiry checks
    today = datetime.today().strftime('%Y-%m-%d')
    if isinstance(df_recipe, list):
        df_recipe = pd.DataFrame(df_recipe)
    if isinstance(matched_inventory, list):
        df_inventory_for_prompt = pd.DataFrame(matched_inventory).to_string(index=False)
    else:
        df_inventory_for_prompt = matched_inventory.to_string(index=False)
    prompt = confirmation_template.format_messages(
        df_recipe=df_recipe.to_string(index=False),
        df_inventory=df_inventory_for_prompt,
        parsed_user_input=parsed_user_input,
        today=today
    )
    with get_openai_callback() as cb:
        response = chat(messages=prompt)
    update_total_tokens(cb.total_tokens, filename=tokens_filename)

    # Parse LLM markdown-string to list
    try:
        return extract_json_list_from_codeblock(response.content)
    except Exception as e:
        print("Error parsing confirmation agent output:", e)
        return []

# Example usage for debugging and development
if __name__ == "__main__":
    parsed_user_input = {
        "dish": "Pasta with tomato sauce",
        "servings": 2,
        "inventory_mentions": ["olive oil"],
        "budget": "under 50 NIS",
        "delivery_mode": "pickup"
    }

    df_recipe = pd.DataFrame([
        {"name": "pasta", "quantity": 200, "unit": "grams"},
        {"name": "tomato sauce", "quantity": 150, "unit": "ml"},
        {"name": "garlic", "quantity": 2, "unit": "units"},
        {"name": "olive oil", "quantity": 50, "unit": "ml"},
        {"name": "salt", "quantity": 1, "unit": "teaspoons"}
    ])

    # Simulated result from matcher agent:
    matched_inventory = [
        {"name": "olive oil", "quantity": 30, "unit": "ml", "expiry": "2025-08-04"},
        {"name": "garlic", "quantity": 1, "unit": "units", "expiry": "2025-07-17"},
        {"name": "salt", "quantity": 5, "unit": "teaspoons", "expiry": "2026-01-01"}
    ]

    result = run_confirmation_agent(df_recipe, matched_inventory, parsed_user_input)
    print(result)
