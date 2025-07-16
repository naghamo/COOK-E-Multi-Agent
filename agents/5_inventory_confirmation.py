import pandas as pd
from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from tokens.tokens_count import update_total_tokens
from dotenv import load_dotenv
import os
from datetime import datetime
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
from datetime import date
#
# context_template = """
# You are a helpful assistant that determines what ingredients the user needs to buy for a given recipe.
#
# You are provided with:
# - `df_recipe`: a DataFrame containing the ingredients and quantities required for the recipe.
# - `df_inventory`: a DataFrame containing ingredients currently at the user's home, with quantity, unit, and expiry date.
# - `parsed_user_input`: a dictionary of structured information extracted from the user's request (e.g., ingredients they say they have, preferences, serving size, etc.).
#
# Your task:
# 1. For each ingredient in `df_recipe`, check whether it is available at home using `df_inventory` and `parsed_user_input`.
#
# 2. An item must be purchased if any of the following apply:
#    - It is missing from the inventory entirely,
#    - It is expired (expiry date on or before today),
#    - It is expiring soon (within 3 days from today),
#    - It is present but the available quantity is not enough (after unit conversion).
#
# 3. If an item exists in the inventory but is **expiring soon**, it must be treated as unusable — even if there is enough of it — and the full quantity must be added to the purchase list. Mention `"expiring in X days"` in the explanation.
#
# 4. If an item is available in sufficient, unexpired quantity, include it with `to_buy_min = 0` and an explanation like `"already at home"` or `"enough in stock"`.
#
# You must:
# - Compare expiry dates in YYYY-MM-DD format against today’s date.
# - Convert units where needed (1 kg = 1000 grams, 1 l = 1000 ml).
# - Use semantic matching for ingredient names (e.g., “red onion” ≈ “onion”).
# - Always use the same unit from the recipe for both `requested_unit` and `to_buy_unit`.
#
# Your output must be a JSON list of dictionaries named `first_conf_req`, one dictionary per recipe ingredient, regardless of whether it needs to be bought.
#
# Each dictionary must include:
# - `name`: the ingredient name
# - `requested_quantity`: the quantity required by the recipe
# - `requested_unit`: the unit used in the recipe
# - `to_buy_min`: the amount that needs to be purchased (0 if not needed)
# - `to_buy_unit`: same as `requested_unit`
# - `explanation`: reason for decision (e.g., "expired", "expiring in 2 days", "only 100ml at home", "already at home")
#
# ❗Return only the JSON list. No extra explanations or formatting.
#
# Example:
# [
#   {{
#     "name": "milk",
#     "requested_quantity": 500,
#     "requested_unit": "ml",
#     "to_buy_min": 500,
#     "to_buy_unit": "ml",
#     "explanation": "expiring in 2 days"
#   }}
# ]
#
# Recipe ingredients:
# {df_recipe}
#
# Home inventory:
# {df_inventory}
#
# Parsed user input:
# {parsed_user_input}
#
# Today's date: {today}
# """
context_template = """
Given:
- df_recipe: list of ingredients with quantity and unit.
- df_inventory: list of home items with quantity, unit, and expiry.
- parsed_user_input: includes anything the user said they already have.
- today: current date.

Rules:
- If ingredient is missing, expired (expiry ≤ today), expiring soon (expiry in ≤3 days), or has insufficient quantity → mark for purchase.
- If expiring soon but available in quantity, still buy full amount.
- Match ingredients loosely (e.g., "onion" ≈ "red onion").
- Use same unit as in recipe.

Return a JSON list of dictionaries named `first_conf_req` with:
- name: ingredient name
- requested_quantity: quantity from recipe
- requested_unit: unit from recipe
- to_buy_min: amount to buy (0 if not needed)
- to_buy_unit: same as requested_unit
- explanation: reason for purchase (e.g., "expired", "expiring in 2 days", "enough at home")
-exp: write if you checked the date and how many days it is until expiry and if not applicable write which date you checked 
Only return the JSON list. Nothing else.

df_recipe:
{df_recipe}

df_inventory:
{df_inventory}

parsed_user_input:
{parsed_user_input}

today: {today}
"""


prompt_template = ChatPromptTemplate.from_template(context_template)

def generate_context(df_recipe, df_inventory, parsed_user_input):
    today = datetime.today().strftime('%Y-%m-%d')
    print(today)
    formatted_prompt = prompt_template.format_messages(
        df_recipe=df_recipe.to_string(index=False),
        df_inventory=df_inventory.to_string(index=False),
        parsed_user_input=parsed_user_input,
    today = today
    )
    with get_openai_callback() as cb:
        response = chat(messages=formatted_prompt)
        print("Prompt tokens:", cb.prompt_tokens)
        print("Completion tokens:", cb.completion_tokens)
        print("Total tokens (this run):", cb.total_tokens)
        update_total_tokens(cb.total_tokens, filename="../tokens/total_tokens_Nagham.txt")
    return response.content

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

    df_inventory = pd.DataFrame([
        {"name": "olive oil", "quantity": 30, "unit": "ml", "expiry": "2025-08-01"},
        {"name": "garlic", "quantity": 1, "unit": "units", "expiry": "2025-07-17"},
        {"name": "salt", "quantity": 5, "unit": "teaspoons", "expiry": "2026-01-01"},
        {"name": "sugar", "quantity": 100, "unit": "grams", "expiry": "2025-7-16"}
    ])

    print(generate_context(df_recipe, df_inventory, parsed_user_input))
