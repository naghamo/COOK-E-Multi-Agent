"""
Agent 5: Inventory Matcher
--------------------------
This agent matches inventory items at home to recipe ingredients, in batches,
by name (case-insensitive, partial matches allowed).
Returns a list of inventory items relevant for the recipe.
"""
import json
import re

import pandas as pd
from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from tokens.tokens_count import update_total_tokens

# --- ENV SETUP ---

# Load API keys and deployment settings from .env
load_dotenv()
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
DEPLOYMENT_NAME = "team10-gpt4o"
AZURE_OPENAI_ENDPOINT = "https://096290-oai.openai.azure.com"
API_VERSION = "2023-05-15"

# Initialize Azure OpenAI LLM for matching
chat = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version=API_VERSION,
    openai_api_type="azure",
    temperature=0,
)

# --- PROMPT ---

matcher_prompt = """
Given:
- recipe_ingredients: list of ingredients for the recipe (name, quantity, unit)
- inventory_batch: list of items at home (name, quantity, unit, expiry)

Return a JSON list of items from inventory_batch that represent the **same actual ingredient** as any item in recipe_ingredients, even if the names are slightly different (e.g., "red onion" and "onion" are the same; "baking soda" and "baking powder" are NOT). 
Base your decision on meaning, not just name similarity.

Do NOT include items that are only similar in name but are not the same actual ingredient.

Each item in the output should have:
- name
- quantity
- unit
- expiry

Return only the JSON list. Nothing else.

recipe_ingredients:
{df_recipe}

inventory_batch:
{df_inventory}
"""

matcher_template = ChatPromptTemplate.from_template(matcher_prompt)
def extract_json_from_llm(text):
    text = text.strip()
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text
def run_matcher_agent(df_recipe, df_inventory='../data/home_inventory.csv', batch_size=5,tokens_filename="../tokens/total_tokens_Nagham.txt"):
    """
    Splits the inventory into batches and matches each batch to the recipe ingredients.
    Returns: List of relevant inventory items (as dicts).
    """
    df_recipe= pd.DataFrame(df_recipe)
    matched_inventory = []
    df_inventory=pd.read_csv(df_inventory)
    recipe_str = df_recipe.to_string(index=False)
    # Split inventory to batches to save tokens/cost.
    inventory_batches = [df_inventory.iloc[i:i+batch_size] for i in range(0, len(df_inventory), batch_size)]

    for batch in inventory_batches:
        prompt = matcher_template.format_messages(
            df_recipe=recipe_str,
            df_inventory=batch.to_string(index=False)
        )
        with get_openai_callback() as cb:
            response = chat(messages=prompt)
        update_total_tokens(cb.total_tokens, filename=tokens_filename)
        # Parse response as JSON and append results
        raw_content = response.content
        json_string = extract_json_from_llm(raw_content)
        try:
            batch_result = json.loads(json_string)
            matched_inventory.extend(batch_result)
        except Exception as e:
            print(f"Error parsing matcher response: {e}\nResponse: {raw_content}")

    return matched_inventory

# Example usage for debugging and development
if __name__ == "__main__":

    df_recipe = pd.DataFrame([
        {"name": "pasta", "quantity": 200, "unit": "grams"},
        {"name": "tomato sauce", "quantity": 150, "unit": "ml"},
        {"name": "garlic", "quantity": 2, "unit": "units"},
        {"name": "olive oil", "quantity": 50, "unit": "ml"},
        {"name": "salt", "quantity": 1, "unit": "teaspoons"}
    ])
    df_inventory = pd.read_csv('../data/home_inventory.csv')
    matches = run_matcher_agent(df_recipe, df_inventory)
    print("Matched inventory:", matches)
