
"""
COOK·E Full Pipeline Runner
--------------------------
This file defines the pipeline orchestration logic for your COOK·E project.
Each agent is responsible for a single, clear task. The pipeline is split into logical blocks
to match the pipline with the web each bloack start for the end point of the brev block and ends with a confirmation need from the user.
Agent mapping (by order & responsibility):
1. LLM Context Parser: Parses user input to extract structured context.
2. Recipe Retriever: Retrieves a recipe based on the parsed context.
3. Recipe Parser: Parses the recipe to build a scaled ingredient list.
4. Inventory Matcher: Matches the ingredient list with the home inventory.
5. Inventory Confirmation: Confirms which ingredients need to be bought.
6. product Matcher: Matches products from different stores.
7. Market Selector: Selects the best market based on product availability and prices.
8. Order Execution: Finalizes the order and generates PDF receipts.
"""
import pandas as pd
# --- Import all agent runners  ---
import warnings
import importlib

# Dynamically import LangChainDeprecationWarning without importing the rest of LangChain yet
LangChainDeprecationWarning = getattr(
    importlib.import_module("langchain_core._api"),
    "LangChainDeprecationWarning"
)

# Hide LangChain deprecation warnings
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# Optionally hide all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from agents._1_llm_context_parser import parse_context
from agents._2_recipe_retriever import retrieve_recipe
from agents._3_recipe_parser import build_scaled_ingredient_list
from agents._4_Inventory_Matcher import run_matcher_agent
from agents._5_inventory_confirmation import run_confirmation_agent
from agents._6_product_matcher import match_all_stores
from agents._7_market_selector import choose_best_market_llm
from agents._8_order_execution import finalize_order_generate_pdfs
import warnings
import asyncio


from pydantic import BaseModel

def _to_dict(x):
    """Converts a Pydantic model or object to a dictionary."""
    if isinstance(x, BaseModel):
        # pydantic v2
        try:
            return x.model_dump()
        except AttributeError:
            # pydantic v1
            return x.dict()
    return x

global_context = {}  # Global variable to store context for later use in the pipeline
# --- Pipeline Functions ---
def run_pipeline_to_inventory_confirmation(user_text, tokens_filename="tokens/total_tokens.txt"):
    """
    Runs pipeline up to the inventory confirmation stage (before user confirmation).
    Returns parsed context, recipe, ingredients, and the list for confirmation.
    """

    # 1. Parse user context
    context = parse_context(user_text, tokens_filename=tokens_filename)
    if context['error']: # If there is an error in context parsing, return it
        return {"error": f"{context['error']}"}
    global global_context
    global_context = context # Store context globally for later use
    # 2. Retrieve recipe based on context
    recipe = retrieve_recipe(context, tokens_filename=tokens_filename)
    if not recipe['feasible']:# If no feasible recipe is found, return an error
        return {"error": f"No feasible recipe found for the given context,{recipe['reason']}"}
    #3. Parse recipe ingredients
    ingredients = build_scaled_ingredient_list(context,recipe, tokens_filename=tokens_filename)
    # print(f"Parsed ingredients: {ingredients}")
    #4. Run inventory matcher agent
    matched_inventory = run_matcher_agent(ingredients, df_inventory='data/home_inventory.csv', tokens_filename=tokens_filename)
    #5. Run confirmation agent
    confirmation_json = run_confirmation_agent(ingredients, matched_inventory, context, tokens_filename=tokens_filename)
    return {
        "context": context,
        "recipe": recipe,
        "ingredients": ingredients,
        "matched_inventory": matched_inventory,
        "confirmation_json": confirmation_json,
    }
def run_pipeline_to_order_confirmation(ingredients, tokens_filename="tokens/total_tokens.txt"):
    '''runs the pipline from first inventory confirmation to order execution.'''
    global global_context

    #6. choose products from stores
    products =asyncio.run(match_all_stores(ingredients, tokens_filename=tokens_filename))
    # 7. Select best market based on products
    best_market = choose_best_market_llm(products,global_context, tokens_filename=tokens_filename)
    best_market = _to_dict(best_market)

    # Works whether best_market is dict or object
    feasible = best_market.get("feasible") if isinstance(best_market, dict) else getattr(best_market, "feasible",
                                                                                         None)
    if feasible is False:
        reason = best_market.get("reason") if isinstance(best_market, dict) else getattr(best_market, "reason", "")
        suggestions = best_market.get("suggestions") if isinstance(best_market, dict) else getattr(best_market,
                                                                                                   "suggestions",
                                                                                                   [])
        return {
            "error": f" {reason}, {suggestions}"
        }

    return best_market
def run_pipeline_to_order_execution(stores_dict, recipe_title, delivery_choices, user_name, pdf_dir='static/receipts'):
    """
    Runs the final part of the pipeline to generate PDF receipts for the order."""

    pdf_filenames = finalize_order_generate_pdfs(
        stores_dict=stores_dict,
        recipe_title=recipe_title,
        delivery_choices=delivery_choices,
        user_name=user_name,
        pdf_dir=pdf_dir
    )
    return pdf_filenames



#
# # --- Usage Example ---
# if __name__ == "__main__":
#     # user_text = "Vegetarian lasagna for 6 people, no mushrooms, under 80 NIS, delivery"
#     #
#     # result=run_pipeline_to_inventory_confirmation(user_text,tokens_filename="tokens/total_tokens_Nagham.txt")
#     #secound part of the pipeline
#     global_context = {
#         "food_name": "Vegetarian lasagna",
#         "people": 4,
#         "delivery": "delivery",
#         "special_requests": "no mushrooms",
#         "budget": 125,
#         "raw_text": "Vegetarian lasagna for 4 people, no mushrooms, under 80 NIS, no delivery",
#         "extra_fields": {},
#         "error": None
#     }
#
#     ingredient = [
#         {'name': 'olive oil', 'to_buy_min': 50, 'to_buy_unit': 'ml'},
#         {'name': 'tomatoes', 'to_buy_min': 300, 'to_buy_unit': 'gr'},
#         {'name': 'cheddar', 'to_buy_min': 250, 'to_buy_unit': 'gr'},
#         {'name': 'salt', 'to_buy_min': 1, 'to_buy_unit': 'teaspoons'},
#         {'name': 'garlic', 'to_buy_min': 3, 'to_buy_unit': 'cloves'},
#         {'name': 'onion', 'to_buy_min': 5, 'to_buy_unit': 'units'},
#         {"name": "tomato sauce", "to_buy_min": 500, "to_buy_unit": "milliliter"}
#         # {'name': 'bread', 'to_buy_min': 1, 'to_buy_unit': 'loaf'},
#         # {'name': 'milk', 'to_buy_min': 1, 'to_buy_unit': 'liter'},
#     ]
#     print(run_pipeline_to_order_confirmation(ingredient, tokens_filename="tokens/total_tokens.txt"))
#     # print(result)
