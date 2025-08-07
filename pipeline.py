# agents_pipeline.py

"""
COOK·E Full Pipeline Runner
--------------------------
This file defines the pipeline orchestration logic for your COOK·E project.
Each agent is responsible for a single, clear task. The pipeline is split into logical blocks
to enable both full runs and partial runs (for debugging or stepwise user interaction).

Agent mapping (by order & responsibility):
 _1_llm_context_parser.py         --> parse_context
 2_recipe_retriever.py           --> retrieve_recipe
 3_cart_delivery_validator.py    --> validate_cart
 4_recipe_parser.py              --> parse_recipe
 5_Inventory Matcher.py          --> run_matcher_agent
 6_inventory_confirmation.py     --> run_confirmation_agent
 7_inventory_filter.py           --> run_inventory_filter_agent
 8_product_matcher.py            --> match_products
 9_market_selector.py            --> select_market
10_order_confirmation.py         --> confirm_order
11_order_execution.py            --> execute_order
"""
import pandas as pd
# --- Import all agent runners  ---
from agents._1_llm_context_parser import parse_context
from agents._2_recipe_retriever import retrieve_recipe
# # from agents._3_cart_delivery_validator import validate_cart
from agents._4_recipe_parser import build_scaled_ingredient_list
from agents._5_Inventory_Matcher import run_matcher_agent
from agents._6_inventory_confirmation import run_confirmation_agent
from agents._7_inventory_filter import run_inventory_filter_agent

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
    # 2. Retrieve recipe based on context
    recipe = retrieve_recipe(context, tokens_filename=tokens_filename)
    if not recipe['feasible']:# If no feasible recipe is found, return an error
        return {"error": f"No feasible recipe found for the given context,{recipe['reason']}"}
    #3. Parse recipe ingredients
    ingredients = build_scaled_ingredient_list(context,recipe, tokens_filename=tokens_filename)
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


# --- Usage Example ---
if __name__ == "__main__":
    user_text = "Vegetarian lasagna for 6 people, no mushrooms, under 80 NIS, delivery"

    result=run_pipeline_to_inventory_confirmation(user_text,tokens_filename="tokens/total_tokens_Nagham.txt")


    print(result)
