# from agents.llm_context_parser import parse_context
# from agents.recipe_retriever import retrieve_recipe
# from agents.feasibility_checker import check_feasibility
# from agents.recipe_parser import parse_recipe
# from agents.inventory_filter import filter_inventory
# from agents.product_matcher import match_products
# from agents.market_selector import select_market
# from agents.cart_delivery_validator import validate_cart
# from agents.order_confirmation import confirm_order
# from agents.order_execution import execute_order
#

def run_cooke_pipeline(user_text, user_inventory):
    print("Running Cooke pipeline with user input:", user_text)
    # # 1. Parse context from user input
    # context = parse_context(user_text)
    # if not context.get('food_name'):
    #     return {"error": "No food name provided."}
    #
    # # 2. Check feasibility
    # feasible = check_feasibility(context)
    # if not feasible:
    #     return {"error": "Request not feasible (budget/ingredients/constraints)."}
    #
    # # 3. Retrieve/generate recipe
    # recipe = retrieve_recipe(context)
    # if not recipe:
    #     return {"error": "No matching recipe found."}
    #
    # # 4. Parse recipe
    # ingredients = parse_recipe(recipe)
    # # 5. Filter inventory
    # to_buy = filter_inventory(ingredients, user_inventory)
    # # 6. Product matching
    # products = match_products(to_buy, context)
    # # 7. Market selection
    # market_plan = select_market(products, context)
    # # 8. Cart & delivery validation
    # cart_ok, cart_result = validate_cart(market_plan, context)
    # if not cart_ok:
    #     return {"error": cart_result}
    # # 9. Confirm order (or return cart for user confirmation)
    # confirmation = confirm_order(cart_result)
    # # 10. Execute order if user confirms
    # # (This can be triggered separately after user approval)
    # return {
    #     "context": context,
    #     "recipe": recipe,
    #     "to_buy": to_buy,
    #     "cart": cart_result,
    #     "confirmation": confirmation
    # }
def run_pipeline_confirmation_inventory(user_text):
    pass
