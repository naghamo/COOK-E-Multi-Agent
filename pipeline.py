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

def run_pipeline_to_inventory_confirmation(user_text, user_inventory):
    """Run until we know what the user already has and need user confirmation"""
    # 1. Parse user context
    # context = parse_context(user_text)
    # if not context.get('food_name'):
    #     return {"error": "No food name provided.", "context": context}
    #
    # # 2. Feasibility check (constraints, allergies, budget, basic plausibility)
    # feasible = check_feasibility(context)
    # if not feasible:
    #     return {"error": "Request not feasible (budget/ingredients/constraints).", "context": context}
    #
    # # 3. Retrieve/generate recipe
    # recipe = retrieve_recipe(context)
    # if not recipe:
    #     return {"error": "No matching recipe found.", "context": context}
    #
    # # 4. Parse recipe into ingredients
    # ingredients = parse_recipe(recipe)
    #
    # # 5. Inventory confirmation: Decide for each ingredient if user has enough at home, and ask user
    # # (This usually returns a list of questions/messages for the user, and the partial to-buy list)
    # ingredient_questions, need_confirmation = inventory_confirmation_agent(ingredients, user_inventory)
    #
    # return {
    #     "context": context,
    #     "recipe": recipe,
    #     "ingredients": ingredients,
    #     "inventory_questions": ingredient_questions,
    #     "need_confirmation": need_confirmation,
    #     # "to_buy_initial": initial_to_buy,  # Optional: initial guess
    # }


def run_pipeline_until_payment(context, recipe, confirmed_to_buy):
    # """Continue from confirmed inventory, through product matching, market selection, validation, confirmation, and payment"""
    # # 1. Product matching (ingredient -> real supermarket products)
    # products = match_products(confirmed_to_buy, context)
    #
    # # 2. Market selection (find cheapest, best, or preferred supermarket(s))
    # market_plan = select_market(products, context)
    #
    # # 3. Cart & delivery validation
    # cart_ok, cart_result = validate_cart(market_plan, context)
    # if not cart_ok:
    #     return {
    #         "error": cart_result,
    #         "market_plan": market_plan,
    #         "context": context
    #     }
    #
    # # 4. (Optional) Show cart to user for confirmation before execution
    # confirmation = confirm_order(cart_result)
    #
    # # 5. Execute order if user confirms
    # execution_result = execute_order(cart_result)
    #
    # return {
    #     "cart": cart_result,
    #     "confirmation": confirmation,
    #     "execution_result": execution_result,
    #     "context": context
    # }

