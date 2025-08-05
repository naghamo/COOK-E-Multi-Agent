# market_selector.py
"""
GROCERY MARKET SELECTOR - LLM-POWERED BASKET OPTIMIZATION

This module takes matched ingredients from multiple supermarkets and uses GPT-4 to select
the optimal shopping basket based on user preferences (cost, single store, budget, etc.).

PIPELINE OVERVIEW:
1. Raw ingredient data → Compact text format (token optimization)
2. Send compact data + analysis to LLM
3. LLM returns simple selection with CIDs (product references)
4. Convert CIDs back to full product objects
5. Calculate totals and generate final basket

KEY DESIGN PATTERNS:
- Two-schema approach: Simple for LLM, rich for output
- CID system: Unique product identifiers for token efficiency
- Graceful error handling: Invalid selections become substitutions
- Complete coverage validation: Ensures all ingredients are handled

Given the {store_id: {store_info, desired_ingredients}} structure produced
by match_all_stores(), choose the optimal basket and return canonical JSON.
"""

import json, math, os
from typing import Dict, List, Tuple
from collections import defaultdict
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI

from langchain_community.callbacks import get_openai_callback
from tokens.tokens_count import update_total_tokens

from dotenv import load_dotenv

load_dotenv()

# ===============================================================================
# AZURE LLM CLIENT CONFIGURATION
# ===============================================================================

AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
DEPLOYMENT_NAME = "team10-gpt4o"
AZURE_OPENAI_ENDPOINT = "https://096290-oai.openai.azure.com"
API_VERSION = "2023-05-15"

llm = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version=API_VERSION,
    openai_api_type="azure",
    temperature=0  # Deterministic responses for consistency
)


# ===============================================================================
# PYDANTIC SCHEMAS - DATA MODELS FOR TYPE SAFETY
# ===============================================================================

class FullProduct(BaseModel):
    """
    Complete product information with all details.
    Used in final output after CID conversion.
    """
    supermarket_id: str  # Store identifier (e.g., "yohananof")
    item_code: str  # Product SKU/barcode
    name: str  # Product name
    size: float  # Quantity/size
    unit: str  # Unit of measurement
    price: float  # Price in local currency
    promo: str | None = None  # Promotion description (if any)
    packs_needed: int  # Number of packs to buy (set by LLM)
    selection_notes: str = ""  # LLM's reasoning for this selection


class IngredientSelection(BaseModel):
    """
    Simplified product selection for LLM function calling.
    Uses CID references instead of full product objects to save tokens.
    """
    store: str  # Store name (for LLM clarity)
    cid: str  # Compact ID referencing product in lookup table
    packs_needed: int  # Quantity to purchase
    notes: str = ""  # LLM's selection reasoning


class StoreSummary(BaseModel):
    """
    Summary information for each store in the final basket.
    Includes costs, delivery info, and LLM notes.
    """
    items_total: float  # Total cost of items (before delivery)
    delivery_fee: float  # Delivery cost
    grand_total: float  # Total including delivery
    meets_min_order: bool  # Whether order meets minimum requirement
    item_count: int = 0  # Number of different products
    notes: str = ""  # LLM's store-level observations


class TempBasketChoice(BaseModel):
    """
    Temporary schema for LLM function calling.
    Uses simple references and gets converted to rich objects later.
    """
    selection: Dict[str, IngredientSelection] = Field(default_factory=dict)
    store_summary: Dict[str, StoreSummary] = Field(default_factory=dict)


class BasketChoice(BaseModel):
    """
    Final output schema with complete product objects.
    This is what gets returned to the user.
    """
    selection: Dict[str, FullProduct] = Field(default_factory=dict)
    store_summary: Dict[str, StoreSummary] = Field(default_factory=dict)


# ===============================================================================
# DATA TRANSFORMATION FUNCTIONS - CORE PIPELINE LOGIC
# ===============================================================================

def _calculate_store_analysis(
        cid_lookup: Dict[str, Dict[str, FullProduct]],
        required_ingredients: List[str],
        stores_meta: Dict) -> str:
    """Calculate single-store coverage analysis for the LLM"""

    store_ingredient_coverage = defaultdict(set)
    store_costs = defaultdict(dict)

    # Calculate coverage and costs per store
    for ing in required_ingredients:
        for cid, product in cid_lookup.get(ing, {}).items():
            store_id = product.supermarket_id
            store_ingredient_coverage[store_id].add(ing)

            # Track cheapest price per ingredient per store
            current_cheapest = store_costs[store_id].get(ing, {}).get('price', float('inf'))
            if product.price < current_cheapest:
                store_costs[store_id][ing] = {
                    'price': product.price,
                    'promo': product.promo,
                    'cid': cid
                }

    analysis_lines = ["### Single Store Analysis:"]

    complete_stores = []
    incomplete_stores = []

    for store_id in store_ingredient_coverage:
        coverage = len(store_ingredient_coverage[store_id])
        missing = set(required_ingredients) - store_ingredient_coverage[store_id]

        # Get store metadata safely
        store_meta = stores_meta.get(store_id, {})
        delivery_fee = store_meta.get("delivery_fee", 0)
        rating = store_meta.get("rating", 0)
        min_order = store_meta.get("min_order", 0)

        estimated_items_total = sum(
            store_costs[store_id].get(ing, {}).get('price', 0)
            for ing in required_ingredients
        )
        total_with_delivery = estimated_items_total + delivery_fee

        if coverage == len(required_ingredients):
            complete_stores.append({
                'store': store_id,
                'items_total': estimated_items_total,
                'total_with_delivery': total_with_delivery,
                'rating': rating,
                'min_order': min_order,
                'delivery_fee': delivery_fee
            })

            analysis_lines.append(
                f"  {store_id}: COMPLETE - Est. total {total_with_delivery:.1f} "
                f"(items {estimated_items_total:.1f} + delivery {delivery_fee}) "
                f"| Rating: {rating}/5 | Min order: {min_order}"
            )
        else:
            incomplete_stores.append({
                'store': store_id,
                'coverage': coverage,
                'missing': sorted(missing)
            })

            analysis_lines.append(
                f" {store_id}: INCOMPLETE ({coverage}/{len(required_ingredients)}) "
                f"| Missing: {', '.join(sorted(missing))}"
                f"| Est. total (without missing) {total_with_delivery:.1f} "
                f"(items {estimated_items_total:.1f} + delivery {delivery_fee})"
                f"| Rating: {rating}/5 | Min order: {min_order}"
            )

    if complete_stores:
        best_complete = min(complete_stores, key=lambda x: x['total_with_delivery'])
        analysis_lines.append(
            f"\nSINGLE STORE RECOMMENDATION: {best_complete['store']} "
            f"({best_complete['total_with_delivery']:.1f} total including delivery)"
        )
        analysis_lines.append(
            f"   This gives you ALL ingredients from one store with ONE delivery fee."
        )
    else:
        # Find the store with the most ingredients
        best_incomplete = max(incomplete_stores, key=lambda x: x['coverage'])

        analysis_lines.append(
            f"\nNo store has all ingredients")
        analysis_lines.append(
            f"   Option: {best_incomplete['store']} has most of ({best_incomplete['coverage']}/{len(required_ingredients)}) ingredients")
        analysis_lines.append(
            f"   Missing: {', '.join(best_incomplete['missing'])}")
        analysis_lines.append(
            f"   Consider one of the options: Single incomplete store + substitutions vs multiple stores + multiple delivery fees")

    return "\n".join(analysis_lines)


def _compact_table(
        result: Dict, max_per_store: int = 3
) -> Tuple[str, Dict[str, Dict[str, FullProduct]]]:
    """
    Convert raw ingredient data to token-efficient format for LLM + lookup table.

    This is the key optimization that makes the system scalable. Instead of sending
    full product objects to the LLM (expensive in tokens), we create:
    1. Compact text table with essential info and CIDs
    2. Lookup table to convert CIDs back to full objects later

    Args:
        result: Raw matched ingredients data from match_all_stores()
        max_per_store: Maximum products per store per ingredient to include

    Returns:
        Tuple of (compact_text_for_llm, cid_lookup_table)

    Example CID format: "0-yohananof-1" = ingredient_index-store_id-product_rank
    """

    # Lookup table: ingredient_name -> cid -> FullProduct
    lookup: Dict[str, Dict[str, FullProduct]] = defaultdict(dict)
    lines = []  # Text lines for the compact table

    # Get all unique ingredients across all stores
    all_ingredients = set()
    for bundle in result.values():
        all_ingredients.update(bundle["desired_ingredients"].keys())

    # Process each ingredient to build compact table and lookup
    for idx, ing in enumerate(sorted(all_ingredients)):  # Sort for consistency
        lines.append(f"\n### {ing}")
        found_products = False

        # Check each store for this ingredient
        for store_id, bundle in result.items():
            picks = bundle["desired_ingredients"].get(ing, [])[:max_per_store]
            if not picks:
                continue  # Store has no products for this ingredient

            found_products = True

            # Create compact entries for each product
            for rank, prod in enumerate(picks):
                # Generate unique CID: ingredient_index-store_id-rank
                cid = f"{idx}-{store_id}-{rank}"

                # Store full product in lookup table
                lookup[ing][cid] = FullProduct(**prod)

                # Create compact text line for LLM
                promo = prod["promo"] or "-"
                lines.append(
                    f"{cid} | price={prod['price']:.2f} | "
                    f"{prod['size']} {prod['unit']} | promo={promo}"
                )

        # Handle ingredients with no available products
        if not found_products:
            lines.append(" No suitable products found in any store")

    return "\n".join(lines), lookup


# ===============================================================================
# UTILITY FUNCTIONS - HELPER LOGIC
# ===============================================================================

def _process_user_preferences(user_prefs: Dict) -> Dict:
    """
    Extract and validate relevant user preferences for the LLM.

    Filters the raw user preferences object to include only fields needed
    for grocery selection, with safe defaults for missing values.

    Args:
        user_prefs: Raw user preferences dictionary

    Returns:
        Cleaned preferences dictionary safe for JSON serialization
    """
    processed = {
        "raw_text": user_prefs.get("raw_text", ""),
        "special_requests": user_prefs.get("special_requests", ""),
        "delivery": user_prefs.get("delivery", "delivery"),
        "budget": user_prefs.get("budget"),
        "extra_fields": user_prefs.get("extra_fields", {}),
    }

    return processed


def _validate_llm_selection(temp_basket: TempBasketChoice, required_ingredients: List[str]) -> TempBasketChoice:
    """
    Ensure all required ingredients have products selected.
    Log warnings for any missing ingredients.
    """
    handled_ingredients = set(temp_basket.selection.keys())
    missing_ingredients = set(required_ingredients) - handled_ingredients

    # Log any missing ingredients as errors (this should not happen)
    for missing in missing_ingredients:
        print(f"ERROR: LLM failed to select product for '{missing}' - this violates the requirement")

    return temp_basket


# ===============================================================================
# LLM FUNCTION CALLING SETUP
# ===============================================================================

# Generate JSON schema for the temporary basket choice
temp_basket_schema = TempBasketChoice.model_json_schema()

# Ensure schema has required type field
if "type" not in temp_basket_schema:
    temp_basket_schema = {"type": "object", **temp_basket_schema}

temp_basket_schema.setdefault("required", ["selection"])

# Define the function that the LLM will call
BASKET_CHOICE_FN = {
    "name": "basket_choice",
    "description": (
        "Return a JSON object with EXACTLY these keys:\n"
        "  selection      – map ingredient → {store,cid,packs_needed,notes}\n"
        "  store_summary  – map store_id  → {items_total,delivery_fee,grand_total,meets_min_order,notes}\n\n"
        "CRITICAL FORMAT REQUIREMENTS:\n"
        "- Each ingredient key MUST be the ingredient's *name* (e.g. 'olive oil'), not an index\n"
        "- Each cid MUST be one that appears in the candidate table\n"
        "- Use 'notes' fields to explain your selection reasoning and any substitutions made\n\n"
        "SUBSTITUTION HANDLING:\n"
        "- If you deviate from user preferences, explain in the product's 'notes' field\n"
        "- Examples: 'Chose different store than preferred', 'Selected substitute product due to price'\n"
    ),
    "parameters": temp_basket_schema
}


# ===============================================================================
# MAIN PROCESSING FUNCTION - ORCHESTRATES THE ENTIRE PIPELINE
# ===============================================================================

def choose_best_market_llm(
        match_output: Dict,
        user_prefs: Dict,
        tokens_filename: str = "../tokens/total_tokens_Seva.txt"
) -> BasketChoice:
    """
    Main function: Convert matched ingredients to optimal shopping basket using LLM.

    PIPELINE STEPS:
    1. Process user preferences and create compact product table
    2. Generate store analysis for LLM decision-making
    3. Send data to LLM with structured function calling
    4. Validate LLM response for completeness
    5. Convert CID references back to full product objects
    6. Calculate store totals and generate final basket

    Args:
        match_output: Dictionary from match_all_stores() with ingredient matches
        user_prefs: User preferences including budget, single store preference, etc.
        tokens_filename: Path to token usage tracking file

    Returns:
        BasketChoice object with selected products, store summaries, and substitutions

    Raises:
        Various exceptions if LLM call fails or data validation issues occur
    """

    # ===== STEP 1: PREPARE DATA FOR LLM =====

    # Clean and validate user preferences
    filtered_prefs = _process_user_preferences(user_prefs)

    # Convert raw ingredient data to compact format + lookup table
    table, cid_lookup = _compact_table(match_output)

    # Extract store metadata for decision-making
    stores_meta = {
        k: {
            "delivery_fee": v["store_info"]["delivery_fee"],
            "min_order": v["store_info"]["min_order"],
            "rating": v["store_info"]["rating"],
        }
        for k, v in match_output.items()
        if v["store_info"]["is_open"]  # Only include open stores
    }

    # Get list of required ingredients (same across all stores)
    first_store = next(iter(match_output.values()))
    required_ingredients = list(first_store["desired_ingredients"].keys())

    # Generate store coverage analysis for LLM
    store_analysis = _calculate_store_analysis(cid_lookup, required_ingredients, stores_meta)

    # ===== STEP 2: PREPARE LLM PROMPTS =====

    system_prompt = (
        "You are a meticulous grocery planner optimizing for cost and user preferences. "
        "CRITICAL RULES:\n"
        "1. EVERY ingredient MUST have a product selected from the available options\n"
        "2. RESPECT user preferences - they are NOT suggestions, they are REQUIREMENTS\n"
        "3. If user wants single_store=True, you MUST prioritize complete stores over individual item prices\n"
        "4. Calculate TOTAL cost including ALL delivery fees - not just item prices\n"
        "5. Use 'notes' fields to explain your reasoning and any trade-offs\n\n"
        "SINGLE STORE PRIORITY:\n"
        "- When single_store is requested, find stores that have ALL or MOST ingredients\n"
        "- e.g. a ₪50 item from one store is better than ₪30 items from multiple stores with delivery fees\n"
        "- Only use multiple stores if NO single store can provide reasonable coverage\n\n"
        "COST CALCULATION:\n"
        "- Always include delivery fees in your cost analysis\n"
        "- Multiple stores = multiple delivery fees = often more expensive total\n"
        "- Compare TOTAL cost (items + delivery) not just item prices\n\n"
        "OPTIMIZATION PRIORITY:\n"
        "1. Follow user's store preference (single vs multi)\n"
        "2. Select products that minimize TOTAL cost (including delivery)\n"
        "3. Explain reasoning and any deviations from user's request clearly in 'notes'\n\n"
        "Remember: User preferences are constraints, not suggestions. Total cost matters more than individual item prices."
    )

    user_prompt = (
        f"### REQUIRED INGREDIENTS (you must match a product for ALL of these):\n{', '.join(required_ingredients)}\n\n"
        f"### User preferences\n{json.dumps(filtered_prefs, ensure_ascii=False)}\n\n"
        f"### Stores metadata\n{json.dumps(stores_meta, ensure_ascii=False)}\n\n"
        "### HOW TO READ THE PRODUCT TABLE:\n"
        "Each product has a unique CID (Compact ID) in format: [ingredient_index]-[store]-[rank]\n"
        "Example: '2-yohananof-0' means:\n"
        "  - Ingredient index 2 (3rd ingredient alphabetically)\n"
        "  - Store: yohananof\n"
        "  - Rank 0: cheapest/first option for this ingredient at this store\n\n"
        "TABLE FORMAT EXPLANATION:\n"
        "CID | price=XX.XX | SIZE UNIT | promo=PROMOTION\n"
        "- CID: Use this exact code in your selection\n"
        "- price: Cost in local currency\n"
        "- SIZE UNIT: Quantity and measurement\n"
        "- promo: Special offers (or '-' if none)\n\n"
        f"### Available products by ingredient\n{table}\n\n"
        f"{store_analysis}\n\n"
        "DECISION FRAMEWORK:\n"
        "1. ONLY use CIDs that appear in the table above - you cannot invent CIDs\n"
        "2. To select a product, copy its exact CID (e.g., '2-yohananof-0')\n"
        "3. EVERY ingredient must have a product selected\n"
        "4. If you deviate from any of the user preferences, explain why in the product's 'notes' field\n"
        "5. Examples of good notes: 'Chose multi-store for better availability', 'Selected substitute for budget'\n\n"
        "When you respond, CALL the function `basket_choice` with your optimized selection."
    )

    # ===== STEP 3: CALL LLM WITH FUNCTION CALLING =====

    messages = [SystemMessage(system_prompt), HumanMessage(user_prompt)]

    with get_openai_callback() as cb:
        # Make the LLM call with structured function calling
        response = llm.invoke(
            messages,
            functions=[BASKET_CHOICE_FN],  # Available functions
            function_call={"name": "basket_choice"}  # Force specific function call
        )

        # Parse the function call response
        payload = json.loads(response.additional_kwargs["function_call"]["arguments"])
        temp_basket = TempBasketChoice.model_validate(payload)

        # Validate that all ingredients are handled
        temp_basket = _validate_llm_selection(temp_basket, required_ingredients)

        # Log token usage for monitoring
        print(f"\nTokens | prompt {cb.prompt_tokens}  completion {cb.completion_tokens}  total {cb.total_tokens}")
        update_total_tokens(cb.total_tokens, filename=tokens_filename)

    # ===== STEP 4: CONVERT CID REFERENCES TO FULL PRODUCTS =====

    # Initialize final basket structure
    final_basket = BasketChoice(
        store_summary=temp_basket.store_summary,  # Copy store summaries as-is
    )

    # Track total spending per store for summary calculations
    store_totals = defaultdict(float)

    # Convert each ingredient selection from CID to full product
    for ing, sel in temp_basket.selection.items():
        # Look up the full product using the CID
        full_product = cid_lookup.get(ing, {}).get(sel.cid)

        if full_product:
            # SUCCESS: Valid CID found, create full product copy
            product_copy = FullProduct(**full_product.model_dump())
            product_copy.packs_needed = sel.packs_needed  # Set quantity from LLM
            product_copy.selection_notes = sel.notes  # Add LLM reasoning

            # Add to final selection
            final_basket.selection[ing] = product_copy

            # Update store total for summary calculations
            store_totals[product_copy.supermarket_id] += product_copy.price * sel.packs_needed
        else:
            # FAILURE: Invalid CID, add to substitutions instead of crashing
            print(f"Warning: Could not find product for ingredient '{ing}' with cid '{sel.cid}'")
            missing_product = FullProduct(
                supermarket_id="unknown",
                item_code="missing",
                name=f"MISSING: {ing}",
                size=0.0,
                unit="unknown",
                price=0.0,
                promo=None,
                packs_needed=1,
                selection_notes=f"ERROR: Invalid CID '{sel.cid}' - product not found in database"
            )

            missing_product.packs_needed = sel.packs_needed
            final_basket.selection[ing] = missing_product

    # ===== STEP 5: CALCULATE STORE SUMMARIES =====

    # Generate store summaries for all stores with selected products
    for store_id, items_total in store_totals.items():
        # Get all products from this store
        store_products = [p for p in final_basket.selection.values()
                          if p.supermarket_id == store_id]

        # Safely extract store metadata
        store_meta = stores_meta.get(store_id, {})
        delivery_fee = store_meta.get("delivery_fee", 0)
        grand_total = items_total + delivery_fee
        min_order = store_meta.get("min_order", 0)
        meets_minimum = items_total >= min_order

        # Use LLM's store summary notes if provided
        if store_id in temp_basket.store_summary:
            llm_summary = temp_basket.store_summary[store_id]
            summary_notes = llm_summary.notes if hasattr(llm_summary, 'notes') else ""
        else:
            summary_notes = ""

        # Create complete store summary
        final_basket.store_summary[store_id] = StoreSummary(
            items_total=round(items_total, 2),
            delivery_fee=delivery_fee,
            grand_total=round(grand_total, 2),
            meets_min_order=meets_minimum,
            item_count=len(store_products),
            notes=summary_notes
        )

    # Clean up: Remove any store summaries for stores with no products
    final_basket.store_summary = {
        store_id: summary for store_id, summary in final_basket.store_summary.items()
        if summary.items_total > 0
    }

    return final_basket


# ===============================================================================
# DEMO/TEST SECTION - EXAMPLE USAGE
# ===============================================================================

if __name__ == "__main__":
    user_prefs = {
        "food_name": "Peanut satay noodles",
        "people": 2,
        "delivery": "delivery",
        "budget": 150,
        "raw_text": "I'd like everything from one store if possible, under ₪150",
        "special_requests": "",
        "extra_fields": {
            "single_store": True
        }
    }

    matched_ingredients = {
      "tiv_taam": {
        "store_info": {
          "supermarket": "tiv_taam",
          "delivery_fee": 18,
          "delivery_time_hr": 2,
          "min_order": 30,
          "is_open": True,
          "rating": 0.5,
          "url": "https://www.tivtaam.co.il/"
        },
        "desired_ingredients": {
          "olive oil": [],
          "tomatoes": [],
          "cheddar": [],
          "salt": [
            {
              "supermarket_id": "tiv_taam",
              "item_code": "4903001925784.0",
              "name": "Yamasa Premium Low Salt Soy Sauce 500 ml",
              "size": 101.44206810552905,
              "unit": "teaspoons",
              "price": 20.9,
              "promo": "5 NIS discount",
              "packs_needed": 1
            }
          ],
          "garlic": [],
          "onion": []
        }
      },
      "yohananof": {
        "store_info": {
          "supermarket": "yohananof",
          "delivery_fee": 15,
          "delivery_time_hr": 4,
          "min_order": 40,
          "is_open": True,
          "rating": 3.0,
          "url": "https://yochananof.co.il/"
        },
        "desired_ingredients": {
          "olive oil": [
            {
              "supermarket_id": "yohananof",
              "item_code": "7290003427154.0",
              "name": "Extra virgin olive oil",
              "size": 59.0,
              "unit": "raw",
              "price": 38.9,
              "promo": None,
              "packs_needed": 1
            }
          ],
          "tomatoes": [
            {
              "supermarket_id": "yohananof",
              "item_code": "7290016372342.0",
              "name": "Kobe trio tomatoes",
              "size": 88.0,
              "unit": "raw",
              "price": 17.9,
              "promo": None,
              "packs_needed": 4
            }
          ],
          "cheddar": [
            {
              "supermarket_id": "yohananof",
              "item_code": "7290102300822.0",
              "name": "English Cheddar Cheese (Cut)",
              "size": 80.0,
              "unit": "raw",
              "price": 28.1,
              "promo": None,
              "packs_needed": 4
            }
          ],
          "salt": [],
          "garlic": [],
          "onion": []
        }
      },
      "shufersal": {
        "store_info": {
          "supermarket": "shufersal",
          "delivery_fee": 21,
          "delivery_time_hr": 3,
          "min_order": 60,
          "is_open": True,
          "rating": 1.3,
          "url": "https://www.shufersal.co.il/online/"
        },
        "desired_ingredients": {
          "olive oil": [],
          "tomatoes": [],
          "cheddar": [],
          "salt": [],
          "garlic": [
            {
              "supermarket_id": "shufersal",
              "item_code": "7290000999456.0",
              "name": "Garlic in a package",
              "size": 16.0,
              "unit": "raw",
              "price": 5.9,
              "promo": "5% off",
              "packs_needed": 1
            }
          ],
          "onion": [
            {
              "supermarket_id": "shufersal",
              "item_code": "7290000001036.0",
              "name": "Dried red onion",
              "size": 97.0,
              "unit": "raw",
              "price": 3.9,
              "promo": None,
              "packs_needed": 1
            }
          ]
        }
      },
      "rami_levy": {
        "store_info": {
          "supermarket": "rami_levy",
          "delivery_fee": 15,
          "delivery_time_hr": 4,
          "min_order": 30,
          "is_open": True,
          "rating": 1.5,
          "url": "https://www.rami-levy.co.il/he"
        },
        "desired_ingredients": {
          "olive oil": [
            {
              "supermarket_id": "rami_levy",
              "item_code": "7290118661382.0",
              "name": "Olive oil for light Primio",
              "size": 77.0,
              "unit": "raw",
              "price": 24.9,
              "promo": None,
              "packs_needed": 1
            }
          ],
          "tomatoes": [
            {
              "supermarket_id": "rami_levy",
              "item_code": "7290000208022.0",
              "name": "Prepare crushed tomatoes",
              "size": 38.0,
              "unit": "raw",
              "price": 10.8,
              "promo": None,
              "packs_needed": 8
            }
          ],
          "cheddar": [
            {
              "supermarket_id": "rami_levy",
              "item_code": "7290002857365.0",
              "name": "White English Cheddar 200 g",
              "size": 200.0,
              "unit": "gr",
              "price": 19.1,
              "promo": "5 NIS discount",
              "packs_needed": 2
            }
          ],
          "salt": [
            {
              "supermarket_id": "rami_levy",
              "item_code": "7290004064464.0",
              "name": "Coarse salt in a 1 kg bag",
              "size": 1.0,
              "unit": "kilogram",
              "price": 1.8,
              "promo": None,
              "packs_needed": 1
            }
          ],
          "garlic": [
            {
              "supermarket_id": "rami_levy",
              "item_code": "7290000208930.0",
              "name": "Yachin crushed garlic 180",
              "size": 72.0,
              "unit": "raw",
              "price": 6.5,
              "promo": None,
              "packs_needed": 1
            }
          ],
          "onion": []
        }
      },
      "victory": {
        "store_info": {
          "supermarket": "victory",
          "delivery_fee": 21,
          "delivery_time_hr": 2,
          "min_order": 30,
          "is_open": True,
          "rating": 1.9,
          "url": "https://www.victoryonline.co.il/"
        },
        "desired_ingredients": {
          "olive oil": [
            {
              "supermarket_id": "victory",
              "item_code": "72961209.0",
              "name": "15% Olive Oil 200 ml in a bottle",
              "size": 200.0,
              "unit": "ml",
              "price": 2.81,
              "promo": None,
              "packs_needed": 1
            }
          ],
          "tomatoes": [
            {
              "supermarket_id": "victory",
              "item_code": "7290000208022.0",
              "name": "Crushed tomatoes Yachin (800 grams)",
              "size": 800.0,
              "unit": "gr",
              "price": 10.9,
              "promo": None,
              "packs_needed": 1
            }
          ],
          "cheddar": [],
          "salt": [
            {
              "supermarket_id": "victory",
              "item_code": "3183280028036.0",
              "name": "Coarse Atlantic Sea Salt in Balein (1 kg)",
              "size": 1.0,
              "unit": "kilogram",
              "price": 17.9,
              "promo": None,
              "packs_needed": 1
            }
          ],
          "garlic": [
            {
              "supermarket_id": "victory",
              "item_code": "7290000000459.0",
              "name": "Bulk garlic made in China",
              "size": 48.0,
              "unit": "raw",
              "price": 29.9,
              "promo": None,
              "packs_needed": 1
            }
          ],
          "onion": [
            {
              "supermarket_id": "victory",
              "item_code": "2072.0",
              "name": "Purple Onion Israel",
              "size": 85.0,
              "unit": "raw",
              "price": 6.9,
              "promo": None,
              "packs_needed": 1
            }
          ]
        }
      },
      "osher_ad": {
        "store_info": {
          "supermarket": "osher_ad",
          "delivery_fee": 16,
          "delivery_time_hr": 2,
          "min_order": 50,
          "is_open": True,
          "rating": 3.7,
          "url": "https://osherad.co.il/"
        },
        "desired_ingredients": {
          "olive oil": [
            {
              "supermarket_id": "osher_ad",
              "item_code": "7290005437069.0",
              "name": "Olive oil for the taste of 1 light",
              "size": 30.0,
              "unit": "raw",
              "price": 19.9,
              "promo": "Buy 1 Get 1",
              "packs_needed": 2
            }
          ],
          "tomatoes": [
            {
              "supermarket_id": "osher_ad",
              "item_code": "7290009000115.0",
              "name": "Cherry tomatoes",
              "size": 85.0,
              "unit": "raw",
              "price": 15.9,
              "promo": None,
              "packs_needed": 4
            }
          ],
          "cheddar": [
            # {
            #   "supermarket_id": "osher_ad",
            #   "item_code": "7290017065786.0",
            #   "name": "Grated Cheddar Cheese 200 g",
            #   "size": 200.0,
            #   "unit": "gr",
            #   "price": 20.9,
            #   "promo": "Buy 1 Get 1",
            #   "packs_needed": 2
            # }
          ],
          "salt": [
            {
              "supermarket_id": "osher_ad",
              "item_code": "7290002319580.0",
              "name": "Mia Lemon Salt 100 gr",
              "size": 100.0,
              "unit": "gram",
              "price": 5.8,
              "promo": "5 NIS discount",
              "packs_needed": 1
            }
          ],
          "garlic": [
            {
              "supermarket_id": "osher_ad",
              "item_code": "6936613888667.0",
              "name": "4 heads of garlic",
              "size": 67.0,
              "unit": "raw",
              "price": 4.9,
              "promo": None,
              "packs_needed": 1
            }
          ],
          "onion": [
            {
              "supermarket_id": "osher_ad",
              "item_code": "7290003996087.0",
              "name": "Green onion Glatt",
              "size": 95.0,
              "unit": "raw",
              "price": 7.9,
              "promo": None,
              "packs_needed": 1
            }
          ]
        }
      },
      "mega": {
        "store_info": {
          "supermarket": "mega",
          "delivery_fee": 11,
          "delivery_time_hr": 4,
          "min_order": 50,
          "is_open": True,
          "rating": 0.5,
          "url": "https://mega-dummy-store.co.il/"
        },
        "desired_ingredients": {
          "olive oil": [
            {
              "supermarket_id": "mega",
              "item_code": "7290006374660.0",
              "name": "Extra virgin olive oil",
              "size": 83.0,
              "unit": "raw",
              "price": 44.9,
              "promo": None,
              "packs_needed": 1
            }
          ],
          "tomatoes": [
            {
              "supermarket_id": "mega",
              "item_code": "3270190192749.0",
              "name": "Whole peeled tomatoes",
              "size": 60.0,
              "unit": "raw",
              "price": 10.9,
              "promo": None,
              "packs_needed": 5
            }
          ],
          "cheddar": [
            {
              "supermarket_id": "mega",
              "item_code": "7290017065786.0",
              "name": "Hard English cheese, Cheddar",
              "size": 91.0,
              "unit": "raw",
              "price": 30.9,
              "promo": None,
              "packs_needed": 3
            }
          ],
          "salt": [
            {
              "supermarket_id": "mega",
              "item_code": "7290107061537.0",
              "name": "Atlantic sea salt",
              "size": 26.0,
              "unit": "raw",
              "price": 60.0,
              "promo": None,
              "packs_needed": 1
            }
          ],
          "garlic": [
            {
              "supermarket_id": "mega",
              "item_code": "7290019094173.0",
              "name": "Peeled garlic 170 grams per pack",
              "size": 170.0,
              "unit": "gram",
              "price": 11.0,
              "promo": "10 NIS discount",
              "packs_needed": 1
            }
          ],
          "onion": []
        }
      }
    }

    basket = choose_best_market_llm(matched_ingredients, user_prefs)
    print(basket.model_dump_json(indent=2))