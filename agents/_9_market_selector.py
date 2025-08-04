# market_selector.py
"""
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

# ---------------------------------------------------------------
# Azure LLM client (unchanged)
# ---------------------------------------------------------------
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
    temperature=0
)


######################################################################
# 1. Pydantic schemas for OpenAI function-calling
######################################################################

class Product(BaseModel):
    supermarket_id: str
    item_code: str
    name: str
    size: float
    unit: str
    price: float
    promo: str | None = None
    packs_needed: int


class IngredientSelection(BaseModel):
    store: str
    cid: str  # short id from the compact table
    packs_needed: int  # let the model decide multiplicity


class StoreSummary(BaseModel):
    items_total: float
    delivery_fee: float
    grand_total: float
    meets_min_order: bool


# Temporary schema for LLM function calling
class TempBasketChoice(BaseModel):
    selection: Dict[str, IngredientSelection] = Field(default_factory=dict)
    store_summary: Dict[str, StoreSummary] = Field(default_factory=dict)
    substitutions: List[Dict] = Field(default_factory=list)


# Final schema with Product objects
class BasketChoice(BaseModel):
    selection: Dict[str, Product] = Field(default_factory=dict)
    store_summary: Dict[str, StoreSummary] = Field(default_factory=dict)
    substitutions: List[Dict] = Field(default_factory=list)


######################################################################
# 2. Helper: compact candidate table ➜ prompt-string
######################################################################

def _compact_table(
        result: Dict, max_per_store: int = 2
) -> Tuple[str, Dict[str, Dict[str, Product]]]:
    """
    Returns a token-lean text table + a lookup[ingredient][cid] = Product.
    Cid format: f"{ingredient_idx}-{store_id}-{rank}"  (unique & short)
    """
    lookup: Dict[str, Dict[str, Product]] = defaultdict(dict)
    lines = []

    for idx, (ing, _) in enumerate(result[next(iter(result))]["desired_ingredients"].items()):
        # iterate ingredients by order of first store (all share same keys)
        lines.append(f"\n### {ing}")
        for store_id, bundle in result.items():
            picks = bundle["desired_ingredients"].get(ing, [])[:max_per_store]
            if not picks:
                continue
            for rank, prod in enumerate(picks):
                cid = f"{idx}-{store_id}-{rank}"
                lookup[ing][cid] = Product(**prod)
                promo = prod["promo"] or "-"
                lines.append(
                    f"{cid} | {prod['price']:.2f} | "
                    f"{prod['size']} {prod['unit']} | promo={promo}"
                )
    return "\n".join(lines), lookup


######################################################################
# 3. Main entry point ------------------------------------------------
######################################################################

# Use temporary schema for LLM function calling
temp_basket_schema = TempBasketChoice.model_json_schema()

if "type" not in temp_basket_schema:
    temp_basket_schema = {"type": "object", **temp_basket_schema}

temp_basket_schema.setdefault("required", ["selection"])

BASKET_CHOICE_FN = {
    "name": "basket_choice",
    "description": (
        "Return a JSON object with EXACTLY these keys:\n"
        "  selection      – map ingredient → {store,cid,packs_needed}\n"
        "  store_summary  – map store_id  → {items_total,delivery_fee,"
        "grand_total,meets_min_order}\n"
        "  substitutions  – array (can be empty)\n"
        "Each ingredient key MUST be the ingredient's *name* (e.g. "
        "'olive oil'), not an index.\n"
        "Each cid MUST be one that appears in the candidate table."
    ),
    "parameters": temp_basket_schema  # Use temporary schema
}


def choose_best_market_llm(
        match_output: Dict,
        user_prefs: Dict,
        tokens_filename: str = "../tokens/total_tokens_Seva.txt"
) -> BasketChoice:
    """
    Params
    ------
    match_output : dict
        Returned by match_all_stores().
    user_prefs : dict
        Parsed high-level prefs, e.g.
        { "budget": 150.0, "single_store": True, "fav_stores": ["yohananof"] }
    """
    # Filtering only necessary user preferences
    filtered_prefs = {
        "raw_text": user_prefs.get("raw_text", ""),
        "special_requests": user_prefs.get("special_requests", ""),
        "extra_fields": user_prefs.get("extra_fields", {}),
        "delivery": user_prefs.get("delivery", "delivery"),
        "budget": user_prefs.get("budget"),
    }

    table, cid_lookup = _compact_table(match_output)
    stores_meta = {
        k: {
            "delivery_fee": v["store_info"]["delivery_fee"],
            "min_order": v["store_info"]["min_order"],
            "rating": v["store_info"]["rating"],
        }
        for k, v in match_output.items()
        if v["store_info"]["is_open"]
    }

    first_store = next(iter(match_output.values()))
    required_ingredients = list(first_store["desired_ingredients"].keys())

    system_prompt = (
        "You are a meticulous grocery planner. You MUST handle ALL requested ingredients. "
        "For each ingredient, either:\n"
        "1. Select a product from the available options, OR\n"
        "2. Add it to substitutions with a clear reason why it cannot be fulfilled.\n"
        "NEVER ignore any ingredient. Select the cheapest workable basket while following user constraints and preferences."
    )

    user_prompt = (
        f"### REQUIRED INGREDIENTS (you must handle ALL of these):\n{', '.join(required_ingredients)}\n\n"
        f"### User preferences\n{json.dumps(filtered_prefs, ensure_ascii=False)}\n\n"
        f"### Stores\n{json.dumps(stores_meta, ensure_ascii=False)}\n\n"
        f"### Candidate products\n{table}\n\n"
        "IMPORTANT: Every ingredient in the required list must either appear in your 'selection' "
        "or in your 'substitutions' array. No exceptions.\n\n"
        "When you respond, CALL the function `basket_choice` "
        "with arguments that follow the schema above."
    )
    messages = [SystemMessage(system_prompt), HumanMessage(user_prompt)]

    with get_openai_callback() as cb:
        response = llm.invoke(
            messages,
            functions=[BASKET_CHOICE_FN],  # must be a list of specs
            function_call={"name": "basket_choice"}  # force a call to this fn
        )
        payload = json.loads(response.additional_kwargs["function_call"]["arguments"])
        temp_basket = TempBasketChoice.model_validate(payload)  # Use temporary schema

        print(f"\nTokens | prompt {cb.prompt_tokens}  completion {cb.completion_tokens}  total {cb.total_tokens}")
        update_total_tokens(cb.total_tokens, filename=tokens_filename)

    # ---------- Convert to final format with Product objects ---------- #
    final_basket = BasketChoice(
        store_summary=temp_basket.store_summary,
        substitutions=temp_basket.substitutions
    )

    store_totals = defaultdict(float)

    # Convert IngredientSelection to Product objects
    for ing, sel in temp_basket.selection.items():
        full_product = cid_lookup.get(ing, {}).get(sel.cid)
        if full_product:
            # Create a copy and update packs_needed
            product_copy = Product(**full_product.model_dump())
            product_copy.packs_needed = sel.packs_needed

            final_basket.selection[ing] = product_copy
            store_totals[product_copy.supermarket_id] += product_copy.price * sel.packs_needed
        else:
            print(f"Warning: Could not find product for ingredient '{ing}' with cid '{sel.cid}'")
            # Add to substitutions
            final_basket.substitutions.append({
                "ingredient": ing,
                "reason": f"Invalid product reference: cid '{sel.cid}' not found",
                "suggested_alternative": "Please select a different product"
            })

    # Create/update store summaries for all stores that have products
    for store_id, total in store_totals.items():
        if store_id not in final_basket.store_summary:
            final_basket.store_summary[store_id] = StoreSummary(
                items_total=0,
                delivery_fee=0,
                grand_total=0,
                meets_min_order=False
            )

        summary = final_basket.store_summary[store_id]
        summary.items_total = round(total, 2)
        summary.delivery_fee = stores_meta.get(store_id, {}).get("delivery_fee", 0)
        summary.grand_total = round(total + summary.delivery_fee, 2)
        summary.meets_min_order = total >= stores_meta.get(store_id, {}).get("min_order", 0)

    # Remove empty store summaries (stores with no products)
    final_basket.store_summary = {
        store_id: summary for store_id, summary in final_basket.store_summary.items()
        if summary.items_total > 0
    }

    return final_basket

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
            {
              "supermarket_id": "osher_ad",
              "item_code": "7290017065786.0",
              "name": "Grated Cheddar Cheese 200 g",
              "size": 200.0,
              "unit": "gr",
              "price": 20.9,
              "promo": "Buy 1 Get 1",
              "packs_needed": 2
            }
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