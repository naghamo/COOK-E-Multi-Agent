"""Agent 6: Product Matcher
---------------------------------------------------------------
This agent matches recipe ingredients to supermarket products,
in batches, using a GPT model.
It returns a list of products for each ingredient,
with details like cid, packs_needed, and discount.
"""


# ---------------------------------------------------------------

# 0. Imports & global data
# ---------------------------------------------------------------
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from langchain_core._api import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
import os, asyncio, concurrent.futures as cf, json, re, pint
from dataclasses import dataclass, asdict
from typing import List

import pandas as pd
from rapidfuzz import process, fuzz
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback

from tokens.tokens_count import update_total_tokens

from pydantic import BaseModel, Field
from typing import List
from pydantic import SecretStr
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------
# Azure LLM client
# ---------------------------------------------------------------
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
DEPLOYMENT_NAME = "team10-gpt4o"
AZURE_OPENAI_ENDPOINT = "https://096290-oai.openai.azure.com"
API_VERSION = "2023-05-15"

llm = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT_NAME,
    api_key=SecretStr(AZURE_OPENAI_API_KEY),
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=API_VERSION,
    openai_api_type="azure",
    temperature=0
)

# Maybe use strciter rules?
# sys_prompt = """
# You are a grocery shopping assistant. Your job is to select ONLY products that match the requested ingredients.
#
# CRITICAL MATCHING RULES:
# 1. DIETARY RESTRICTIONS ARE ABSOLUTE:
#    - If ingredient says "vegan", ONLY select vegan products (no dairy, no cheese, no cream unless explicitly vegan)
#    - "vegan sour cream" ≠ regular "sour cream"
#    - "vegan cheddar cheese" ≠ regular "cheddar cheese"
#
# 3. FORM/TYPE MATCHING:
#    - "refrigerated pizza dough" ≠ frozen pizza
#    - "raw broccoli florets" can include fresh or frozen broccoli (both are acceptable)
#    - "dressing mix" means powder/packet, NOT liquid dressing
#
# 4. SELECTION CRITERIA:
#    - Return empty picks [] if NO suitable candidates match
#    - Better to return nothing than wrong products
#    - Consider product name carefully - don't just match on keywords
#
# For each selected product:
# - Calculate packs_needed so total quantity ≥ to_buy_min
# - Extract discount from promo field:
#   * "10% off" → "10%"
#   * "5 NIS discount" → "5"
#   * "Buy 1 Get 1" → "50%"
#   * No promo → null
#
# Return ONLY valid JSON:
# [
#   {
#     "idx": 0,
#     "picks": [
#       {"cid": "c0", "packs_needed": 2, "discount": "10%"}
#     ]
#   }
# ]
# """

# ---------- shared catalog (loaded once) ----------

# Load product and supermarket databases
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

PRODUCTS_DF = pd.read_csv(os.path.join(DATA_DIR, "unit_productsDB.csv"))
SUPERMARKETS_DF = pd.read_csv(os.path.join(DATA_DIR, "supermarketsDB.csv"))

# ---------------------------------------------------------------
# 1. Helpers (for supermarket api)
# ---------------------------------------------------------------
ureg = pint.UnitRegistry()

# Expanded unit synonyms and forms
UNIT_SYNONYMS = {
    "ml": "milliliter", "milliliters": "milliliter",
    "l": "liter", "liters": "liter",
    "cl": "centiliter", "dl": "deciliter",
    "g": "gram", "grams": "gram", "gr": "gram",
    "kg": "kilogram", "kilograms": "kilogram",
    "mg": "milligram", "milligrams": "milligram",
    "oz": "ounce", "ounces": "ounce",
    "lb": "pound", "lbs": "pound", "pounds": "pound",
    "pc": "piece", "pcs": "piece", "pieces": "piece",
    "tab": "tablet", "tabs": "tablet", "tablets": "tablet",
    "pack": "pack", "packs": "pack",

    # Cooking-specific units
    "tsp": "teaspoon", "teaspoons": "teaspoon", "teasp": "teaspoon",
    "tbsp": "tablespoon", "tablespoons": "tablespoon", "tbsps": "tablespoon",
    "cup": "cup", "cups": "cup", "c": "cup",
    "fl oz": "fluid_ounce", "floz": "fluid_ounce", "fluid ounce": "fluid_ounce",

    # Common grocery units
    "each": "piece", "ea": "piece", "unit": "piece", "units": "piece",
    "can": "can", "cans": "can", "jar": "jar", "jars": "jar",
    "bottle": "bottle", "bottles": "bottle", "btl": "bottle",

    # Weight variations
    "gm": "gram", "grm": "gram", "kilo": "kilogram", "kilos": "kilogram",
}

# ---------------------------------------------------------------
# 2. Core classes
# ---------------------------------------------------------------
@dataclass
class Product:
    supermarket_id: str
    item_code: str
    name: str
    size: float
    unit: str
    price: float
    discount: str | None = None
    promo: str | None = None

def prod_to_dict(p) -> dict:
    return asdict(p)

class CsvSupermarket:
    products_df     = PRODUCTS_DF
    supermarkets_df = SUPERMARKETS_DF

    def __init__(self, market_id: str):
        self.market_id = market_id.lower()
        self.df = CsvSupermarket.products_df.loc[
            CsvSupermarket.products_df["supermarket"] == self.market_id
        ].copy()

        info_df = CsvSupermarket.supermarkets_df.loc[
            CsvSupermarket.supermarkets_df["supermarket"] == self.market_id
            ]
        if info_df.empty:
            self.meta_data = {"supermarket": self.market_id, "note": "metadata not found"}
        else:
            self.meta_data = info_df.iloc[0].to_dict()

    def search(self, need: dict, k: int = 5) -> List[Product]:
        query = need["name"].lower().strip()
        desired_unit = UNIT_SYNONYMS.get(need["to_buy_unit"].lower(),
                                         need["to_buy_unit"].lower())

        # --- 1. work on a VALUE list so RapidFuzz indices are positional ---
        choices = self.df["translated_itemname"].str.lower().tolist()

        # Check if there are any choices to search
        if not choices:
            return []

        matches = process.extract(
            query,
            choices,
            scorer=fuzz.token_set_ratio,  # better for multi-word overlap
            score_cutoff=70,  # a bit stricter
            limit=k
        )

        # Check if any matches were found
        if not matches:
            return []

        # Positional indices are now safe with .iloc
        subset = self.df.iloc[[m[2] for m in matches]].copy()

        # Check if subset is empty
        if subset.empty:
            return []

        # Trying to convert to desired units
        def convert_or_none(row):
            try:
                qty_conv = (row.parsed_qty * ureg(row.parsed_unit)).to(desired_unit).magnitude
                return qty_conv
            except (pint.DimensionalityError, pint.UndefinedUnitError, AttributeError, ValueError):
                return None

        # Use a more robust way to add the column
        subset.loc[:, "qty_conv"] = subset.apply(convert_or_none, axis=1)

        products = []
        for _, row in subset.head(k).iterrows():
            if pd.notna(row.get("qty_conv")):  # convertible
                size = row.qty_conv
                unit = need["to_buy_unit"]  # canonical user-unit
            else:  # keep original
                size = row.parsed_qty if pd.notna(row.parsed_qty) else 0
                unit = row.parsed_unit if pd.notna(row.parsed_unit) else "unit"

            products.append(Product(
                supermarket_id=self.market_id,
                item_code=str(row.itemcode),
                name=row.translated_itemname,
                size=size,
                unit=unit,
                price=row.itemprice if pd.notna(row.itemprice) else 0,
                promo=None if pd.isna(row.promo_desc) else row.promo_desc,
            ))

        return products

# ---------------------------------------------------------------
# 3. One batched LLM call  (cid + packs_needed)
# ---------------------------------------------------------------
BATCH_SIZE   = 3
N_CANDIDATES = 3

class Pick(BaseModel):
    cid: str                   # "c0"…
    packs_needed: int = Field(ge=1)
    discount: str = None  # Add discount field (e.g., "5", "10%")

class IngredientCall(BaseModel):
    idx: int                   # 0-based position inside the payload
    picks: List[Pick]

ingredient_list_schema = IngredientCall.model_json_schema()

INGREDIENT_FN = {
    "name": "ingredient_list",
    "description": "Array of IngredientCall objects",
    "parameters": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": ingredient_list_schema,
                "description": "One entry per ingredient in the same order"
            }
        },
        "required": ["items"]
    }
}


def gpt_pick_for_batch(
        needs: list[dict],
        cands: list[list[Product]],
        tokens_filename: str = "../tokens/total_tokens.txt"
) -> list[dict]:
    """
    Returns validated payload:
      [ { idx: 0, picks: [ {cid: "c1", packs_needed: 2, discount: "10%"}, … ] }, … ]
    """
    sys_prompt = """
        Choose candidate cids (zero or more) that match each ingredient name.
        Buy enough packs so total qty ≥ to_buy_min (handle unit conversion if necessary).

        IMPORTANT: Extract discount information from the promo field:
        - Look for discount patterns and convert them:
          * Money off (e.g., "5 NIS") → return as number: "5"
          * Percentage off (e.g., "10%", "20% off") → return as percentage: "10%", "20%"
        - Return discount ONLY as: a number ("5", "10") OR a percentage ("5%", "10%")
        - If promo is empty or no discount found in promo, set discount to null

        Return **only** JSON (example):

        [
          { "idx": 0,
            "picks":[{"cid":"c1","packs_needed":2,"discount":"10%"}]
          }
        ]
        """

    # * Multi-buy deals (e.g., "1+1", "2+1") → convert to percentage: "50%", "33%"
    #   AND adjust packs_needed accordingly (double for 1+1, etc.)

    payload = [
        {
            "idx": i,
            "ingredient_name": n["name"],
            "to_buy_min": n["to_buy_min"],
            "to_buy_unit": n["to_buy_unit"],
            "candidates": [
                {
                    "cid": f"c{j}",
                    "name": p.name,
                    "qty": p.size,
                    "unit": p.unit,
                    "promo": p.promo  # Include promo field for LLM to extract discount
                }
                for j, p in enumerate(pl)
            ],
        }
        for i, (n, pl) in enumerate(zip(needs, cands))
    ]

    msgs = [
        SystemMessage(sys_prompt),
        HumanMessage(json.dumps(payload, separators=(',', ':'), ensure_ascii=False)),
    ]

    with get_openai_callback() as cb:
        response = llm.invoke(
            msgs,
            functions=[INGREDIENT_FN],
            function_call={"name": "ingredient_list"}
        )

        # print(f"\nTokens | prompt {cb.prompt_tokens}  completion {cb.completion_tokens}  total {cb.total_tokens}")
        update_total_tokens(cb.total_tokens, filename=tokens_filename)

    # ---------- parse GPT output ----------
    try:
        raw_container = json.loads(
            response.additional_kwargs["function_call"]["arguments"]
        )
        raw = raw_container.get("items", [])
    except Exception as e:
        print(f"Warning: LLM response parsing failed: {e}")
        raw = [{"idx": i, "picks": []} for i in range(len(needs))]

    # ---------- cid → full-product dict with discount ----------
    cid_tables = [
        {f"c{j}": prod_to_dict(p) for j, p in enumerate(plist)}
        for plist in cands
    ]

    for obj in raw:
        i = obj["idx"]
        translated = []
        for pick in obj.get("picks", []):
            base = cid_tables[i].get(pick["cid"])
            if base:
                base = base.copy()
                base["packs_needed"] = int(pick["packs_needed"])
                base["discount"] = pick.get("discount")  # Add discount from LLM response
                translated.append(base)
        obj["picks"] = translated

    return raw


# ---------------------------------------------------------------
# 4. Worker for ONE store
# ---------------------------------------------------------------
def match_one_store(all_needs: list[dict], store_id: str,tokens_filename: str = "../tokens/total_tokens.txt") -> tuple[str, dict]:
    api = CsvSupermarket(store_id)
    cand_per_need = [api.search(n, N_CANDIDATES) for n in all_needs]

    store_meta = api.meta_data
    desired: dict[str, list[dict]] = {}

    for start in range(0, len(all_needs), BATCH_SIZE):
        needs_batch  = all_needs[start:start+BATCH_SIZE]
        cands_batch  = cand_per_need[start:start+BATCH_SIZE]

        gpt_out = gpt_pick_for_batch(needs_batch, cands_batch,tokens_filename=tokens_filename)

        for obj in gpt_out:
            global_idx = start + obj["idx"]
            ing_name = all_needs[global_idx]["name"]

            # obj["picks"] is already a list of full product dicts
            desired.setdefault(ing_name, []).extend(obj["picks"])

    return store_id, {
        "store_info": store_meta,
        "desired_ingredients": desired,
    }


# ---------------------------------------------------------------
# 5. Fan-out across supermarkets
# ---------------------------------------------------------------
SUPERMARKET_IDS = ['tiv_taam',
              'yohananof',
              'shufersal',
              'rami_levy',
              'victory',
              'osher_ad',
              'mega']
async def match_all_stores(ingredients: list[dict],tokens_filename: str = "../tokens/total_tokens.txt") -> dict:

    loop = asyncio.get_running_loop()
    with cf.ThreadPoolExecutor() as pool:
        tasks = [
            loop.run_in_executor(pool, match_one_store, ingredients, sid,tokens_filename)
            for sid in SUPERMARKET_IDS
        ]
        return dict(await asyncio.gather(*tasks))

# ---------------------------------------------------------------
# 5. Demo run
# ---------------------------------------------------------------
if __name__ == "__main__":
    ingredient_list = [
        {'name': 'olive oil', 'to_buy_min': 50, 'to_buy_unit': 'ml'},
        {'name': 'tomatoes', 'to_buy_min': 300, 'to_buy_unit': 'gr'},
        {'name': 'cheddar', 'to_buy_min': 250, 'to_buy_unit': 'gr'},
        {'name': 'salt', 'to_buy_min': 1, 'to_buy_unit': 'teaspoons'},
        {'name': 'garlic', 'to_buy_min': 3, 'to_buy_unit': 'cloves'},
        {'name': 'onion', 'to_buy_min': 5, 'to_buy_unit': 'units'},
        {'name': 'bread', 'to_buy_min': 1, 'to_buy_unit': 'loaf'},
        {'name': 'milk', 'to_buy_min': 1, 'to_buy_unit': 'liter'},
    ]

    ingredient_list_2 = [
        {'name': 'refrigerated pizza dough', 'to_buy_min': 0.625, 'to_buy_unit': 'can'},
        {'name': 'vegan sour cream', 'to_buy_min': 147.87, 'to_buy_unit': 'milliliter'},
        {'name': 'vegan ranch dressing mix', 'to_buy_min': 0.625, 'to_buy_unit': 'packet'},
        {'name': 'silken soft tofu', 'to_buy_min': 147.87, 'to_buy_unit': 'milliliter'},
        {'name': 'raw broccoli florets', 'to_buy_min': 0.0, 'to_buy_unit': 'milliliter'},
        {'name': 'raw cauliflower', 'to_buy_min': 0.0, 'to_buy_unit': 'milliliter'},
        {'name': 'raw carrot', 'to_buy_min': 147.87, 'to_buy_unit': 'milliliter'},
        {'name': 'raw yellow pepper', 'to_buy_min': 73.93, 'to_buy_unit': 'milliliter'},
        {'name': 'raw celery', 'to_buy_min': 73.93, 'to_buy_unit': 'milliliter'},
        {'name': 'cherry tomatoes', 'to_buy_min': 295.74, 'to_buy_unit': 'milliliter'},
        {'name': 'vegan cheddar cheese', 'to_buy_min': 147.87, 'to_buy_unit': 'milliliter'}
    ]

    matched = asyncio.run(match_all_stores(ingredient_list_2))
    print("Matched ingredients for each store: ")
    print(json.dumps(matched, indent=2, ensure_ascii=False))

