# ---------------------------------------------------------------
# 0. Imports & global data (unchanged)
# ---------------------------------------------------------------
import os, asyncio, concurrent.futures as cf, json, re, pint
from dataclasses import dataclass, asdict
from typing import List

import pandas as pd
from rapidfuzz import process, fuzz
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
from tokens.tokens_count import update_total_tokens

from pydantic import BaseModel, Field
from typing import List

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

# ---------- shared catalog (loaded once) ----------
PRODUCTS_DF     = pd.read_csv("../data/unit_productsDB.csv")
SUPERMARKETS_DF = pd.read_csv("../data/supermarketsDB.csv")

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
            # graceful fallback so the pipeline doesn't crash
            self.meta_data = {"supermarket": self.market_id, "note": "metadata not found"}
        else:
            self.meta_data = info_df.iloc[0].to_dict()

    def search(self, need: dict, k: int = 5) -> List[Product]:
        query = need["name"].lower().strip()
        desired_unit = UNIT_SYNONYMS.get(need["to_buy_unit"].lower(),
                                         need["to_buy_unit"].lower())

        # --- 1. work on a VALUE list so RapidFuzz indices are positional ---
        choices = self.df["translated_itemname"].str.lower().tolist()

        matches = process.extract(
            query,
            choices,
            scorer=fuzz.token_set_ratio,   # better for multi-word overlap
            score_cutoff=70,               # a bit stricter
            limit=k
        )

        # Positional indices are now safe with .iloc
        subset = self.df.iloc[[m[2] for m in matches]].copy()

        def convert_or_none(row):
            try:
                qty_conv = (row.parsed_qty * ureg(row.parsed_unit)).to(desired_unit).magnitude
                return qty_conv
            except (pint.DimensionalityError, pint.UndefinedUnitError):
                return None

        subset["qty_conv"] = subset.apply(convert_or_none, axis=1)

        products = []
        for _, row in subset.head(k).iterrows():
            if row.qty_conv is not None:  # convertible ✔
                size = row.qty_conv
                unit = need["to_buy_unit"]  # canonical user-unit
            else:  # keep original
                size = row.parsed_qty
                unit = row.parsed_unit

            products.append(Product(
                supermarket_id=self.market_id,
                item_code=str(row.itemcode),
                name=row.translated_itemname,
                size=size,
                unit=unit,
                price=row.itemprice,
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
    tokens_filename: str = "../tokens/total_tokens_Seva.txt"
) -> list[dict]:
    """
    Returns validated payload:
      [ { idx: 0, picks: [ {cid: "c1", packs_needed: 2}, … ] }, … ]
    """
    sys_prompt = """
        Choose candidate cids (zero or more) that match each ingredient name.
        Buy enough packs so total qty ≥ to_buy_min (handle unit conversion).
        Return **only** JSON (example):
        
        [
          { "idx": 0,
            "picks":[{"cid":"c1","packs_needed":2}]
          }
        ]
        """

    payload = [
        {
            "idx": i,
            "ingredient_name": n["name"],
            "to_buy_min": n["to_buy_min"],
            "to_buy_unit": n["to_buy_unit"],
            "candidates": [
                {"cid": f"c{j}", "name": p.name, "qty": p.size, "unit": p.unit}
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
            functions=[INGREDIENT_FN],  # hard schema
            function_call={"name": "ingredient_list"}  # force call
        )

        print(f"\nTokens | prompt {cb.prompt_tokens}  completion {cb.completion_tokens}  total {cb.total_tokens}")
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

    # ---------- cid  →  full-product dict ----------
    # Build one lookup table per ingredient
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
                translated.append(base)
        obj["picks"] = translated  # overwrite with full dicts

    return raw


# ---------------------------------------------------------------
# 4. Worker for ONE store
# ---------------------------------------------------------------
def match_one_store(all_needs: list[dict], store_id: str):
    api = CsvSupermarket(store_id)
    cand_per_need = [api.search(n, N_CANDIDATES) for n in all_needs]

    store_meta = api.meta_data
    desired: dict[str, list[dict]] = {}

    for start in range(0, len(all_needs), BATCH_SIZE):
        needs_batch  = all_needs[start:start+BATCH_SIZE]
        cands_batch  = cand_per_need[start:start+BATCH_SIZE]

        gpt_out = gpt_pick_for_batch(needs_batch, cands_batch)

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
async def match_all_stores(ingredients: list[dict]):
    loop = asyncio.get_running_loop()
    with cf.ThreadPoolExecutor() as pool:
        tasks = [
            loop.run_in_executor(pool, match_one_store, ingredients, sid)
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
    ]

    matched = asyncio.run(match_all_stores(ingredient_list))
    print("Matched ingredients for each store: ")
    print(json.dumps(matched, indent=2, ensure_ascii=False))

