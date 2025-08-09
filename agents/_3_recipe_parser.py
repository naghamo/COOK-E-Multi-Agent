"""
Agent 3: Recipe Parser
--------------------------------------
This agent parses recipe ingredients using an LLM, scales them based on user request,
 and returns structured ingredient data in metric units.

"""
from __future__ import annotations

import os, ast, uuid, json
from typing import List, Dict, Any
from typing_extensions import TypedDict, Optional

from langchain_community.callbacks import get_openai_callback
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from tokens.tokens_count import update_total_tokens

from dotenv import load_dotenv
load_dotenv()

from tqdm import tqdm
from fractions import Fraction
from pint import UnitRegistry
from pint.errors import UndefinedUnitError

# ──────────────────────────────────────────────────────────────
# 1. Azure OpenAI LLM Initialization
# ──────────────────────────────────────────────────────────────
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
DEPLOYMENT_NAME = "team10-gpt4o"
AZURE_OPENAI_ENDPOINT = "https://096290-oai.openai.azure.com"
API_VERSION = "2023-05-15"

chat_llm = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version=API_VERSION,
    openai_api_type="azure",
    temperature=0
)

# ──────────────────────────────────────────────────────────────
# 2. Function Schemas for Ingredient Parsing
# ──────────────────────────────────────────────────────────────

# Single ingredient schema
ingredient_schema = {
    "name": "ingredient_item",
    "description": "Parse a single recipe ingredient line.",
    "parameters": {
        "type": "object",
        "properties": {
            "name":     {"type": "string"},
            "quantity": {"type": ["number", "null"]},
            "unit":     {"type": ["string", "null"]},
            "note":     {"type": ["string", "null"]}
        },
        "required": ["name", "quantity", "unit", "note"]
    },
}

# Batch ingredient list schema
ingredient_list_schema = {
    "name": "ingredient_list",
    "description": "Parse a whole list of recipe lines at once.",
    "parameters": {
        "type": "object",
        "properties": {
            "ingredients": {
                "type": "array",
                # Reuse the inner object schema for each item
                "items": ingredient_schema["parameters"],
            }
        },
        "required": ["ingredients"]
    },
}

# ──────────────────────────────────────────────────────────────
# 3. LLM Parsing Functions
# ──────────────────────────────────────────────────────────────

def parse_ingredients_llm(lines: List[str],tokens_filename="../tokens/total_tokens.txt") -> List[Dict[str, Any]]:
    """
    Parse a list of ingredient lines into structured JSON using the LLM.
    Returns a list of dicts with keys: name, quantity, unit, note.
    """
    # Number and join lines for clarity
    numbered = "\n".join(f"{i+1}. {ln}" for i, ln in enumerate(lines))

    # Build messages for the LLM
    msgs = [
        SystemMessage(content=(
            "You are **IngredientParser v1**, a specialist in parsing recipe lines. "
            "Your only output must be a single function call to `ingredient_list`, "
            "with its `ingredients` key set to a JSON array of objects—one object per input line."
            "Make sure you provide quantity and units as listed in the ingredient.   \n\n"

            "**Field requirements per ingredient object:**  \n"
            "• `name` (string): the ingredient name, lower‑cased, no quantity or unit.  \n"
            "• `quantity` (number): a decimal number; parse fractions (e.g. “½” or “1/2”) to floats. If missing, put 1. \n"
            "• `unit` (string): canonical units such as: teaspoon, tablespoon, cup, ounce, pound, gram, kilogram, milliliter, liter, pinch, piece, clove, slice, stick, bunch, units, and so on... (the way provided in the recipe). If missing, put `unit` instead.  \n"
            "• `note` (string|null): any additions, or trailing text (parentheses, descriptors like “minced”, “optional”), or null if none.  \n\n"

            "**Parsing rules:**  \n"
            "1. **Strict JSON**: do not emit any text outside the function call.  \n"
            "2. **Canonical units only**: if the line says “tbs.” or “Tbsp”, map to “tablespoon”; “g” or “grams” → “gram”; “oz” → “ounce”; etc.  \n"
            "3. **Fractions**: convert “¾” or “3/4” to 0.75.  \n"
            "4. **Missing fields**: if the quantity is missing, put 1. if unit is missing, put unit.  \n\n"
            
            "Input is numbered like “1. 2 Tbsp sugar”.  "
            "Produce exactly one call:\n"
            "`ingredient_list(ingredients=[{...}, {...}, …])`"
        )),
        HumanMessage(content=numbered)
    ]

    # Invoke LLM with function calling
    with get_openai_callback() as cb:
        response = chat_llm.invoke(
            msgs,
            functions=[ingredient_list_schema],
            function_call={"name": "ingredient_list"}  # force function call
        )
        # Log token usage
        # print(f"\nTokens | prompt {cb.prompt_tokens}  completion {cb.completion_tokens}  total {cb.total_tokens}")
        update_total_tokens(cb.total_tokens, filename=tokens_filename)

    # Extract and parse function arguments
    func_call = response.additional_kwargs["function_call"]
    data = json.loads(func_call["arguments"])
    return data["ingredients"]

# ──────────────────────────────────────────────────────────────
# 4. Quantity Scaling and Parsing Helpers
# ──────────────────────────────────────────────────────────────

def scale_qty(qty, unit: str, ratio: float) -> float | None:
    """
    Scale quantity by a given ratio, but skip scaling if unit is 'unit'
    (for ingredients without specific measurements like 'soy sauce').
    """
    if qty is None:
        return None

    # Skip scaling for 'unit' measurements - these are typically
    # ingredients added "to taste" or without specific quantities
    if unit == "unit":
        return qty

    return qty * ratio


def parse_quantity(q) -> float | None:
    """Convert string fractions (e.g. '1/2') to float, or pass through."""
    if q is None:
        return None
    if isinstance(q, str):
        try:
            return float(Fraction(q))
        except ValueError:
            pass
    return q

# ──────────────────────────────────────────────────────────────
# 5. Unit Registry and Definitions
# ──────────────────────────────────────────────────────────────

# Initialize pint's UnitRegistry
ureg = UnitRegistry()
# Define common cooking count units
for custom_unit in ["piece", "slice", "pinch", "clove", "bunch", "stick"]:
    try:
        ureg.define(f"{custom_unit} = [count]")
    except ValueError:
        pass  # skip if already defined

# Conversion thresholds
tsp_in_tbsp = 3
tbsp_in_cup = 16
oz_in_lb = 16

# ──────────────────────────────────────────────────────────────
# 6. Pretty-Printing Functions for Metric/US
# ──────────────────────────────────────────────────────────────

def _metric_pretty(q):
    """Format mass and volume into gram/kg or ml/l."""
    if q.check('[mass]'):
        return q.to('kilogram') if q.to('gram').magnitude >= 1000 else q.to('gram')
    if q.check('[volume]'):
        return q.to('liter')    if q.to('milliliter').magnitude >= 1000 else q.to('milliliter')
    return q


def _us_pretty(q):
    """Choose teaspoon/tablespoon/cup or ounce/pound based on quantity."""
    if q.check('[mass]'):
        q_oz = q.to('ounce')
        return q_oz.to('pound') if q_oz.magnitude >= oz_in_lb else q_oz
    if q.check('[volume]'):
        tsp = q.to('teaspoon')
        if tsp.magnitude < tsp_in_tbsp:
            return tsp
        tbsp = tsp.to('tablespoon')
        if tbsp.magnitude < tbsp_in_cup:
            return tbsp
        return tbsp.to('cup')
    return q

# ──────────────────────────────────────────────────────────────
# 7. Conversion Wrapper: Metric & US Output
# ──────────────────────────────────────────────────────────────

def convert_both(qty, unit: str) -> Dict[str, Any]:
    """
    Convert a quantity+unit into both pretty metric and US versions.
    Returns keys: metric_qty, metric_unit, us_qty, us_unit.
    """
    if qty is None or unit is None:
        return dict(metric_qty=1, metric_unit='unit',
                    us_qty=1,     us_unit='unit')

    # For 'unit' measurements, return as-is without conversion
    if unit == "unit":
        return dict(
            metric_qty=qty,
            metric_unit=unit,
            us_qty=qty,
            us_unit=unit
        )

    try:
        q = qty * ureg(unit)
    except UndefinedUnitError:
        # Fallback: return raw values if unit is unknown
        return dict(metric_qty=qty, metric_unit=unit,
                    us_qty=qty,     us_unit=unit)

    # Apply prettifier functions
    q_m  = _metric_pretty(q)
    q_us = _us_pretty(q)

    return dict(
        metric_qty=round(q_m.magnitude, 2),
        metric_unit=str(q_m.units),
        us_qty=round(q_us.magnitude, 2),
        us_unit=str(q_us.units)
    )
def get_scaled_ingredients_list(ingredients) -> List[dict]:
    """
    Extracts only the ingredient name, scaled quantity, and scaled unit for each ingredient.
    Returns: List of dicts with keys: name, quantity, unit
    """

    scaled_list = []
    for ing in ingredients:

        scaled = ing["scaled"]
        scaled_list.append({
            "name": ing["name"],
            "quantity": scaled["metric_qty"],
            "unit": scaled["metric_unit"]
        })
    return scaled_list

# ──────────────────────────────────────────────────────────────
# 8. Full Pipeline: Scale & Convert Ingredients
# ──────────────────────────────────────────────────────────────

def build_scaled_ingredient_list(user_req: Dict[str, Any], recipe: Dict[str, Any],tokens_filename) -> list[dict]:
    """
    Given a user request and recipe JSON, parse ingredients, scale by servings,
    and return both metric & US conversions for each ingredient.
    """
    # Calculate scaling ratio (requested servings / original servings)
    ratio = user_req.get("people", 1) / recipe.get("servings", 1)
    raw_lines = recipe["ingredients"]

    # Step 1: Parse raw ingredient lines via LLM
    parsed = parse_ingredients_llm(raw_lines,tokens_filename)

    result = []
    for ing in parsed:
        # Step 2: Parse and scale quantity
        original_qty = parse_quantity(ing["quantity"])
        qty_scaled   = scale_qty(original_qty, ing["unit"], ratio)

        # Step 3: Convert to metric & US
        dual = convert_both(qty_scaled, ing["unit"])

        # Collect structured output
        result.append({
            "name":   ing["name"],
            "note":   ing["note"],
            "orig":   {"qty": ing["quantity"], "unit": ing["unit"]},
            "scaled": dual
        })
    return get_scaled_ingredients_list(result)
    # Return full payload
    # return {
    #     "food_name":           recipe["title"],
    #     "servings_original":   recipe["servings"],
    #     "servings_requested":  user_req["people"],
    #     "scale_factor":        round(ratio, 3),
    #     "ingredients":         result
    # }


# ──────────────────────────────────────────────────────────────
# 9. CLI / Script Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example user request and recipe for quick testing
    user_request_json = {
        "food_name": "cold sesame noodles",
        "people": 6,
        "delivery": "pickup",
        "special_requests": "no peanuts",
        "budget": None,
        "raw_text": "Pickup cold sesame noodles for 6, no peanuts, metric units please",
        "extra_fields": {"allergies": "peanuts", "unit_system": "metric"},
        "error": None
    }

    recipe_json = {
        "title": "Cold Sesame Noodles with Peanut‑Free Satay Sauce",
        "servings": 4,
        "ingredients": [
            "1 pound Chinese egg noodles (or spaghetti or linguine)",
            "1/2 teaspoon toasted sesame oil",
            "6 tablespoons tahini (substitute for peanut butter)",
            "3/4 cup water, plus more if needed",
            "1 tablespoon rice vinegar",
            "3 tablespoons soy sauce",
            "1 1/2 teaspoons sugar",
            "2 garlic cloves, minced",
            "Scallions, thinly sliced",
            "1 tablespoon fresh ginger, chopped",
            "Chinese chili oil (optional)",
            "Sesame seeds (optional)"
        ],
        "directions": ["Cook noodles...", "Whisk sauce...", "Combine..."]
    }

    final_json = build_scaled_ingredient_list(user_request_json, recipe_json, tokens_filename="../tokens/total_tokens.txt")
    print(json.dumps(final_json, indent=2, ensure_ascii=False))