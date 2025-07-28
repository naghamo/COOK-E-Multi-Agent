"""
Agent that
  • embeds queries locally (BAAI/bge‑small‑en‑v1.5)
  • LLM decides include/exclude keywords (ALL vs ANY)
  • returns a feasible recipe JSON or a structured error JSON
"""


import os, json, ast
from typing import Optional, Dict, List, Union
from dotenv import load_dotenv; load_dotenv()

from pydantic import BaseModel, Field
from langchain_community.chat_models import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.schema import SystemMessage
from langchain.callbacks.manager import get_openai_callback
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import StructuredTool

from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models as qm

from tokens.tokens_count import update_total_tokens    # <‑‑ your util

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "https://096290-oai.openai.azure.com")
CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT", "team10-gpt4o")
API_VERSION = "2023-05-15"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "recipes"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 3

# 1. Embedding & vector store -------------------------------------------------
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
client_qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
vectorstore = Qdrant(client=client_qdr,
                     collection_name=COLLECTION,
                     embeddings=embedder,
                     content_payload_key="title",
                     metadata_payload_key="metadata"
                     )

# 2. Helpers -----------------------------------------------------------------
def build_filter(qp: Optional[Dict]) -> Optional[qm.Filter]:
    """
    Build a Qdrant Filter.

    qp["match_mode"] == "any"  → at least one include keyword
    qp["match_mode"] == "all"  → every include keyword
    """
    if not qp:
        return None

    include = [w.lower() for w in qp.get("include", [])]
    exclude = [w.lower() for w in qp.get("exclude", [])]
    mode = qp.get("match_mode", "all").lower()

    must, must_not = [], []

    # -------- include keywords ----------
    if include:
        if mode == "any":
            # one condition with many options
            must.append(
                qm.FieldCondition(
                    key="keywords",
                    match=qm.MatchAny(any=include)
                )
            )
        else:  # "all"
            # one condition per ingredient
            for kw in include:
                must.append(
                    qm.FieldCondition(
                        key="keywords",
                        match=qm.MatchAny(any=[kw])
                    )
                )

    # -------- exclude keywords ----------
    if exclude:
        must_not.append(
            qm.FieldCondition(
                key="keywords",
                match=qm.MatchAny(any=exclude)
            )
        )

    return qm.Filter(must=must, must_not=must_not) if (must or must_not) else None

def docs_to_json(hits):
    out = []
    for h in hits:
        out.append({
            "payload": h.payload,  # <‑‑ full recipe data
            "score": float(h.score)
        })
    return out

# 3. Structured tool ---------------------------------------------------------
class RecipeArgs(BaseModel):
    request_json: Union[Dict, str] = Field(
        ...,
        description="Parser output (dict) or the same serialized as JSON string"
    )
    query_params: Optional[Dict] = Field(
        None,
        description="{include:[str], exclude:[str], match_mode:'all'|'any'}"
    )

def search_recipes(request_json: Union[Dict, str],
                   query_params: Optional[Dict] = None) -> str:
    if isinstance(request_json, str):
        # tolerate single‑quoted pseudo‑JSON by replacing with double quotes fallback
        try:
            req = json.loads(request_json)
        except json.JSONDecodeError:
            req = ast.literal_eval(request_json)
    else:
        req = request_json

    pieces = [req.get("food_name", "")]
    if sr := req.get("special_requests"):
        pieces.append(sr)
    if not pieces:
        pieces.append(req.get("raw_text", ""))
    query_text = " ".join(pieces).strip()

    hits = vectorstore.client.search(
        collection_name=COLLECTION,
        query_vector=vectorstore._embed_query(query_text),
        limit=TOP_K,
        query_filter=build_filter(query_params)
    )
    return json.dumps({"hits": docs_to_json(hits)}, ensure_ascii=False)

get_recipes = StructuredTool.from_function(
    func=search_recipes,
    name="get_recipes",
    description=(
        "Retrieve candidate recipes.\n"
        "Args: request_json (str, required) and optional query_params dict "
        "with include / exclude / match_mode"
    ),
    args_schema=RecipeArgs,
)

# 4. Planner LLM + system prompt --------------------------------------------
planner_llm = AzureChatOpenAI(
    azure_deployment   = CHAT_DEPLOYMENT,
    openai_api_key     = AZURE_OPENAI_API_KEY,
    azure_endpoint     = AZURE_ENDPOINT,
    openai_api_version = API_VERSION,
    openai_api_type    = "azure",
    temperature        = 0.3,
)

SYSTEM_PROMPT = SystemMessage(content="""
You are **COOK‑E**, an expert culinary assistant.

Input  
You receive ONE JSON object from an upstream parser agent. 

Your task is to generate a **suitable recipe** that matches the user’s
preferences.  
• You may alter, add ingredients or make substitutions to satisfy dietary
  restrictions or other constraints
• You may adapt a retrieved recipe by **deleting ingredients** that the user forbids 
(e.g., remove “mushrooms” from a pizza topping list), but only do so when you are confident the change preserves the dish’s quality and users request.  
• Document every substitution and change you made in the _notes_ field of your final output.


Ignore logistics fields
• delivery / pickup  
• budget / price ceilings  
• people / servings counts  
These constraints are handled later in the pipeline; do not filter on them.

Single tool — `get_recipes`  
Args schema:
{
  "request_json": <parser JSON string>,     // required  (ALWAYS include)
  "query_params": {                         // optional  (see below)
      "include":    [str],                  // keywords to INCLUDE
      "exclude":    [str],                  // keywords to EXCLUDE
      "match_mode": "all" | "any"           // default "all"
  }
}

Tool‑use rules  (MAX 3 calls)  
1. **Call#1:** broad search — omit `query_params` entirely.  
2. Inspect the hits. If they contain forbidden ingredients that can be
    simply omitted or swapped, **keep the hit and edit it** instead of
    excluding those ingredients in the filter.
3. Use `exclude` keywords **only** when the ingredient is fundamental or poses a strict allergy risk.
4. At most 3 total calls; otherwise return the *Not feasible* JSON.

Return EXACTLY ONE of

Feasible recipe
{
  "feasible": true,
  "title":        str,
  "servings":     int,
  "ingredients":  [str],
  "directions":   [str],
  "notes":        str   // list EVERY substitution / allergy tweak
}

Not feasible
{
  "feasible": false,
  "reason":      str,
  "violations":  [str],
  "suggestions": [str]
}

Workflow summary  
• Start with broad search.  
• Refine only when necessary (≤2 extra calls).  
• Prefer `include`/`exclude` on ingredients or keywords; adjust `match_mode`.  
• Finally, output ONE of the JSON structures above, nothing else.
""".strip())

agent = initialize_agent(
    tools=[get_recipes],
    llm=planner_llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    agent_kwargs={"system_message": SYSTEM_PROMPT},
    verbose=True,
)

# 5. Wrapper with token logging ---------------------------------------------
def run_agent(parsed_request: dict) -> str:
    with get_openai_callback() as cb:
        result = agent.invoke({"input": parsed_request})   # LC JSON‑serialises automatically
        print(f"\nTokens | prompt {cb.prompt_tokens}  "
              f"completion {cb.completion_tokens}  total {cb.total_tokens}")
        update_total_tokens(cb.total_tokens,
                            filename="../tokens/total_tokens_Seva.txt")
    return result

# 6. CLI demo ----------------------------------------------------------------
if __name__ == "__main__":
    req1 = {
        "food_name": "Vegan pizza",
        "people": 5,
        "special_requests": "no mushrooms",
        "raw_text": "Vegan pizza for 5 under 40₪, no mushrooms",
        "extra_fields": {},
        "error": None
    }
    req2 = {
        "food_name": "Peanut satay noodles",
        "people": 2,
        "special_requests": "no peanuts, must be peanut satay sauce",
        "raw_text": "Peanut satay noodles without peanuts",
        "extra_fields": {},
        "error": None
    }

    print("\n=== Feasible Request ===")
    print(json.dumps(req1, indent=2, ensure_ascii=False))
    print("\nAgent response:\n", run_agent(req1))

    print("\n=== Impossible Request ===")
    print(json.dumps(req2, indent=2, ensure_ascii=False))
    print("\nAgent response:\n", run_agent(req2))
