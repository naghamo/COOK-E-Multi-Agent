"""
Agent 2: Recipe Retriever
RAG Agent for Recipe Retrieval
========================
This agent retrieves recipes based on user preferences and dietary restrictions.
It uses a vector database (Qdrant) to find relevant recipes and returns them in a structured format.
"""
import copy
# ------------------------------------------------------------------
# 0. Imports & Environment Setup
# ------------------------------------------------------------------
import os
import json
import ast
import re
import time
from typing import Optional, Dict, List, Union
import numpy as np
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"langchain(\.|$)")
warnings.filterwarnings("ignore", category=FutureWarning, message=r"`encoder_attention_mask` is deprecated")

# Load environment variables from .env
load_dotenv()

# Third-party libraries
from pydantic import BaseModel, Field
from langchain_community.chat_models import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool, StructuredTool
from langchain_core.embeddings import Embeddings
from langchain.schema import SystemMessage
from langchain.callbacks.manager import get_openai_callback
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models as qm

# Utility for token logging
from tokens.tokens_count import update_total_tokens

from openai import AzureOpenAI, RateLimitError
from langchain_openai import AzureOpenAIEmbeddings

# ------------------------------------------------------------------
# 1. Configuration Constants
# ------------------------------------------------------------------
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT        = os.getenv("AZURE_ENDPOINT", "https://096290-oai.openai.azure.com")
CHAT_DEPLOYMENT       = os.getenv("CHAT_DEPLOYMENT", "team10-gpt4o")
EMBED_DEPLOYMENT       = os.getenv("EMBED_DEPLOYMENT", "team10-embedding")
API_VERSION           = "2023-05-15"

# Qdrant vector store settings
QDRANT_URL       = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY   = os.getenv("QDRANT_API_KEY")
COLLECTION       = "recipes_openai"

# Embedding model and retrieval parameters
EMBED_MODEL      = "BAAI/bge-small-en-v1.5"
TOP_K            = 3

# ------------------------------------------------------------------
# 2. Embedding & Vector Store Initialization
# ------------------------------------------------------------------
# Connect to Qdrant vector database
client_qdr = QdrantClient(url=QDRANT_URL,
                          api_key=QDRANT_API_KEY,
                          prefer_grpc=True,
                          timeout=120,
                          )

# Azure client for embeddings
client_azr = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_ENDPOINT
)

# We'll use our wrapper, but it still requires embedder
embedder = AzureOpenAIEmbeddings(
    azure_deployment=EMBED_DEPLOYMENT,
    openai_api_version=API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)

vectorstore = Qdrant(
    client=client_qdr,
    collection_name=COLLECTION,
    embeddings=embedder,
    content_payload_key="title",
    metadata_payload_key="metadata",
)



# ------------------------------------------------------------------
# 3. Helper Functions for Filtering & Formatting
# ------------------------------------------------------------------
def build_filter(qp: Optional[Dict]) -> Optional[qm.Filter]:
    """
    Construct a Qdrant Filter from query_params:
      include keywords (match any vs all)
      exclude keywords
    """
    if not qp:
        return None

    include = [w.lower() for w in qp.get("include", [])]
    exclude = [w.lower() for w in qp.get("exclude", [])]
    mode    = qp.get("match_mode", "all").lower()

    must, must_not = [], []

    # Include logic: ANY vs ALL
    if include:
        if mode == "any":
            must.append(
                qm.FieldCondition(
                    key="keywords",
                    match=qm.MatchAny(any=include)
                )
            )
        else:  # all
            for kw in include:
                must.append(
                    qm.FieldCondition(
                        key="keywords",
                        match=qm.MatchAny(any=[kw])
                    )
                )

    # Exclude logic: always match any forbidden keyword
    if exclude:
        must_not.append(
            qm.FieldCondition(
                key="keywords",
                match=qm.MatchAny(any=exclude)
            )
        )

    # Return a Qdrant Filter if any conditions exist
    return qm.Filter(must=must, must_not=must_not) if (must or must_not) else None


def docs_to_json(hits) -> List[Dict]:
    """
    Convert Qdrant search hits into serializable JSON with payload + score
    """
    out = []
    for h in hits:
        out.append({
            "payload": h.payload,  # full recipe data
            "score":   float(h.score)
        })
    return out

# ------------------------------------------------------------------
# 4. Structured Search Tool Definition
# ------------------------------------------------------------------
class RecipeArgs(BaseModel):
    """
    Defines the arguments for searching recipes:
      - request_json: parser output as dict or JSON string
      - query_params: include/exclude keywords + match_mode
    """
    request_json: Union[Dict, str] = Field(
        ..., description="Parser output (dict) or serialized JSON string"
    )
    query_params: Optional[Dict] = Field(
        None, description="{'include':[str], 'exclude':[str], 'match_mode':'all'|'any'}"
    )

def get_embeddings_with_retry(texts, max_retries=3, filename='../tokens/total_tokens_embed.txt'):
    """Get embeddings/batch tokens with retry logic for rate limits"""
    if type(texts) is not list:
        texts = [texts]

    for retry in range(max_retries):
        try:
            response = client_azr.embeddings.create(
                input=texts,
                model=EMBED_DEPLOYMENT
            )

            vectors = [np.array(embedding.embedding).tolist() for embedding in response.data]
            batch_tokens = response.usage.total_tokens
            update_total_tokens(batch_tokens, filename)
            # print(f'Counted {batch_tokens} tokens!')

            if len(vectors) == 1:
                return vectors[0]

            return vectors
        except RateLimitError as e:
            if retry < max_retries - 1:
                wait_time = (2 ** retry) * 5  # 5, 10, 20 seconds
                print(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Max retries reached. Error: {e}")
                raise
        except Exception as e:
            print(f"Error on retry {retry + 1}: {e}")
            if retry < max_retries - 1:
                time.sleep(5)
            else:
                raise


def search_with_token_counting(query_text, query_params, max_retries=2):
    """Search with retry logic using your existing embedding function"""
    for attempt in range(max_retries + 1):
        try:
            # Use your existing function to get embedding with token counting
            query_embedding = get_embeddings_with_retry(
                query_text,
                max_retries=3,
                filename='../tokens/total_tokens_embed.txt'
            )

            # Perform search using the embedding
            hits = vectorstore.client.query_points(
                collection_name=COLLECTION,
                query=query_embedding,  # Use your embedding directly
                limit=TOP_K,
                query_filter=build_filter(query_params),
                with_payload=True,
                with_vectors=False,
            )

            # Return results - tokens already counted in get_embeddings_with_retry
            return json.dumps({"hits": docs_to_json(hits.points)}, ensure_ascii=False)

        except Exception as e:
            print(f"Search error (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                time.sleep((2 ** attempt))
            else:
                break

    return json.dumps({"hits": []}, ensure_ascii=False)


def format_hits_for_llm(hits_json):
    """Format search results in a more LLM-friendly way"""
    data = json.loads(hits_json)

    formatted_results = []
    for i, hit in enumerate(data["hits"], 1):
        payload = hit["payload"]
        formatted_recipe = f"""
        Recipe {i}: {payload["title"]} (score: {hit["score"]:.2f})
        'ingredients': {payload["ingredients"]}
        'directions'': {payload["directions"]}
        'keywords': {payload["keywords"]}
        'recipe_id': {payload["recipe_id"]}
        ---
        """

        formatted_results.append(formatted_recipe)

    return "\n".join(formatted_results)

def search_recipes(
    request_json: Union[Dict, str],
    query_params: Optional[Dict] = None
) -> str:
    """
    Query the vectorstore for candidate recipes.
    1. Deserialize parser output
    2. Build a text query
    3. Execute semantic search with optional filter
    4. Return top-K hits as JSON
    """
    # 1. Deserialize parser output (allow ast fallback)
    if isinstance(request_json, str):
        try:
            req = json.loads(request_json)
        except json.JSONDecodeError:
            req = ast.literal_eval(request_json)
    else:
        req = request_json

    # 2. Construct search text from food_name + special_requests
    pieces = [req.get("food_name", "")]
    if sr := req.get("special_requests"):
        pieces.append(sr)
    if not any(pieces):
        pieces.append(req.get("raw_text", ""))
    query_text = " ".join(pieces).strip()

    hits_json = search_with_token_counting(query_text=query_text, query_params=query_params)
    return format_hits_for_llm(hits_json)

# Wrap as a StructuredTool for the agent
get_recipes = StructuredTool.from_function(
    func=search_recipes,
    name="get_recipes",
    description=(
        "Retrieve candidate recipes. Args: request_json(str) + optional query_params"
    ),
    args_schema=RecipeArgs,
)

# ------------------------------------------------------------------
# 5. Planner LLM & System Prompt Setup
# ------------------------------------------------------------------
planner_llm = AzureChatOpenAI(
    azure_deployment   = CHAT_DEPLOYMENT,
    openai_api_key     = AZURE_OPENAI_API_KEY,
    azure_endpoint     = AZURE_ENDPOINT,
    openai_api_version = API_VERSION,
    openai_api_type    = "azure",
    temperature        = 0.0,
)

SYSTEM_PROMPT = SystemMessage(content="""
You are **COOK‑E**, an expert culinary assistant.

Input  
You receive ONE JSON object from an upstream parser agent. 

Your task is to generate a **suitable recipe** that matches the user's preferences and restrictions by making thoughtful modifications to retrieved recipes.

**MODIFICATION GUIDELINES:**

- **Ingredient Substitutions**: You may substitute ingredients to accommodate:
  - Dietary restrictions (vegan, vegetarian, keto, gluten-free, etc.)
  - Religious requirements (kosher, halal)
  - Allergies and intolerances (nuts, dairy, shellfish, etc.)
  - Any preferences listed in the request json (in raw_text too)

- **Substitution Examples**:
  - Kosher/Halal: Replace pork with beef, chicken, or turkey
  - Vegan: Use plant milk instead of dairy, aquafaba instead of eggs
  - Nut allergies: Replace peanuts with sunflower seeds or pumpkin seeds
  - Gluten-free: Use almond flour instead of wheat flour
  - Keto: Replace sugar with stevia, use cauliflower instead of rice

- **Ingredient Removal**: You may remove ingredients that the user specifically forbids (e.g., "no mushrooms"), but ONLY when:
  - The user explicitly requests the removal
  - You are confident the removal won't compromise the dish's core identity or quality
  - The ingredient isn't structurally essential (e.g., don't remove flour from bread)

- **Quality Preservation**: All modifications must:
  - Maintain the dish's fundamental character and appeal
  - Preserve cooking techniques and timing when possible
  - Use substitutions that work functionally in the recipe
  - Consider flavor balance and texture

- **Documentation**: Record ALL changes in the *notes* field, including:
  - What was substituted and why
  - What was removed and the reason
  - Any cooking adjustments needed due to changes
  - Warnings about texture or flavor differences

**IMPORTANT**: Only make modifications that are explicitly requested or clearly necessary for stated restrictions. Don't make unnecessary changes to perfectly suitable recipes.

Ignore logistics fields entirely, like:
• delivery / pickup  
• budget / price ceilings  
• people / servings counts  
These constraints are handled later in the pipeline; do not filter on them.

Single tool — get_recipes  
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
1. **Call#1:** Perform a broad search using `get_recipes` — **omit** `query_params` entirely.  
2. Inspect the hits.  
   • If they contain forbidden ingredients that can be simply omitted or swapped, **keep the hit and edit it** instead of excluding those ingredients in the filter.  
   • Otherwise, try a refined search by adding appropriate `include` and/or `exclude` keywords in `query_params` to narrow the results.  
3. If you find a candidate recipe but think it could be improved, you may make **one additional refinement call** (e.g., adjust filters or include extra keywords) to try and get a better match.  
4. You may make at most **3 total calls**. If you find a suitable recipe, or recipe with simple substitutions, work with it. Otherwise return the *Not feasible* JSON.

Return EXACTLY ONE of the following JSON schemas. Start the response with ```json and end it with ```.
Feasible recipe
{
  "feasible": true,
  "title":        str,
  "servings":     int,
  "ingredients":  [str],
  "directions":   [str],
  "notes":        str   // list EVERY substitution / allergy tweak you made to the recipe
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
• Prefer include/exclude on ingredients or keywords; adjust match_mode.  
• Finally, Output ONLY valid JSON, one of the structures with the fields above. No extra text.

""".strip())

# Initialize the agent with function-calling support
agent = initialize_agent(
    tools      = [get_recipes],
    llm        = planner_llm,
    agent      = AgentType.OPENAI_FUNCTIONS,
    agent_kwargs = {"system_message": SYSTEM_PROMPT},
    verbose    = True,
)
def extract_recipe_dict(agent_response: dict) -> dict:
    """
    Takes the agent's output dict and returns the parsed recipe dict.
    Expects the recipe JSON to be inside a markdown ```json ... ``` block.
    """
    output = agent_response.get('output', '')
    # Find code block (markdown triple-backtick, possibly with 'json')
    match = re.search(r"```json\s*(\{.*?\})\s*```", output, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not find recipe JSON in output.")
    recipe_json_str = match.group(1)
    # Parse JSON
    recipe_dict = json.loads(recipe_json_str)
    return recipe_dict
# ------------------------------------------------------------------
# 6. Agent Runner with Token Logging
# ------------------------------------------------------------------
def filter_request(data):
    filtered = copy.deepcopy(data)

    for key in ['delivery', 'budget', 'people']:
        if key in filtered:
            del filtered[key]

    return filtered

def retrieve_recipe(parsed_request, tokens_filename="../tokens/total_tokens.txt"):
    """
    Invoke the agent on parsed_request and log token usage.
    Returns the agent's JSON response as a string.
    """
    filtered_request = filter_request(parsed_request)

    with get_openai_callback() as cb:
        result = agent.invoke({"input": filtered_request})
        # print(
        #     f"\nTokens | prompt {cb.prompt_tokens}  "
        #     f"completion {cb.completion_tokens}  total {cb.total_tokens}"
        # )
        update_total_tokens(cb.total_tokens, filename=tokens_filename)
    if isinstance(result, str):
        return json.loads(result)
    return extract_recipe_dict(result)

# ------------------------------------------------------------------
# 7. CLI Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Example requests to test the pipeline
    req1 = {
        "food_name": "Vegan creamy pasta",
        "people": 5,
        "special_requests": "no mushrooms",
        "raw_text": "Hello, I want a Vegan creamy pasta for 5 people under 40₪, no mushrooms",
        "extra_fields": {},
        "error": None
    }

    req2 = {
        "food_name": "Peanut satay noodles",
        "people": 2,
        "special_requests": "no peanuts",
        "raw_text": "Peanut satay noodles without peanuts, You may switch the peanuts with something else instead",
        "extra_fields": {},
        "error": None
    }

    print("\n=== Request 1 ===")
    print(json.dumps(req1, indent=2, ensure_ascii=False))
    print("\nAgent response:\n", retrieve_recipe(req1))

    # print("\n=== Request 2 ===")
    # print(json.dumps(req2, indent=2, ensure_ascii=False))
    # print("\nAgent response:\n", retrieve_recipe(req2))