from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain
from langchain_openai import OpenAI
LANGSMITH_TRACING=True
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="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"
LANGSMITH_PROJECT="pr-upbeat-implement-92"
# OPENAI_API_KEY="<your-openai-api-key>"
#system prompt
llm = OpenAI(openai_api_key="sk-...")

template = """
You are an expert cooking assistant.
Given a user's free-text request, extract as much information as possible into a single valid JSON object, using the following fields:

- food_name: string. (Required. If not found, set to null and include a short error.)
- people: integer. (Default to 1 if not specified.)
- delivery: string. ("delivery", "pickup", or null; default to "delivery" if not specified.)
- special_requests: string or null.
- budget: string or null.
- raw_text: the original user input.
- extra_fields: dictionary for any other constraints or preferences (e.g., allergies, brands, supermarkets, dietary restrictions, tools, timing, etc.), using snake_case for keys.
- error: string, only if the input is ambiguous, not a food request, or if food_name is missing.

Instructions:
- If a field is missing, set to null (unless a default is specified).
- If food_name is missing, set error and do not proceed further.
- If the input is not a food request or is ambiguous, set error with a short message.
- Output ONLY valid JSON with the fields above. No extra text.

User Request: {user_input}
"""


prompt = PromptTemplate(input_variables=["user_input"], template=template)

llm = OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=256)
chain = LLMChain(prompt=prompt, llm=llm)

def parse_context(user_input):
    result = chain.run(user_input=user_input)
    print(result)
    #  validate and load JSON:
    import json
    try:
        data = json.loads(result)
    except Exception:
        data = {"error": "Could not parse JSON"}
    return data

# Example usage
if __name__ == "__main__":
    inp = "Vegan pizza for 5 under 40₪ with delivery, no mushrooms, with whole wheat"
    context = parse_context(inp)
    print(context)
