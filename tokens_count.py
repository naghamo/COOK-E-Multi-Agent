def update_total_tokens(new_tokens, filename="data/total_tokens.txt"):
    #current total
    try:
        with open(filename, "r") as f:
            total = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        total = 0
    # Add new tokens and write back
    total += new_tokens
    with open(filename, "w") as f:
        f.write(str(total))
    print(f"Updated grand total tokens: {total}")
    return total


#use this in the pipeline to update the total tokens remember to change name
# from langchain.callbacks import get_openai_callback
#from tokens_count import update_total_tokens
# formatted_prompt = prompt_template.format(user_input=user_input)
# messages = [HumanMessage(content=formatted_prompt)]
# with get_openai_callback() as cb:
#
#     response = chat(messages=messages)
#     print("Prompt tokens:", cb.prompt_tokens)
#     print("Completion tokens:", cb.completion_tokens)
#     print("Total tokens (this run):", cb.total_tokens)
#     update_total_tokens(cb.total_tokens, filename="data/total_tokens_Nagham.txt")
