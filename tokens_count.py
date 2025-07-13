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
#
# with get_openai_callback() as cb:
#     result = run_cooke_pipeline(user_text, inventory)
#     print("Tokens this run:", cb.total_tokens)
#
# # After the run, update the total:
# update_total_tokens(cb.total_tokens, filename="data/total_tokens_Nagham.txt")
