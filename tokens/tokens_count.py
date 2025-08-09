def update_total_tokens(new_tokens, filename="../tokens/total_tokens.txt"):
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
    # print(f"Updated grand total tokens: {total}")
    return total


#merge the tokens files
def merge_tokens_files(input_files, output_file="../tokens/total_tokens.txt"):
    """
    Merges multiple token count files into a single total count file.

    Args:
        input_files (list): List of input token files to merge.
        output_file (str): Path to the output file where the total will be saved.
    """
    total_tokens = 0
    for file in input_files:
        try:
            with open(file, "r") as f:
                tokens = int(f.read().strip())
                total_tokens += tokens
        except (FileNotFoundError, ValueError):
            print(f"Skipping invalid or missing file: {file}")

    with open(output_file, "w") as f:
        f.write(str(total_tokens))
    print(f"Total tokens merged into {output_file}: {total_tokens}")
if __name__ == "__main__":
    # Example usage for debugging and development
    input_files = [
        "total_tokens_Nagham.txt",
        "total_tokens_Seva.txt",
        "total_tokens.txt",
    ]
    merge_tokens_files(input_files, output_file="total_tokens1.txt")
    print("Tokens merged successfully.")