import tiktoken


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count the number of tokens in a text string using tiktoken.

    Args:
        text: The text to count tokens for
        encoding_name: The tiktoken encoding to use (default: cl100k_base for GPT-4)

    Returns:
        The number of tokens in the text
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))
