def build_prompt(text: str) -> str:
    return f"""Classify sentiment using one label: Positive, Negative, or Neutral.

Example 1:
Text: I love this product.
Sentiment: Positive

Example 2:
Text: This is the worst experience ever.
Sentiment: Negative

Example 3:
Text: It is okay, nothing special.
Sentiment: Neutral

Now classify:
Text: {text}
Sentiment:"""
