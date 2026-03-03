def build_prompt(text: str) -> str:
    return f"""Classify the sentiment of the following text as one label only:
Positive, Negative, or Neutral.

Text: {text}
Answer with just the label."""
