def build_prompt(text: str) -> str:
    return f"""You are a strict sentiment analysis evaluator for an NLP lab.
Classify the text into exactly one label from:
Positive, Negative, Neutral.
Do not explain.

Text: {text}
Label:"""
