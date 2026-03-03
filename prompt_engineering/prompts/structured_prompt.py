def build_prompt(text: str) -> str:
    return f"""Classify the sentiment of the text.
Return only valid JSON with this exact schema:
{{
  "sentiment": "Positive|Negative|Neutral"
}}

Text: {text}"""
