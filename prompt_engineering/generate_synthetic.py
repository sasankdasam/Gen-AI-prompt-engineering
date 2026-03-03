import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

VALID_LABELS = ["Positive", "Negative", "Neutral"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic sentiment dataset via Groq")
    parser.add_argument("--model", default=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    parser.add_argument("--samples", type=int, default=120, help="Total samples (>=100 recommended)")
    parser.add_argument("--output", default="synthetic_data/synthetic_reviews.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_prompt(label: str, count: int) -> str:
    return f"""Generate exactly {count} realistic and diverse product reviews.
Each review must have sentiment "{label}".
Output only JSON array with this schema:
[
  {{"review": "text", "sentiment": "{label}"}}
]
No extra text."""


def request_reviews(client: Groq, model: str, label: str, count: int) -> list[dict]:
    prompt = build_prompt(label=label, count=count)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    raw = response.choices[0].message.content or "[]"
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Model response is not a JSON list.")

    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue
        review = str(item.get("review", "")).strip()
        sentiment = str(item.get("sentiment", "")).strip().title()
        if review and sentiment == label:
            cleaned.append({"review": review, "sentiment": sentiment})
    return cleaned


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY. Add it to .env or your environment.")

    args = parse_args()
    if args.samples < 100:
        raise ValueError("Use at least 100 samples for this lab.")

    client = Groq(api_key=api_key)
    random.seed(args.seed)

    base = args.samples // len(VALID_LABELS)
    remainder = args.samples % len(VALID_LABELS)
    targets = {label: base for label in VALID_LABELS}
    for i in range(remainder):
        targets[VALID_LABELS[i]] += 1

    all_rows: list[dict] = []
    for label in VALID_LABELS:
        rows = request_reviews(client, args.model, label, targets[label])
        if len(rows) < targets[label]:
            missing = targets[label] - len(rows)
            rows.extend(request_reviews(client, args.model, label, missing))
        all_rows.extend(rows[: targets[label]])

    random.shuffle(all_rows)
    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("No synthetic data produced.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    meta = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": args.model,
        "samples": int(df.shape[0]),
        "class_counts": df["sentiment"].value_counts().to_dict(),
        "output_file": str(output_path),
    }
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    with open(logs_dir / "synthetic_generation_log.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved synthetic dataset: {output_path}")
    print(f"Class distribution: {meta['class_counts']}")


if __name__ == "__main__":
    main()
