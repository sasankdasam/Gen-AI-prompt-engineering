import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from prompts.few_shot import build_prompt as few_shot_prompt
from prompts.role_prompt import build_prompt as role_prompt
from prompts.structured_prompt import build_prompt as structured_prompt
from prompts.zero_shot import build_prompt as zero_shot_prompt
from utils import compliance_for_strategy

TEST_INPUTS = [
    "I absolutely loved the movie!",
    "The product quality is terrible.",
    "It was okay, not great but not bad.",
    "Worst customer service ever.",
    "Amazing experience, would buy again!",
    "Delivery was late and packaging was damaged.",
    "Fantastic app interface.",
    "The food tasted horrible.",
    "Very satisfied with the purchase.",
    "Not worth the price.",
    "Exceeded my expectations.",
    "Battery life is disappointing.",
    "Super fast and reliable.",
    "Waste of money.",
    "Highly recommend this!",
    "It stopped working after one week.",
    "Pretty decent performance.",
    "Absolutely terrible design.",
    "Loved the color and build quality.",
    "Not happy with the service.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt engineering lab experiment runner")
    parser.add_argument("--model", default=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--consistency-runs", type=int, default=3)
    parser.add_argument("--output-dir", default="logs")
    parser.add_argument("--max-inputs", type=int, default=None, help="Use only first N inputs for quick tests")
    parser.add_argument("--timeout-sec", type=float, default=60.0, help="Per API call timeout in seconds")
    return parser.parse_args()


def run_single_completion(
    client: Groq, model: str, prompt: str, temperature: float, timeout_sec: float
) -> tuple[str, float, dict]:
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        timeout=timeout_sec,
    )
    latency_sec = time.perf_counter() - start

    usage = response.usage
    usage_dict = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    output = response.choices[0].message.content or ""
    return output, latency_sec, usage_dict


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY. Add it to .env or your environment.")

    args = parse_args()
    client = Groq(api_key=api_key)
    active_inputs = TEST_INPUTS[: args.max_inputs] if args.max_inputs else TEST_INPUTS

    strategies = {
        "Zero-Shot": zero_shot_prompt,
        "Role-Based": role_prompt,
        "Few-Shot": few_shot_prompt,
        "Structured": structured_prompt,
    }

    rows = []
    run_timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    total_calls = len(strategies) * len(active_inputs) * args.consistency_runs
    call_counter = 0
    print(
        f"Starting experiment: strategies={len(strategies)}, inputs={len(active_inputs)}, "
        f"runs={args.consistency_runs}, total_calls={total_calls}",
        flush=True,
    )

    for strategy_name, prompt_builder in strategies.items():
        for input_id, input_text in enumerate(active_inputs, start=1):
            normalized_outputs = []
            per_input_rows = []

            for run_id in range(1, args.consistency_runs + 1):
                prompt = prompt_builder(input_text)
                call_counter += 1
                print(
                    f"[{call_counter}/{total_calls}] {strategy_name} input={input_id} run={run_id}",
                    flush=True,
                )
                try:
                    output, latency, usage = run_single_completion(
                        client=client,
                        model=args.model,
                        prompt=prompt,
                        temperature=args.temperature,
                        timeout_sec=args.timeout_sec,
                    )
                except Exception as exc:
                    output = f"ERROR: {type(exc).__name__}: {exc}"
                    latency = 0.0
                    usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
                normalized_label, format_ok = compliance_for_strategy(strategy_name, output)

                row = {
                    "timestamp_utc": run_timestamp,
                    "model": args.model,
                    "strategy": strategy_name,
                    "input_id": input_id,
                    "input_text": input_text,
                    "run_id": run_id,
                    "prompt": prompt,
                    "raw_output": output,
                    "normalized_label": normalized_label,
                    "format_compliant": format_ok,
                    "latency_sec": latency,
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                }
                per_input_rows.append(row)
                normalized_outputs.append(normalized_label if normalized_label else "__invalid__")

            max_count = max(normalized_outputs.count(x) for x in set(normalized_outputs))
            consistency_score = max_count / len(normalized_outputs)
            for row in per_input_rows:
                row["consistency_score"] = consistency_score
                rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_df = pd.DataFrame(rows)
    detailed_path = output_dir / "experiment_log_detailed.csv"
    detailed_df.to_csv(detailed_path, index=False)

    summary_df = (
        detailed_df.groupby("strategy", as_index=False)
        .agg(
            avg_latency_sec=("latency_sec", "mean"),
            avg_total_tokens=("total_tokens", "mean"),
            format_compliance_rate=("format_compliant", "mean"),
            avg_consistency=("consistency_score", "mean"),
        )
        .sort_values("strategy")
    )
    summary_path = output_dir / "experiment_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved detailed logs to: {detailed_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
