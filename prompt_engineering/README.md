# Lab 4: Prompt Engineering and Synthetic Data (Groq API)

This project implements:
- Part 1: Prompt engineering experiments for sentiment classification
- Part 2: Synthetic dataset generation (balanced classes)
- Part 3: Baseline ML model evaluation

## Project Structure

```text
lab4_prompt_engineering/
├── prompts/
│   ├── zero_shot.py
│   ├── role_prompt.py
│   ├── few_shot.py
│   └── structured_prompt.py
├── logs/
│   └── .gitkeep
├── synthetic_data/
│   └── synthetic_reviews_sample.csv
├── ml/
│   └── baseline_model.py
├── main_experiment.py
├── generate_synthetic.py
├── utils.py
├── experimental_report_template.md
├── requirements.txt
└── .env.example
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

## Part 1: Prompt Engineering Experiment

Runs 4 strategies over 20 test inputs and repeats each input multiple times for consistency.

```bash
python main_experiment.py --consistency-runs 3 --temperature 0.0
```

Quick smoke test (faster):
```bash
python main_experiment.py --max-inputs 3 --consistency-runs 1 --temperature 0.0
```

Outputs:
- `logs/experiment_log_detailed.csv`
- `logs/experiment_summary.csv`

Logged fields include prompt, raw output, latency, tokens, format compliance, and consistency score.

## Part 2: Generate Synthetic Dataset

Generate at least 100 samples with balanced class distribution.

```bash
python generate_synthetic.py --samples 120
```

Outputs:
- `synthetic_data/synthetic_reviews.csv`
- `logs/synthetic_generation_log.json`

## Part 3: Train Baseline ML Model

```bash
python ml/baseline_model.py --data synthetic_data/synthetic_reviews.csv
```

Outputs:
- `ml/metrics.json`
- `ml/classification_report.txt`

## Notes for Submission

- Include screenshots of:
  - terminal run for experiment script
  - summary CSV and detailed log
  - synthetic dataset sample
  - model metrics output
- Use `experimental_report_template.md` to write your report.
