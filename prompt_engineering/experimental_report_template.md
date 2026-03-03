# Experimental Report: Prompt Engineering and Synthetic Data Generation

## 1. Introduction
- Briefly explain prompt engineering and its importance in controlling LLM behavior.
- State the objective: compare prompting strategies and evaluate synthetic data usefulness for ML.

## 2. Task Selection
- Selected NLP Task: **Sentiment Classification**
- Labels used: Positive, Negative, Neutral

## 3. Methodology

### 3.1 Model and API
- API: Groq Chat Completions API
- Model: `<your_model_name>`
- Date of experiments: `<date>`

### 3.2 Prompting Strategies
1. Zero-shot
2. Role-based
3. Few-shot (3 examples)
4. Structured output (JSON)

### 3.3 Experimental Setup
- Number of test inputs per strategy: 20
- Consistency runs per input: `<n>`
- Metrics captured:
  - Output text
  - Latency (seconds)
  - Token usage
  - Format compliance
  - Consistency score

## 4. Results

### 4.1 Prompt Strategy Comparison
Fill from `logs/experiment_summary.csv`.

| Strategy | Avg Latency (s) | Avg Tokens | Format Compliance | Avg Consistency |
|---|---:|---:|---:|---:|
| Zero-shot |  |  |  |  |
| Role-based |  |  |  |  |
| Few-shot |  |  |  |  |
| Structured |  |  |  |  |

### 4.2 Example Outputs
- Include 1-2 representative outputs per strategy from `logs/experiment_log_detailed.csv`.

### 4.3 Observations
- Which strategy gave highest consistency?
- Which had best format compliance?
- Token and latency trade-offs.

## 5. Synthetic Data Generation
- Total generated samples: `<count>`
- Class balance:
  - Positive: `<count>`
  - Negative: `<count>`
  - Neutral: `<count>`
- Mention any generation issues and how they were handled.

## 6. ML Evaluation
Baseline model: Logistic Regression with TF-IDF features.

Fill from `ml/metrics.json` and `ml/classification_report.txt`.

| Metric | Value |
|---|---:|
| Accuracy |  |
| Precision (weighted) |  |
| Recall (weighted) |  |
| F1-score (weighted) |  |

## 7. Discussion
- Did synthetic data help produce a usable classifier?
- Did model performance indicate overfitting, label simplicity, or limited diversity?
- Compare expected vs observed behavior of each prompt strategy.

## 8. Reflection and Limitations
- LLM outputs are probabilistic and may vary across runs.
- Structured prompts improve parseability but not necessarily correctness.
- Synthetic data may not represent real-world noise/domain variation.
- Potential hallucination/format drift in generation.

## 9. Conclusion
- Summarize key findings from prompt comparison and synthetic-data-based ML performance.

## 10. Appendix (Optional)
- Command history used
- Additional screenshots
- Sample rows from synthetic dataset
