# GradePal
**Rubric-Optimized LLM Grading with Feedback and Adversarial Robustness**

CSCI 544 Applied NLP — Team 20 | University of Southern California

---

## Team Details

| Name | USC ID | Role |
|------|--------|------|
| Keerthi Sankaralingam | sankaral | Infrastructure, GradeOpt Pipeline |
| Prajaktha Nelamangala Jayakumar | pnelaman | Grader, Reflector, Refiner Agents |
| Dhyey Shah | dhyeyvsh | Baseline Grader |
| Sharvari Patil | sharvari | Static Rubric Grader |
| Poojasree Dwarakanath | pdwaraka | Dataset Analysis, Adversarial Evaluation |

---

## Project Overview

GradePal is a multi-agent automatic short-answer grading (ASAG) system that:

1. **Grades** student answers using an LLM with an iteratively optimized rubric (GradeOpt-style)
2. **Generates feedback** explaining what the student got right and wrong
3. **Evaluates robustness** against adversarial answers — keyword-stuffed or 
   fluent-but-wrong responses that try to trick the grader

We compare three grading systems:
- **System 1 — Baseline (no rubric):** LLM sees only the question and student answer
- **System 2 — Static Rubric:** LLM also sees the reference answer
- **System 3 — GradeOpt:** Multi-agent loop (Grader → Reflector → Refiner) 
  that iteratively improves per-question grading notes

---

## System and Device

| Component | Details |
|-----------|---------|
| OS | macOS (tested on macOS Sequoia) |
| Python | 3.11.x |
| Conda environment | `gradepal` |
| LLM | Google Gemini 2.5 Flash via `google-genai` package |
| API | Google AI Studio (paid tier, 10,000 RPD) |
| Runtime | GradeOpt full pipeline (~9 hours on 3,662 training examples, 3 iterations) |

> **Note:** All experiments were run on personal MacBook laptops using the 
> Gemini API. No GPU required — all inference is API-based.

---

## Repository Structure
GradePal/
├── data/                          # EngSAF dataset CSVs (not committed — add locally)
│   ├── train.csv
│   ├── val.csv
│   ├── unseen_answers.csv
│   ├── unseen_question.csv
│   └── adversarial_dataset_with_feedback.csv
├── graders/                       # All three grading systems
│   ├── baseline.py                # System 1 — no rubric (Dhyey)
│   ├── static_rubric.py           # System 2 — static rubric (Sharvari)
│   └── gradeopt/                  # System 3 — GradeOpt pipeline
│       ├── grader_agent.py        # Grades with current notes + generates feedback
│       ├── reflector_agent.py     # Analyzes errors, writes critique
│       ├── refiner_agent.py       # Updates grading notes from critique
│       └── pipeline.py            # Orchestrates full Grader→Reflector→Refiner loop
├── evaluation/                    # Evaluation scripts
│   ├── evaluate_unseen.py         # Evaluates GradeOpt on unseen splits
│   ├── bertscore_eval.py          # BERTScore feedback evaluation
│   └── adversarial_eval.py        # Adversarial robustness evaluation
├── adversarial/                   # Adversarial data generation and testing
│   ├── generate_adversarial_dataset.py
│   └── test_adversarial_data.py
├── results/                       # Evaluation outputs (not committed)
│   ├── gradeopt_metrics.json      # GradeOpt val metrics per iteration
│   ├── gradeopt_notes.json        # Optimized per-question grading notes
│   ├── gradeopt_unseen_metrics.json
│   ├── baseline_metrics.json
│   ├── bertscore_metrics.json
│   └── adversarial_metrics.json
├── utils/
│   ├── llm.py                     # Shared Gemini API wrapper
│   └── metrics.py                 # Shared evaluation metrics
├── config.py                      # All paths, model names, label maps
├── .env                           # API key — NEVER commit
├── requirements.txt
└── README.md
---

## Environment Setup

### Step 1 — Clone the repository
```bash
git clone https://github.com/your-org/GradePal.git
cd GradePal
```

### Step 2 — Create and activate conda environment
```bash
conda create -n gradepal python=3.11
conda activate gradepal
```

> Run `conda activate gradepal` every time you open a new terminal.

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Set up API key
Create a `.env` file in the project root:
GEMINI_API_KEY=your_gemini_api_key_here
Get a free API key at [aistudio.google.com](https://aistudio.google.com).

> **Never commit your `.env` file.**

### Step 5 — Add the dataset
Download the EngSAF dataset and place CSV files in the `data/` folder:
data/train.csv
data/val.csv
data/unseen_answers.csv
data/unseen_question.csv
data/adversarial_dataset_with_feedback.csv

### Step 6 — Verify setup
```bash
python -c "
import pandas as pd
from config import TRAIN_PATH, VAL_PATH
train = pd.read_csv(TRAIN_PATH)
val = pd.read_csv(VAL_PATH)
print(f'Train: {len(train)} rows')
print(f'Val: {len(val)} rows')
print('Setup complete!')
"
```

---

## Running the Code

### System 1 — Baseline Grader
```bash
# Run on all splits
python -m graders.baseline --split all

# Run on specific split
python -m graders.baseline --split val
python -m graders.baseline --split unseen_answers
python -m graders.baseline --split unseen_question
```
Results saved to `results/baseline_metrics.json`

### System 2 — Static Rubric Grader
```bash
python -m graders.static_rubric
```
Results saved to `results/system2_val_predictions.json`

### System 3 — GradeOpt Pipeline
```bash
# Full run — 3 iterations on all 3,662 training examples (~9 hours)
python -m graders.gradeopt.pipeline --iters 3

# Development mode — quick test on 50 examples
python -m graders.gradeopt.pipeline --iters 1 --sample 50

# Resume interrupted run automatically
python -m graders.gradeopt.pipeline --iters 3
# checkpoint saved after every row — safe to interrupt and resume
```
Results saved to:
- `results/gradeopt_metrics.json` — QWK per iteration
- `results/gradeopt_notes.json` — optimized per-question grading notes
- `results/gradeopt_val_iter{N}.csv` — predictions + feedback per iteration

### Unseen Split Evaluation
```bash
python -m evaluation.evaluate_unseen
```
Results saved to `results/gradeopt_unseen_metrics.json`

### BERTScore Feedback Evaluation
```bash
pip install bert-score
python -m evaluation.bertscore_eval
```
Results saved to `results/bertscore_metrics.json`

### Adversarial Robustness Evaluation
```bash
python -m adversarial.test_adversarial_data
```
Results saved to `results/adversarial_metrics.json`

---

## How Results Are Generated

All paper results are produced by running the scripts above in this order:

python -m graders.baseline --split all
→ results/baseline_metrics.json
python -m graders.static_rubric
→ results/system2_*_predictions.json
python -m graders.gradeopt.pipeline --iters 3
→ results/gradeopt_metrics.json
→ results/gradeopt_notes.json
python -m evaluation.evaluate_unseen
→ results/gradeopt_unseen_metrics.json
python -m evaluation.bertscore_eval
→ results/bertscore_metrics.json
python -m adversarial.test_adversarial_data
→ results/adversarial_metrics.json

### Key Results Summary

| System | Val QWK | Unseen-A QWK | Unseen-Q QWK |
|--------|---------|--------------|--------------|
| Baseline | 0.357 | 0.440 | 0.380 |
| Static Rubric | 0.609 | 0.664 | 0.548 |
| GradeOpt | 0.638 | 0.653 | 0.473 |

| Split | BERTScore F1 |
|-------|-------------|
| Val (best iter) | 0.8889 |
| Unseen Answers | 0.8886 |
| Unseen Questions | 0.8911 |

| Attack Type | Baseline OGR | Static Rubric OGR | GradeOpt OGR |
|-------------|-------------|-------------------|--------------|
| Keyword Stuffing | 97.5% | 95.0% | 90.0% |
| Fluent but Wrong | 32.5% | 10.0% | 25.0% |
| Off-topic | 0.0% | 0.0% | 0.0% |
| Flattery | 0.0% | 0.0% | 0.0% |

---

## Configuration

All settings are centralized in `config.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `GEMINI_MODEL` | `gemini-2.5-flash` | LLM used for all agents |
| `BATCH_DELAY` | `0.1` | Seconds between API calls |
| `TRAIN_PATH` | `data/train.csv` | Training data path |
| `VAL_PATH` | `data/val.csv` | Validation data path |
| `LABEL_MAP` | `{0: "incorrect", ...}` | Label integer to string |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `conda activate gradepal` first |
| `429 RESOURCE_EXHAUSTED` | Daily quota hit — wait until midnight PST or switch API key |
| `503 UNAVAILABLE` | Model overloaded — script retries automatically |
| `404 NOT_FOUND` for model | Check available models at aistudio.google.com |
| Pipeline interrupted | Just rerun — checkpoint resumes from last saved row |

---

*GradePal — CSCI 544 Team 20 — University of Southern California — Spring 2026*
