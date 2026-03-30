
# GradePal
**Rubric-Optimized LLM Grading with Feedback and Adversarial Robustness**

CSCI 544 Applied NLP — Team 20 | University of Southern California

> **Status report deadline: April 3, 2026**

---
## Teaam details
---

## Project Overview

GradePal is a multi-agent automatic short-answer grading (ASAG) system that:

1. **Grades** student answers using an LLM with an iteratively optimized rubric (GradeOpt-style)
2. **Generates feedback** explaining what the student got right and wrong
3. **Resists adversarial answers** — keyword-stuffed or fluent-but-wrong responses that try to trick the grader

We compare three grading systems:
- **System 1 — Baseline (no rubric):** LLM sees only the question and student answer
- **System 2 — Static rubric:** LLM also sees the reference answer/rubric
- **System 3 — GradeOpt:** Multi-agent loop (Grader → Reflector → Refiner) that iteratively improves the rubric

---

## Repository Structure

```
GradePal/
├── data/                  # EngSAF dataset CSVs (not committed — add locally)
│   ├── train.csv
│   ├── val.csv
│   ├── unseen_answers.csv
│   └── unseen_question.csv
├── graders/               # All three grading systems
│   ├── __init__.py
│   ├── baseline.py        # System 1 — no rubric (Dhyey)
│   ├── static_rubric.py   # System 2 — static rubric (Sharvari)
│   └── gradeopt/          # System 3 — GradeOpt pipeline (Keerthi + Prajaktha)
│       ├── __init__.py
│       ├── grader_agent.py
│       ├── reflector_agent.py
│       ├── refiner_agent.py
│       └── pipeline.py
├── evaluation/            # Evaluation scripts
├── adversarial/           # Adversarial subset generation (Poojasree + Keerthi)
├── rubrics/               # Auto-generated rubric JSONs (not committed)
├── results/               # Evaluation outputs (not committed)
├── utils/
│   ├── __init__.py
│   ├── llm.py             # Shared Gemini API wrapper — everyone imports from here
│   └── metrics.py         # Shared evaluation metrics — accuracy, QWK, F1
├── config.py              # All paths, model names, label maps — single source of truth
├── .env                   # Your personal API key — NEVER commit this
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Dataset — EngSAF

We use the **EngSAF dataset** for automatic short-answer grading with feedback.

| Split | Rows | Description |
|-------|------|-------------|
| `train.csv` | 3,662 | Training examples with labels and feedback |
| `val.csv` | 407 | Validation set |
| `unseen_answers.csv` | 980 | New student answers to seen questions |
| `unseen_question.csv` | 765 | Answers to entirely new questions |

**Schema (all 4 files have the same 6 columns):**

| Column | Type | Description |
|--------|------|-------------|
| `Question_id` | float | Unique question identifier |
| `Question` | str | The exam question |
| `Student Answer` | str | Student's response |
| `Correct Answer` | str | Reference/model answer |
| `output_label` | int | 0 = incorrect, 1 = partially correct, 2 = correct |
| `feedback` | str | Gold feedback comment explaining the label |

**Important:** The dataset files are not committed to GitHub. Copy them into the `data/` folder locally after cloning.

---

## Setup Instructions

### Prerequisites
- macOS or Linux
- [Anaconda](https://www.anaconda.com/download) installed
- A free Gemini API key from [aistudio.google.com](https://aistudio.google.com)

### Step 1 — Clone the repo
```bash
git clone https://github.com/your-org/GradePal.git
cd GradePal
```

### Step 2 — Create the conda environment
```bash
conda create -n gradepal python=3.11
conda activate gradepal
```

> **Every time you open a new terminal**, run `conda activate gradepal` before doing anything. Your terminal should show `(gradepal)` not `(base)`.

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Create your .env file
Create a file called `.env` in the root of the project:
```
GEMINI_API_KEY=your_key_here
```

Get your free API key at [aistudio.google.com](https://aistudio.google.com) → Get API Key → Create API Key. Takes 2 minutes.

> **Never commit your `.env` file.** It is in `.gitignore` and must stay local only.

### Step 5 — Add the dataset
Copy the EngSAF CSV files into the `data/` folder:
```
data/train.csv
data/val.csv
data/unseen_answers.csv
data/unseen_question.csv
```

### Step 6 — Test your setup
```bash
python test.py
```

You should see:
```
Train loaded: 3662 rows
API response: API working
```

### Step 7 — Set VS Code interpreter (if using VS Code)
1. Press `Cmd + Shift + P`
2. Type `Python: Select Interpreter`
3. Choose: `Python 3.11.x ('gradepal': conda)`

This removes yellow squiggly underlines on imports.

---

## LLM Configuration

We use **Google Gemini** via the `google-genai` package.

| Setting | Value |
|---------|-------|
| Primary model | `gemini-2.5-flash` |
| Fallback model | `gemini-2.5-pro` |
| Temperature (grading) | `0.0` — deterministic labels |
| Temperature (feedback) | `0.3–0.5` — more natural language |
| Free tier limit | ~15 requests/minute |

**Why Gemini?** The original GradeOpt paper used GPT-4. Using Gemini gives us a cross-model comparison angle — does rubric optimization transfer across model families? This is a genuine research contribution.

**Why `gemini-2.5-flash` specifically?** It's available on the free tier and fast. We tested `gemini-2.0-flash` (quota 0 on free tier) and `gemini-1.5-flash` (deprecated) before landing on this.

---

## Key Files Explained

### `utils/llm.py` — The shared API wrapper

**Everyone imports from here. Never call the Gemini API directly in your own files.**

```python
from utils.llm import call_llm, call_llm_json, batch_call
```

| Function | Use case | Returns |
|----------|----------|---------|
| `call_llm(prompt, system, ...)` | Single LLM call | `str` |
| `call_llm_json(prompt, system, ...)` | When you need structured JSON back | `dict` |
| `batch_call(prompts, delay=4.0)` | Running many examples with rate limiting | `list[str]` |

Key parameters:
- `temperature=0.0` for grading (always use this for labels — deterministic)
- `temperature=0.4` for feedback generation
- `max_tokens=20` for grading calls (label is very short — saves quota)
- `expect_json=True` strips markdown fences if Gemini wraps JSON in ```json blocks

**Why does this file exist?** If we all wrote our own API calls, switching models or adding retry logic would require changing 5+ files. With `llm.py`, you change one file and everything updates automatically.

### `utils/metrics.py` — The scoring referee

```python
from utils.metrics import evaluate, over_grading_rate
```

| Function | Use case |
|----------|----------|
| `evaluate(y_true, y_pred, split_name)` | Run all metrics, print summary, return dict |
| `over_grading_rate(y_true, y_pred)` | Adversarial evaluation only |

Metrics computed by `evaluate()`:
- **Accuracy** — fraction of exact label matches
- **Quadratic Weighted Kappa (QWK)** — most important metric, matches GradeOpt paper
- **Weighted F1** — accounts for class imbalance
- **Macro F1** — treats all three classes equally
- **Classification report** — per-label precision/recall/F1

**Why QWK?** Being one step off (predicting "partially correct" when true label is "correct") is penalized less than being two steps off. Makes more sense for ordinal grading than plain accuracy.

### `config.py` — Single source of truth for all settings

```python
from config import TRAIN_PATH, VAL_PATH, LABEL_MAP, LABEL_MAP_INV, BATCH_DELAY
```

Never hardcode file paths or label mappings in your own files. Always import from `config.py`.

| Constant | Value | Use |
|----------|-------|-----|
| `TRAIN_PATH` | `data/train.csv` | Load training data |
| `VAL_PATH` | `data/val.csv` | Load validation data |
| `LABEL_MAP` | `{0: "incorrect", 1: "partially correct", 2: "correct"}` | int → str for prompts |
| `LABEL_MAP_INV` | `{"incorrect": 0, ...}` | str → int for metrics |
| `BATCH_DELAY` | `4.0` | Seconds between API calls on free tier |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Current model |

---

## How the Three Systems Connect

```
EngSAF Dataset (train.csv, val.csv, unseen_answers.csv, unseen_question.csv)
        │
        ▼
rubrics/generate_rubrics.py  ──►  rubrics/rubrics.json  (one rubric per question)
        │
        ├──► graders/baseline.py          (System 1 — no rubric)
        │         └── uses: call_llm()
        │
        ├──► graders/static_rubric.py     (System 2 — static rubric)
        │         └── uses: call_llm(), rubrics.json
        │
        └──► graders/gradeopt/pipeline.py (System 3 — GradeOpt)
                  ├── grader_agent.py     grades with current rubric
                  ├── reflector_agent.py  analyzes errors, writes critique
                  ├── refiner_agent.py    updates rubric with adaptation rules
                  └── saves: rubrics/rubrics_optimized.json
                  └── uses: call_llm_json()
        │
        ▼
evaluation/  ──►  results/  ──►  Paper results table
        │
        └── utils/metrics.py  (same metrics for all three systems)
```

---

## Common Mistakes to Avoid

**Always activate gradepal before running anything:**
```bash
conda activate gradepal  # do this every time you open a terminal
```

**Never run pip install when you see (base):**
```bash
# WRONG — installs to base environment
(base) % pip install pandas

# RIGHT — activate gradepal first
(gradepal) % pip install pandas
```

**Never hardcode file paths:**
```python
# WRONG
train = pd.read_csv("data/train.csv")

# RIGHT
from config import TRAIN_PATH
train = pd.read_csv(TRAIN_PATH)
```

**Never call the Gemini API directly:**
```python
# WRONG
import google.generativeai as genai
model = genai.GenerativeModel("gemini-2.5-flash")

# RIGHT
from utils.llm import call_llm
response = call_llm(prompt, system="You are a grader.")
```

**Never commit .env or CSV files:**
```bash
# check what you're about to commit before every push
git status
git diff --cached
```

---

## Development Workflow

```bash
# start of every session
conda activate gradepal
cd path/to/GradePal

# before running on full dataset, always test on small sample first
python -c "
import pandas as pd
from config import VAL_PATH
val = pd.read_csv(VAL_PATH)
sample = val.head(5)
print(sample[['Question', 'Student Answer', 'output_label']].to_string())
"

# after making changes, commit and push
git add .
git commit -m "descriptive message about what you changed"
git push origin main
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `zsh: no such file or directory: pip` | Use `python -m pip install ...` instead |
| `ModuleNotFoundError: No module named 'pandas'` | You're in the wrong environment — run `conda activate gradepal` |
| `429 RESOURCE_EXHAUSTED` | Rate limit hit — wait a minute, or use a different Google account's API key |
| `404 NOT_FOUND` for model | Model name changed — check `list_models.py` for available models |
| Yellow squiggles in VS Code | Select the `gradepal` interpreter: `Cmd+Shift+P` → `Python: Select Interpreter` |
| `(base)` instead of `(gradepal)` | Run `conda activate gradepal` |

---

## Environment Details

| Tool | Version |
|------|---------|
| Python | 3.11.x |
| Conda env | `gradepal` |
| Gemini package | `google-genai` (NOT `google-generativeai` — that's deprecated) |
| Primary model | `gemini-2.5-flash` |

> **Note:** We migrated from `google-generativeai` to `google-genai` during setup. If you see `FutureWarning: All support for the google.generativeai package has ended` — you have the old package installed. Run `pip uninstall google-generativeai && pip install google-genai`.

---

## April 3 Status Report Checklist

- [ ] Dataset loaded and splits verified (Poojasree)
- [ ] Conda environment set up across all 5 machines
- [ ] `utils/llm.py` and `utils/metrics.py` working for everyone
- [ ] Baseline grader running with dev set metrics (Dhyey)
- [ ] Static rubric grader running with dev set metrics (Sharvari)
- [ ] Rubric generation script complete (Keerthi)
- [ ] GradeOpt loop running end-to-end for 1+ iterations (Keerthi + Prajaktha)
- [ ] Adversarial subset generated and labeled (Poojasree + Keerthi)
- [ ] Results folder has at least baseline + static rubric metrics logged

---

*GradePal — CSCI 544 Team 20 — University of Southern California*