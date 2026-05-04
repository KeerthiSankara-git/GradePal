# GradePal
**Rubric-Optimized LLM Grading with Feedback and Adversarial Robustness**

CSCI 544 Applied NLP — Team 20 | University of Southern California

---

## Team Details

| Name | USC email | Role |
|------|--------|------|
| Keerthi Sankaralingam | sankaral@usc.edu | Infrastructure, GradeOpt Pipeline |
| Prajaktha Nelamangala Jayakumar | pnelaman@usc.edu | Grader, Reflector, Refiner Agents |
| Dhyey Shah | dhyeyvsh@usc.edu | Baseline Grader |
| Sharvari Patil | sharvari@usc.edu | Static Rubric Grader |
| Poojasree Dwarakanath | pdwaraka@usc.edu | Dataset Analysis, Adversarial Evaluation |

---

## Project Overview

GradePal is a multi-agent automatic short-answer grading (ASAG) system that:

1. **Grades** student answers using an LLM with an iteratively optimized rubric (GradeOpt-style)
2. **Generates feedback** explaining what the student got right and wrong
3. **Evaluates robustness** against adversarial answers — keyword-stuffed or fluent-but-wrong responses that try to trick the grader

We compare three grading systems:
- **System 1 — Baseline (no rubric):** LLM sees only the question and student answer
- **System 2 — Static Rubric:** LLM also sees the reference answer
- **System 3 — GradeOpt:** Multi-agent loop (Grader → Reflector → Refiner) that iteratively improves per-question grading notes

---

## System and Device

| Component | Details |
|-----------|---------|
| OS | macOS (tested on macOS Sequoia) |
| Python | 3.11.x |
| Conda environment | `gradepal` |
| LLM | Google Gemini 2.5 Flash via `google-genai` package |
| API | Google AI Studio (paid tier, 10,000 RPD) |
| Runtime | GradeOpt full pipeline (~20 hours on 3,662 training examples, 3 iterations) |

> **Note:** All experiments were run on personal MacBook laptops using the Gemini API. No GPU required — all inference is API-based.

---

## Repository Structure

```
GradePal/
├── data/                          # EngSAF dataset CSVs (not committed — add locally)
│   ├── train.csv
│   ├── val.csv
│   ├── unseen_answers.csv
│   ├── unseen_question.csv
│   ├── adversarial_dataset.csv
|   └── adv_train/                 # Used only for gradeopt pipeline iterations with adversarial data 
|       ├── original_plus_adv_train.csv
|       └── Original_plus_adv_val.csv
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
|   └── split_adv_data.py
|   └── original_dataset_analysis.py
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
```


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
```
GEMINI_API_KEY=your_gemini_api_key_here
```

Get a free API key at [aistudio.google.com](https://aistudio.google.com).

> **Never commit your `.env` file.**

### Step 5 — Add the dataset
## Dataset — EngSAF

We use the **EngSAF dataset** for automatic short-answer grading with feedback.
The dataset is not publicly available — access must be requested directly from 
the authors.

### Requesting Access
Request access to EngSAF here:
**[EngSAF Dataset Access Form](https://docs.google.com/forms/d/e/1FAIpQLSdeRuvfE6b1Jhrhq6p7o_9dsjEvVWO7y9Eqmjun1R4tdWOeUg/viewform)**

If you need additional help to get access to the data faster, please contact one of the contributors of this project.

To get access to our generated adversarial data, request access in the link below (access already provided to Professors and TAs)
```bash
https://drive.google.com/drive/u/0/folders/1gooIgs8Hk1RwbHc7CvokgW_SHZj8njTU
```


Once approved, you will receive the following CSV files. Place them in the 
`data/` folder:
```
data/train.csv
data/val.csv
data/unseen_answers.csv
data/unseen_question.csv
data/adversarial_dataset.csv
```

### Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `Question_id` | float | Unique question identifier |
| `Question` | str | The exam question |
| `Student Answer` | str | Student's response |
| `Correct Answer` | str | Reference answer |
| `output_label` | int | 0 = incorrect, 1 = partially correct, 2 = correct |
| `feedback` | str | Gold feedback explaining the label |

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

### Original dataset analysis
```bash
python -m adversarial.original_dataset_analysis
```
Results saved to `adversarial/results/original_dataset_analysis_results`
This folder contains the graphs and plots showing our analysis of the original EngSAF dataset.


### System 1 — Baseline Grader
```bash
# Run on all splits
python -m graders.baseline --split all

# Run on a specific split
python -m graders.baseline --split val
python -m graders.baseline --split unseen_answers
python -m graders.baseline --split unseen_question
```
Results saved to `results/baseline_metrics.json`

---

### System 2 — Static Rubric Grader
```bash
python -m graders.static_rubric
```
Results saved to `results/system2_val_predictions.json`

---

### System 3 — GradeOpt Pipeline
```bash
# Full run — 3 iterations on all 3,662 training examples (~20 hours)
python -m graders.gradeopt.pipeline --iters 3

# Development mode — quick test on 50 examples
python -m graders.gradeopt.pipeline --iters 1 --sample 50

# Resume an interrupted run automatically
python -m graders.gradeopt.pipeline --iters 3
# Checkpoint is saved after every row — safe to interrupt and resume
```

Results saved to:
- `results/gradeopt_metrics.json` — QWK per iteration
- `results/gradeopt_notes.json` — optimized per-question grading notes
- `results/gradeopt_val_iter{N}.csv` — predictions + feedback per iteration

---

### Unseen Split Evaluation
```bash
python -m evaluation.evaluate_unseen
```
Results saved to `results/gradeopt_unseen_metrics.json`

---

### BERTScore Feedback Evaluation
```bash
pip install bert-score
python -m evaluation.bertscore_eval
```
Results saved to `results/bertscore_metrics.json`

---

### Adversarial Dataset Generation
These were the commands used to generate the adversarial data and get the splits. You do not have to run these commmands.
```bash
python -m adversarial.generate_adversarial_dataset
# Creating data split
python -m adversarial.split_adv_data
```
The results after running these commands are stored in the 2nd drive link (`https://drive.google.com/drive/u/0/folders/1gooIgs8Hk1RwbHc7CvokgW_SHZj8njTU`), and these files can directly be downloaded.


### Adversarial Robustness Evaluation
```bash
python -m adversarial.test_adversarial_data
```
Results saved to `results/adversarial_metrics.json`


### Adding Adversarial data to the Gradeopt Training pipeline
In order to test how Gradeopt performs when adversarial data is part of the grader pipeline, use the `/adv_train` folder fron the drive link 2, and add it to your data folder. Then run the gradeopt pipeline command using the following flags - 
```bash
python -m graders.gradeopt.pipeline --iters <number_of_iterations> --use_adv_train --run_name <name_for_saving_results>
# Example command:
# python -m graders.gradeopt.pipeline --iters 3 --use_adv_train --run_name gradeopt_adv 
```

---

## How Results Are Generated

All paper results are produced by running the scripts in this order:

```
Step 1: python -m graders.baseline --split all
        → results/baseline_metrics.json

Step 2: python -m graders.static_rubric
        → results/system2_*_predictions.json

Step 3: python -m graders.gradeopt.pipeline --iters 3
        → results/gradeopt_metrics.json
        → results/gradeopt_notes.json

Step 4: python -m evaluation.evaluate_unseen
        → results/gradeopt_unseen_metrics.json

Step 5: python -m evaluation.bertscore_eval
        → results/bertscore_metrics.json

Step 6: python -m adversarial.test_adversarial_data
        → results/adversarial_metrics.json
```

---

## Key Results Summary

### Grading Performance (QWK)

| System | Val | Unseen-A | Unseen-Q |
|--------|-----|----------|----------|
| Baseline | 0.357 | 0.440 | 0.380 |
| Static Rubric | 0.609 | 0.664 | 0.548 |
| GradeOpt | 0.638 | 0.653 | 0.473 |

### Feedback Quality (BERTScore F1)

| Split | BERTScore F1 |
|-------|-------------|
| Val Iter 1 | 0.8881 |
| Val Iter 2 | 0.8889 |
| Val Iter 3 | 0.8886 |
| Unseen Answers | 0.8886 |
| Unseen Questions | 0.8911 |

### Adversarial Robustness (Over-Grading Rate %)

| Attack Type | Baseline | Static Rubric | GradeOpt (no notes) | GradeOpt |
|-------------|----------|---------------|---------------------|----------|
| Keyword Stuffing | 97.5 | 95.0 | 92.5 | 90.0 |
| Fluent but Wrong | 32.5 | 10.0 | 25.0 | 25.0 |
| Off-topic | 0.0 | 0.0 | 0.0 | 0.0 |
| Flattery | 0.0 | 0.0 | 0.0 | 0.0 |

> Lower Over-Grading Rate = more robust system

---

## Configuration

All settings are centralized in `config.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `GEMINI_MODEL` | `gemini-2.5-flash` | LLM used for all agents |
| `BATCH_DELAY` | `0.1` | Seconds between API calls |
| `TRAIN_PATH` | `data/train.csv` | Training data path |
| `VAL_PATH` | `data/val.csv` | Validation data path |
| `UNSEEN_ANS_PATH` | `data/unseen_answers.csv` | Unseen answers path |
| `UNSEEN_Q_PATH` | `data/unseen_question.csv` | Unseen questions path |
| `LABEL_MAP` | `{0: "incorrect", 1: "partially correct", 2: "correct"}` | Label mapping |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `conda activate gradepal` first |
| `429 RESOURCE_EXHAUSTED` | Daily quota hit — wait until midnight PST or switch API key |
| `503 UNAVAILABLE` | Model temporarily overloaded — script retries automatically |
| `404 NOT_FOUND` for model | Check available models at aistudio.google.com |
| Pipeline interrupted | Just rerun — checkpoint resumes from last saved row automatically |
| `(base)` instead of `(gradepal)` | Run `conda activate gradepal` |

---

## References

1. **GradeOpt** — Chu, Y., Li, H., Yang, K., Shomer, H., Liu, H., Copur-Gencturk, Y., & Tang, J. (2025).
   *A LLM-Powered Automatic Grading Framework with Human-Level Guidelines Optimization.*
   In Proceedings of the 18th International Conference on Educational Data Mining (EDM 2025).
   https://arxiv.org/abs/2410.02165

2. **EngSAF** — Aggarwal, D., Sil, P., Raman, B., & Bhattacharyya, P. (2025).
   *"I understand why I got this grade": Automatic Short Answer Grading with Feedback.*
   arXiv preprint arXiv:2407.12818.
   https://arxiv.org/abs/2407.12818

3. **GradingAttack** — Li, X., Zhou, Z., Liu, Z., Wu, Y., & Luo, W. (2026).
   *GradingAttack: Attacking Large Language Models Towards Short Answer Grading Ability.*
   arXiv preprint arXiv:2602.00979.
   https://arxiv.org/abs/2602.00979

4. **LLM Evaluation** — Todorov, A., Klunder, E., & Belloni, J. E. (2025).
   *Evaluating the Potential of LLMs for Better Short Answer Scoring.*
   In Proceedings of the 17th International Conference on Computer Supported Education
   (CSEDU 2025), pages 108–119.
   https://doi.org/10.5220/0013291700003932

*GradePal — CSCI 544 Team 20 — University of Southern California — Spring 2026*
