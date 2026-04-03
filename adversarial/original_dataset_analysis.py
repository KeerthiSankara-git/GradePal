import pandas as pd
import os
import matplotlib.pyplot as plt


# ---------------------------
# 1. Load Splits
# ---------------------------
def load_splits(base_path="./data"):
    splits = {}
    
    for split in ["train", "val", "unseen_answers", "unseen_question"]:
        file_path = os.path.join(base_path, f"{split}.csv")
        
        if os.path.exists(file_path):
            splits[split] = pd.read_csv(file_path)
            print(f"Loaded {split}: {splits[split].shape}")
        else:
            print(f"Warning: {file_path} not found")
    
    return splits

# ---------------------------
# 2. Schema
# ---------------------------
def print_schema(splits):
    print("\n===== SCHEMA =====")
    
    # Get first split safely
    name, df = next(iter(splits.items()))
    
    print(f"\n{name.upper()} columns:")
    print(df.columns.tolist())


# ---------------------------
# 3. Basic Overview per Split
# ---------------------------
def basic_overview(splits):
    for name, df in splits.items():
        print(f"\n===== {name.upper()} INFO =====")
        print(df.info())


# ---------------------------
# 4. Missing Values
# ---------------------------
def check_missing(splits):
    for name, df in splits.items():
        print(f"\n===== {name.upper()} MISSING VALUES =====")
        
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        summary = pd.DataFrame({
            "missing_count": missing,
            "missing_percent": missing_pct
        })
        
        print(summary.sort_values(by="missing_percent", ascending=False))


# ---------------------------
# 5. Sample Examples
# ---------------------------
def sample_examples(splits, n=2):
    for name, df in splits.items():
        print(f"\n===== {name.upper()} SAMPLES =====")
        
        for i in range(min(n, len(df))):
            print(f"\n--- Example {i+1} ---")
            for col in df.columns:
                print(f"{col}: {df.iloc[i][col]}")


# ---------------------------
# 6. Unique Counts
# ---------------------------
def unique_counts(splits):
    for name, df in splits.items():
        print(f"\n===== {name.upper()} SUMMARY =====")
        
        # Total number of rows
        print(f"Total rows: {len(df)}")
        
        print("\nUnique value counts:")
        for col in df.columns:
            print(f"{col}: {df[col].nunique()} unique values")


# ---------------------------
# 7. Label Distribution Analysis
# ---------------------------
def label_distribution_analysis(splits, label_col="output_label"):
    print("\n===== LABEL DISTRIBUTION =====")
    
    for name, df in splits.items():
        print(f"\n--- {name.upper()} ---")
        
        labels = df[label_col].dropna()
        
        # Count
        counts = labels.value_counts().sort_index()
        percentages = (counts / len(labels)) * 100
        
        summary = counts.to_frame(name="count")
        summary["percentage"] = percentages
        
        print(summary)
        
        # Plot
        fig, ax = plt.subplots()
        
        counts.plot(kind="bar", ax=ax)
        
        ax.set_title(f"{name.upper()} Label Distribution")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        
        # Save plot
        filename = f"{name.lower()}_label_distribution.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
        
        plt.show()
        plt.close(fig)

# ---------------------------
# 8. Per-Question Label Distribution Analysis
# ---------------------------
def per_question_distribution(df, question_col="Question", qid_col="Question_id", label_col="output_label"):
    print("\n===== PER-QUESTION LABEL DISTRIBUTION =====")
    
    # Group by Question_id + Question (so you keep readable text)
    grouped = (
        df.groupby([qid_col, question_col])[label_col]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    
    grouped.columns.name = None  
    grouped = grouped.rename(columns={
        0: "label_0",
        1: "label_1",
        2: "label_2"
    })
    
    # Compute imbalance score
    grouped["total"] = grouped[["label_0", "label_1", "label_2"]].sum(axis=1)
    grouped["imbalance"] = grouped[["label_0", "label_1", "label_2"]].max(axis=1) / grouped["total"]
    
    # Sort by imbalance (most skewed first)
    grouped = grouped.sort_values(by="imbalance", ascending=False)
    
    print("\nTop 25 most skewed questions:")
    print(grouped.head(25))
    
    return grouped

def plot_imbalance(grouped):
    
    # 1. Histogram
    fig, ax = plt.subplots()
    grouped["imbalance"].hist(ax=ax)
    ax.set_title("Per-Question Imbalance Distribution")
    ax.set_xlabel("Imbalance Ratio")
    ax.set_ylabel("Number of Questions")
    
    plt.savefig(os.path.join(OUTPUT_DIR, "imbalance_histogram.png"))
    plt.show()
    plt.close(fig)
    
    # 2. Top-K skewed
    top_k = 25
    top_skewed = grouped.head(top_k)
    
    fig, ax = plt.subplots()
    ax.barh(range(top_k), top_skewed["imbalance"])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_skewed["Question_id"].astype(str))
    ax.set_xlabel("Imbalance Ratio")
    ax.set_title("Top 25 Most Skewed Questions")
    ax.invert_yaxis()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "top25_skewed_questions.png"))
    plt.show()
    plt.close(fig)
    
    # 3. Stacked bar
    labels = ["label_0", "label_1", "label_2"]
    
    fig, ax = plt.subplots()
    top_skewed.set_index("Question_id")[labels].plot(
        kind="bar",
        stacked=True,
        ax=ax
    )
    ax.set_title("Label Distribution for Top Skewed Questions")
    ax.set_xlabel("Question_id")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "label_distribution_top25.png"))
    plt.show()
    plt.close(fig)

def imbalance_summary(grouped):
    print("\n===== IMBALANCE SUMMARY =====")
    total = len(grouped)
    
    balanced = (grouped["imbalance"] <= 0.6).sum()
    moderate = ((grouped["imbalance"] > 0.6) & (grouped["imbalance"] <= 0.75)).sum()
    high = (grouped["imbalance"] > 0.75).sum()
    
    print(f"Balanced (<=0.6): {balanced} ({balanced/total:.2%})")
    print(f"Moderate (0.6–0.75): {moderate} ({moderate/total:.2%})")
    print(f"High skew (>0.75): {high} ({high/total:.2%})")


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    base_path = "./data" 

    OUTPUT_DIR = "adversarial/results/original_dataset_analysis_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    splits = load_splits(base_path)
    
    basic_overview(splits)
    print_schema(splits)
    check_missing(splits)
    unique_counts(splits)
    sample_examples(splits)

    label_distribution_analysis(splits)
    
    if "train" in splits:
        grouped = per_question_distribution(splits["train"])
        plot_imbalance(grouped)
        imbalance_summary(grouped)