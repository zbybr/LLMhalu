import os

import matplotlib.pyplot as plt
import pandas as pd

dataset_path = "/home/mdafifal.mamun/research/LLMhalu/gpt3/data/gpt3_outputs_check_auto_truthfulqa1.1.csv"
eval_dir = "/home/mdafifal.mamun/research/LLMhalu/llama3/evaluation"
df = pd.read_csv(dataset_path, encoding="latin-1")


def get_metric_score(score, threshold, ground_truth):
    ground_truth = ground_truth.strip().lower()
    label = "yes" if score >= threshold else "no"

    tp = 1 if ground_truth == "yes" and label == "yes" else 0
    fp = 1 if ground_truth == "no" and label == "yes" else 0
    tn = 1 if ground_truth == "no" and label == "no" else 0
    fn = 1 if ground_truth == "yes" and label == "no" else 0

    return tp, fp, tn, fn


def calculate_mt_hallucination_score(row):
    score = 0

    weights = {"yes": 0.1, "no": 0.1, "not sure": 0.05}

    synonym_responses = row["synonym_responses"].split(";")

    for syn in synonym_responses:
        syn_l = syn.lower()
        if syn_l == "no":
            score += weights[syn_l]
        elif syn_l == "not sure":
            score += weights[syn_l]

    antonym_responses = row["antonym_responses"].split(";")

    for ant in antonym_responses:
        ant_l = ant.lower()
        if ant_l == "yes":
            score += weights[ant_l]
        elif ant_l == "not sure":
            score += weights[ant_l]

    return score


def calculate_metrics(data):
    TP = sum(row[0] for row in data)
    FP = sum(row[1] for row in data)
    TN = sum(row[2] for row in data)
    FN = sum(row[3] for row in data)

    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if precision + recall != 0
        else 0
    )

    return precision, recall, f1_score


thresholds = [i / 100 for i in range(101)]  # Thresholds from 0 to 1 with step 0.01

our_precision = []
our_recall = []
our_f1_score = []
selfcheck_precision = []
selfcheck_recall = []
selfcheck_f1_score = []

for threshold in thresholds:
    print(f"Threshold: {threshold}")
    our_scores = []
    selfcheck_scores = []
    for index, row in df.iterrows():
        score = calculate_mt_hallucination_score(row)
        print("MT Score:", score)
        halu = row["auto_hallucination_check"]  # Replace this key with ground truths
        score_s = float(row["selfcheck_score"])

        row_score = get_metric_score(score, threshold, halu)
        selfcheck_score = get_metric_score(score_s, threshold, halu)

        our_scores.append(row_score)
        selfcheck_scores.append(selfcheck_score)

    precision, recall, f1_score = calculate_metrics(our_scores)
    print(f"Our Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    our_precision.append(precision)
    our_recall.append(recall)
    our_f1_score.append(f1_score)

    precision, recall, f1_score = calculate_metrics(selfcheck_scores)
    print(f"Selfcheck Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    print("================================================================")
    selfcheck_precision.append(precision)
    selfcheck_recall.append(recall)
    selfcheck_f1_score.append(f1_score)

# Ensure the plot directory exists
os.makedirs(eval_dir, exist_ok=True)

# Plot precision-recall curve
plt.plot(our_recall, our_precision, label="Our Approach")
plt.plot(selfcheck_recall, selfcheck_precision, label="Selfcheck GPT")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(eval_dir, "precision_recall_curve.png"))
plt.close()

# Plot F1 score curve
plt.plot(thresholds, our_f1_score, label="Our Approach")
plt.plot(thresholds, selfcheck_f1_score, label="Selfcheck GPT")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("F1 Score Curve")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(eval_dir, "f1_score_curve.png"))
plt.close()

# Plot Precision for thresholds
plt.plot(thresholds, our_precision, label="Our Precision")
plt.plot(thresholds, selfcheck_precision, label="Selfcheck GPT")
plt.xlabel("Threshold")
plt.ylabel("Precision")
plt.title("Precision Curve")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(eval_dir, "precision_curve.png"))
plt.close()

# Plot Recall for thresholds
plt.plot(thresholds, our_recall, label="Our Recall")
plt.plot(thresholds, selfcheck_recall, label="Selfcheck GPT")
plt.xlabel("Threshold")
plt.ylabel("Recall")
plt.title("Recall Curve")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(eval_dir, "recall_curve.png"))
plt.close()

# Save outputs
output_df = pd.DataFrame(
    {
        "Threshold": thresholds,
        "Our Precision": our_precision,
        "Our Recall": our_recall,
        "Our F1": our_f1_score,
        "Selfcheck Precision": selfcheck_precision,
        "Selfcheck Recall": selfcheck_recall,
        "Selfcheck F1": selfcheck_f1_score,
    }
)

output_df.to_csv(os.path.join(eval_dir, "llama3_comparison.csv"), index=False)
