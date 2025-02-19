import os

import matplotlib.pyplot as plt
import pandas as pd

dataset_path = "selfcheckgpttest.csv"
eval_dir = ""
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
    score = 0.0

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

selfcheck_precision = []
selfcheck_recall = []
selfcheck_f1_score = []
selfcheck_precision1 = []
selfcheck_recall1 = []
selfcheck_f1_score1 = []
selfcheck_precision2 = []
selfcheck_recall2 = []
selfcheck_f1_score2 = []

for threshold in thresholds:
    print(f"Threshold: {threshold}")

    selfcheck_scores = []
    for index, row in df.iterrows():
        halu = row["hallucination_check"]  # Replace this key with ground truths
        score_s = float(row["selfcheck_score_0"])
        selfcheck_score_0 = get_metric_score(score_s, threshold, halu)
        selfcheck_scores.append(selfcheck_score_0)

    precision, recall, f1_score = calculate_metrics(selfcheck_scores)
    print(f"SelfcheckGPT Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    print("================================================================")
    selfcheck_precision.append(precision)
    selfcheck_recall.append(recall)
    selfcheck_f1_score.append(f1_score)

    selfcheck_scores = []
    for index, row in df.iterrows():
        halu = row["hallucination_check"]  # Replace this key with ground truths
        score_s = float(row["selfcheck_score_5"])
        selfcheck_score_5 = get_metric_score(score_s, threshold, halu)
        selfcheck_scores.append(selfcheck_score_5)

    precision, recall, f1_score = calculate_metrics(selfcheck_scores)
    print(f"SelfcheckGPT Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    print("================================================================")
    selfcheck_precision1.append(precision)
    selfcheck_recall1.append(recall)
    selfcheck_f1_score1.append(f1_score)

    selfcheck_scores = []
    for index, row in df.iterrows():
        halu = row["hallucination_check"]  # Replace this key with ground truths
        score_s = float(row["selfcheck_score_10"])
        selfcheck_score_10 = get_metric_score(score_s, threshold, halu)
        selfcheck_scores.append(selfcheck_score_10)

    precision, recall, f1_score = calculate_metrics(selfcheck_scores)
    print(f"SelfcheckGPT Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    print("================================================================")
    selfcheck_precision2.append(precision)
    selfcheck_recall2.append(recall)
    selfcheck_f1_score2.append(f1_score)


fig = plt.figure(figsize=(24, 6))

# Plot F1 score curve
ax3 = fig.add_subplot(1, 3, 3)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.plot(thresholds, selfcheck_f1_score, label="T = 0.0")
ax3.plot(thresholds, selfcheck_f1_score1, label="T = 0.5")
ax3.plot(thresholds, selfcheck_f1_score2, label="T = 1.0")
ax3.set_xlabel("Threshold", fontsize=16)
ax3.set_ylabel("F1 Score", fontsize=16)
ax3.set_title("F1 Score Curve on Multiple Temperature", fontsize=16)
ax3.grid(False)

# Plot Precision for thresholds
ax1 = fig.add_subplot(1, 3, 1)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.plot(thresholds, selfcheck_precision, label="T = 0.0")
ax1.plot(thresholds, selfcheck_precision1, label="T = 0.5")
ax1.plot(thresholds, selfcheck_precision2, label="T = 1.0")
ax1.set_xlabel("Threshold", fontsize=16)
ax1.set_ylabel("Precision", fontsize=16)
ax1.set_title("Precision Curve on Multiple Temperature", fontsize=16)
ax1.grid(False)

# Plot Recall for thresholds
ax2 = fig.add_subplot(1, 3, 2)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.plot(thresholds, selfcheck_recall, label="T = 0.0")
ax2.plot(thresholds, selfcheck_recall1, label="T = 0.5")
ax2.plot(thresholds, selfcheck_recall2, label="T = 1.0")
ax2.set_xlabel("Threshold", fontsize=16)
ax2.set_ylabel("Recall", fontsize=16)
ax2.set_title("Recall Curve on Multiple Temperature", fontsize=16)
ax2.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=4, borderaxespad=0)
ax2.grid(False)

fig.savefig("temperature.pdf", dpi=300, bbox_inches='tight', pad_inches=0)


