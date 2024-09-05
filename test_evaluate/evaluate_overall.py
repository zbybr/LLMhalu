import os

import matplotlib.pyplot as plt
import pandas as pd

dataset_path = "gpt4_outputs_overall.csv"
dataset_path1 = "gpt3_outputs_overall.csv"
dataset_path2 = "llama3_outputs_overall.csv"
dataset_path3 = "mistral_outputs_overall.csv"
eval_dir = ""
df = pd.read_csv(dataset_path, encoding="latin-1")
df1 = pd.read_csv(dataset_path1, encoding="latin-1")
df2 = pd.read_csv(dataset_path2, encoding="latin-1")
df3 = pd.read_csv(dataset_path3, encoding="latin-1")

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

our_precision = []
our_recall = []
our_f1_score = []
selfcheck_precision = []
selfcheck_recall = []
selfcheck_f1_score = []
our_precision1 = []
our_recall1 = []
our_f1_score1 = []
selfcheck_precision1 = []
selfcheck_recall1 = []
selfcheck_f1_score1 = []
our_precision2 = []
our_recall2 = []
our_f1_score2 = []
selfcheck_precision2 = []
selfcheck_recall2 = []
selfcheck_f1_score2 = []
our_precision3 = []
our_recall3 = []
our_f1_score3 = []
selfcheck_precision3 = []
selfcheck_recall3 = []
selfcheck_f1_score3 = []

for threshold in thresholds:
    print(f"Threshold: {threshold}")

    our_scores = []
    selfcheck_scores = []
    for index, row in df.iterrows():
        score = calculate_mt_hallucination_score(row)
        print("MT Score:", score)

        halu = row["hallucination_check"]  # Replace this key with ground truths
        score_s = float(row["selfcheck_score"])

        row_score = get_metric_score(score, threshold, halu)
        selfcheck_score = get_metric_score(score_s, threshold, halu)

        our_scores.append(row_score)
        selfcheck_scores.append(selfcheck_score)

    precision, recall, f1_score = calculate_metrics(our_scores)
    print(f"MetaQA Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    our_precision.append(precision)
    our_recall.append(recall)
    our_f1_score.append(f1_score)

    precision, recall, f1_score = calculate_metrics(selfcheck_scores)
    print(f"SelfcheckGPT Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    print("================================================================")
    selfcheck_precision.append(precision)
    selfcheck_recall.append(recall)
    selfcheck_f1_score.append(f1_score)

    our_scores1 = []
    selfcheck_scores1 = []
    for index, row in df1.iterrows():
        score = calculate_mt_hallucination_score(row)
        print("MT Score:", score)

        halu = row["hallucination_check"]  # Replace this key with ground truths
        score_s = float(row["selfcheck_score"])

        row_score = get_metric_score(score, threshold, halu)
        selfcheck_score = get_metric_score(score_s, threshold, halu)

        our_scores1.append(row_score)
        selfcheck_scores1.append(selfcheck_score)

    precision, recall, f1_score = calculate_metrics(our_scores1)
    print(f"MetaQA Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    our_precision1.append(precision)
    our_recall1.append(recall)
    our_f1_score1.append(f1_score)

    precision, recall, f1_score = calculate_metrics(selfcheck_scores1)
    print(f"SelfcheckGPT Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    print("================================================================")
    selfcheck_precision1.append(precision)
    selfcheck_recall1.append(recall)
    selfcheck_f1_score1.append(f1_score)

    our_scores2 = []
    selfcheck_scores2 = []
    for index, row in df2.iterrows():
        score = calculate_mt_hallucination_score(row)
        print("MT Score:", score)

        halu = row["hallucination_check"]  # Replace this key with ground truths
        score_s = float(row["selfcheck_score"])

        row_score = get_metric_score(score, threshold, halu)
        selfcheck_score = get_metric_score(score_s, threshold, halu)

        our_scores2.append(row_score)
        selfcheck_scores2.append(selfcheck_score)

    precision, recall, f1_score = calculate_metrics(our_scores2)
    print(f"MetaQA Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    our_precision2.append(precision)
    our_recall2.append(recall)
    our_f1_score2.append(f1_score)

    precision, recall, f1_score = calculate_metrics(selfcheck_scores2)
    print(f"SelfcheckGPT Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    print("================================================================")
    selfcheck_precision2.append(precision)
    selfcheck_recall2.append(recall)
    selfcheck_f1_score2.append(f1_score)

    our_scores3 = []
    selfcheck_scores3 = []
    for index, row in df3.iterrows():
        score = calculate_mt_hallucination_score(row)
        print("MT Score:", score)

        halu = row["hallucination_check"]  # Replace this key with ground truths
        score_s = float(row["selfcheck_score"])

        row_score = get_metric_score(score, threshold, halu)
        selfcheck_score = get_metric_score(score_s, threshold, halu)

        our_scores3.append(row_score)
        selfcheck_scores3.append(selfcheck_score)

    precision, recall, f1_score = calculate_metrics(our_scores3)
    print(f"MetaQA Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    our_precision3.append(precision)
    our_recall3.append(recall)
    our_f1_score3.append(f1_score)

    precision, recall, f1_score = calculate_metrics(selfcheck_scores3)
    print(f"SelfcheckGPT Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    print("================================================================")
    selfcheck_precision3.append(precision)
    selfcheck_recall3.append(recall)
    selfcheck_f1_score3.append(f1_score)

# Ensure the plot directory exists
os.makedirs(eval_dir, exist_ok=True)

fig = plt.figure(figsize=(24, 6))

# Plot F1 score curve
ax3 = fig.add_subplot(1, 3, 3)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.plot(thresholds, our_f1_score, label="MetaQA-GPT4", color="cyan")
ax3.plot(thresholds, our_f1_score1, label="MetaQA-GPT3.5", color="purple")
ax3.plot(thresholds, our_f1_score2, label="MetaQA-Llama3", color="green")
ax3.plot(thresholds, our_f1_score3, label="MetaQA-Mistral", color="blue")
ax3.plot(thresholds, selfcheck_f1_score, label="SelfCheckGPT-GPT4o", color="pink", linestyle="--")
ax3.plot(thresholds, selfcheck_f1_score1, label="SelfCheckGPT-GPT3.5", color="red", linestyle="--")
ax3.plot(thresholds, selfcheck_f1_score2, label="SelfCheckGPT-Llama3", color="brown", linestyle="--")
ax3.plot(thresholds, selfcheck_f1_score3, label="SelfCheckGPT-Mistral", color="orange", linestyle="--")
ax3.set_xlabel("Threshold", fontsize=16)
ax3.set_ylabel("F1 Score", fontsize=16)
ax3.set_title("F1 Score Curve on Multiple Models", fontsize=16)
ax3.grid(False)

# Plot Precision for thresholds
ax1 = fig.add_subplot(1, 3, 1)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.plot(thresholds, our_precision, label="MetaQA-GPT4", color="cyan")
ax1.plot(thresholds, our_precision1, label="MetaQA-GPT3.5", color="purple")
ax1.plot(thresholds, our_precision2, label="MetaQA-Llama3", color="green")
ax1.plot(thresholds, our_precision3, label="MetaQA-Mistral", color="blue")
ax1.plot(thresholds, selfcheck_precision, label="SelfCheckGPT-GPT4", color="pink", linestyle="--")
ax1.plot(thresholds, selfcheck_precision1, label="SelfCheckGPT-GPT3.5", color="red", linestyle="--")
ax1.plot(thresholds, selfcheck_precision2, label="SelfCheckGPT-Llama3", color="brown", linestyle="--")
ax1.plot(thresholds, selfcheck_precision3, label="SelfCheckGPT-Mistral", color="orange", linestyle="--")
ax1.set_xlabel("Threshold", fontsize=16)
ax1.set_ylabel("Precision", fontsize=16)
ax1.set_title("Precision Curve on Multiple Models", fontsize=16)
ax1.grid(False)

# Plot Recall for thresholds
ax2 = fig.add_subplot(1, 3, 2)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.plot(thresholds, our_recall, label="MetaQA-GPT4", color="cyan")
ax2.plot(thresholds, our_recall1, label="MetaQA-GPT3.5", color="purple")
ax2.plot(thresholds, our_recall2, label="MetaQA-Llama3", color="green")
ax2.plot(thresholds, our_recall3, label="MetaQA-Mistral", color="blue")
ax2.plot(thresholds, selfcheck_recall, label="SelfCheckGPT-GPT4", color="pink", linestyle="--")
ax2.plot(thresholds, selfcheck_recall1, label="SelfCheckGPT-GPT3.5", color="red", linestyle="--")
ax2.plot(thresholds, selfcheck_recall2, label="SelfCheckGPT-Llama3", color="brown", linestyle="--")
ax2.plot(thresholds, selfcheck_recall3, label="SelfCheckGPT-Mistral", color="orange", linestyle="--")
ax2.set_xlabel("Threshold", fontsize=16)
ax2.set_ylabel("Recall", fontsize=16)
ax2.set_title("Recall Curve on Multiple Models", fontsize=16)
ax2.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=4, borderaxespad=0)
ax2.grid(False)

fig.savefig(os.path.join(eval_dir, "multimodel.pdf"), dpi=300, bbox_inches='tight', pad_inches=0)


