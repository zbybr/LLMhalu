import math
import os

import matplotlib.pyplot as plt
import pandas as pd

dataset_path4 = "output/final/llama3_outputs_truthfulqa1.3_100samples_run1.csv"
dataset_path5 = "output/final/llama3_outputs_truthfulqa1.3_100samples_run2.csv"
dataset_path6 = "output/final/llama3_outputs_truthfulqa1.3_100samples_run3.csv"
dataset_path1 = "output/final/gpt3_outputs_truthfulqa1.3_100samples_run1.csv"
dataset_path2 = "output/final/gpt3_outputs_truthfulqa1.3_100samples_run2.csv"
dataset_path3 = "output/final/gpt3_outputs_truthfulqa1.3_100samples_run3.csv"
eval_dir = "evaluation_metric/final/overall"
df1 = pd.read_csv(dataset_path1, encoding="latin-1")
df2 = pd.read_csv(dataset_path2, encoding="latin-1")
df3 = pd.read_csv(dataset_path3, encoding="latin-1")
df4 = pd.read_csv(dataset_path4, encoding="latin-1")
df5 = pd.read_csv(dataset_path5, encoding="latin-1")
df6 = pd.read_csv(dataset_path6, encoding="latin-1")


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

precision1 = []
recall1 = []
f1_score1 = []
precision2 = []
recall2 = []
f1_score2 = []
precision3 = []
recall3 = []
f1_score3 = []
mean_precision = []
mean_recall = []
mean_f1 = []
variance_r1 = []
variance_p1 = []
variance_f1 = []
variance_r2 = []
variance_p2 = []
variance_f2 = []

for threshold in thresholds:
    print(f"Threshold: {threshold}")
    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    scores1 = []
    for index, row in df1.iterrows():
        score = calculate_mt_hallucination_score(row)
        print("MT Score:", score)

        halu = row["hallucination_check"]  # Replace this key with ground truths

        row_score = get_metric_score(score, threshold, halu)

        scores1.append(row_score)

    precision_1, recall_1, f1_score_1 = calculate_metrics(scores1)
    print(f"Our Score\nPrecision: {precision_1}, Recall: {recall_1}, F1: {f1_score_1}")
    precision1.append(precision_1)
    recall1.append(recall_1)
    f1_score1.append(f1_score_1)
    sum_precision += precision_1
    sum_recall += recall_1
    sum_f1 += f1_score_1

    scores2 = []
    for index, row in df2.iterrows():
        score = calculate_mt_hallucination_score(row)
        print("MT Score:", score)

        halu = row["hallucination_check"]  # Replace this key with ground truths

        row_score = get_metric_score(score, threshold, halu)

        scores2.append(row_score)

    precision_2, recall_2, f1_score_2 = calculate_metrics(scores2)
    print(f"Our Score\nPrecision: {precision_2}, Recall: {recall_2}, F1: {f1_score_2}")
    precision2.append(precision_2)
    recall2.append(recall_2)
    f1_score2.append(f1_score_2)
    sum_precision += precision_2
    sum_recall += recall_2
    sum_f1 += f1_score_2

    scores3 = []
    for index, row in df3.iterrows():
        score = calculate_mt_hallucination_score(row)
        print("MT Score:", score)

        halu = row["hallucination_check"]  # Replace this key with ground truths

        row_score = get_metric_score(score, threshold, halu)

        scores3.append(row_score)

    precision_3, recall_3, f1_score_3 = calculate_metrics(scores3)
    print(f"Our Score\nPrecision: {precision_3}, Recall: {recall_3}, F1: {f1_score_3}")
    precision3.append(precision_3)
    recall3.append(recall_3)
    f1_score3.append(f1_score_3)
    sum_precision += precision_3
    sum_recall += recall_3
    sum_f1 += f1_score_3

    mean_precision_t = sum_precision / 3.0
    mean_precision.append(mean_precision_t)
    mean_recall_t = sum_recall / 3.0
    mean_recall.append(mean_recall_t)
    mean_f1_t = sum_f1 / 3.0
    mean_f1.append(mean_f1_t)

    variance = ((mean_precision_t - precision_1)**2 + (mean_precision_t - precision_2)**2 + (mean_precision_t - precision_3)**2 ) / 3.0
    variance_p1.append(variance)
    variance = ((mean_recall_t - recall_1) ** 2 + (mean_recall_t - recall_2) ** 2 + (
                mean_recall_t - recall_3) ** 2) / 3.0
    variance_r1.append(variance)
    variance = ((mean_f1_t - f1_score_1) ** 2 + (mean_f1_t - f1_score_2) ** 2 + (
                mean_f1_t - f1_score_3) ** 2) / 3.0
    variance_f1.append(variance)

    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    scores1 = []
    for index, row in df4.iterrows():
        score = calculate_mt_hallucination_score(row)
        print("MT Score:", score)

        halu = row["hallucination_check"]  # Replace this key with ground truths

        row_score = get_metric_score(score, threshold, halu)

        scores1.append(row_score)

    precision_1, recall_1, f1_score_1 = calculate_metrics(scores1)
    print(f"Our Score\nPrecision: {precision_1}, Recall: {recall_1}, F1: {f1_score_1}")
    precision1.append(precision_1)
    recall1.append(recall_1)
    f1_score1.append(f1_score_1)
    sum_precision += precision_1
    sum_recall += recall_1
    sum_f1 += f1_score_1

    scores2 = []
    for index, row in df5.iterrows():
        score = calculate_mt_hallucination_score(row)
        print("MT Score:", score)

        halu = row["hallucination_check"]  # Replace this key with ground truths

        row_score = get_metric_score(score, threshold, halu)

        scores2.append(row_score)

    precision_2, recall_2, f1_score_2 = calculate_metrics(scores2)
    print(f"Our Score\nPrecision: {precision_2}, Recall: {recall_2}, F1: {f1_score_2}")
    precision2.append(precision_2)
    recall2.append(recall_2)
    f1_score2.append(f1_score_2)
    sum_precision += precision_2
    sum_recall += recall_2
    sum_f1 += f1_score_2

    scores3 = []
    for index, row in df6.iterrows():
        score = calculate_mt_hallucination_score(row)
        print("MT Score:", score)

        halu = row["hallucination_check"]  # Replace this key with ground truths

        row_score = get_metric_score(score, threshold, halu)

        scores3.append(row_score)

    precision_3, recall_3, f1_score_3 = calculate_metrics(scores3)
    print(f"Our Score\nPrecision: {precision_3}, Recall: {recall_3}, F1: {f1_score_3}")
    precision3.append(precision_3)
    recall3.append(recall_3)
    f1_score3.append(f1_score_3)
    sum_precision += precision_3
    sum_recall += recall_3
    sum_f1 += f1_score_3

    mean_precision_t = sum_precision / 3.0
    mean_precision.append(mean_precision_t)
    mean_recall_t = sum_recall / 3.0
    mean_recall.append(mean_recall_t)
    mean_f1_t = sum_f1 / 3.0
    mean_f1.append(mean_f1_t)

    variance = ((mean_precision_t - precision_1) ** 2 + (mean_precision_t - precision_2) ** 2 + (
                mean_precision_t - precision_3) ** 2) / 3.0
    variance_p2.append(variance)
    variance = ((mean_recall_t - recall_1) ** 2 + (mean_recall_t - recall_2) ** 2 + (
            mean_recall_t - recall_3) ** 2) / 3.0
    variance_r2.append(variance)
    variance = ((mean_f1_t - f1_score_1) ** 2 + (mean_f1_t - f1_score_2) ** 2 + (
            mean_f1_t - f1_score_3) ** 2) / 3.0
    variance_f2.append(variance)


# Ensure the plot directory exists
os.makedirs(eval_dir, exist_ok=True)


# # Plot F1 score curve
# plt.plot(thresholds, f1_score1, label="E1")
# plt.plot(thresholds, f1_score2, label="E2")
# plt.plot(thresholds, f1_score3, label="E3")
# plt.xlabel("Threshold")
# plt.ylabel("F1 Score")
# plt.title("F1 Score Curve")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(eval_dir, "f1_score_curve.png"))
# plt.close()
#
# # Plot Precision for thresholds
# plt.plot(thresholds, precision1, label="E1")
# plt.plot(thresholds, precision2, label="E2")
# plt.plot(thresholds, precision3, label="E3")
# plt.xlabel("Threshold")
# plt.ylabel("Precision")
# plt.title("Precision Curve")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(eval_dir, "precision_curve.png"))
# plt.close()
#
# # Plot Recall for thresholds
# plt.plot(thresholds, recall1, label="E1")
# plt.plot(thresholds, recall2, label="E2")
# plt.plot(thresholds, recall3, label="E3")
# plt.xlabel("Threshold")
# plt.ylabel("Recall")
# plt.title("Recall Curve")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(eval_dir, "recall_curve.png"))
# plt.close()
#
# # Plot Recall for thresholds
# plt.plot(thresholds, mean_precision, label="Mean Precision")
# plt.plot(thresholds, mean_recall, label="Mean Recall")
# plt.plot(thresholds, mean_f1, label="Mean F1")
# plt.xlabel("Threshold")
# plt.ylabel("Value")
# plt.title("Mean Curve")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(eval_dir, "mean.png"))
# plt.close()

fig = plt.figure(figsize=(16, 6))

ax1 = fig.add_subplot(1, 2, 1)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.plot(thresholds, variance_p1, label="Precision Variance")
ax1.plot(thresholds, variance_r1, label="Recall Variance")
ax1.plot(thresholds, variance_f1, label="F1 Score Variance")
ax1.set_xlabel("Threshold", fontsize=16)
ax1.set_ylabel("Value", fontsize=16)
ax1.set_title("Deviation Curve on GPT-3.5", fontsize=16)
ax1.legend(bbox_to_anchor=(1, -0.15), ncol=3, loc='upper center', borderaxespad=0)

ax2 = fig.add_subplot(1, 2, 2)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.plot(thresholds, variance_p2, label="Precision Variance")
ax2.plot(thresholds, variance_r2, label="Recall Variance")
ax2.plot(thresholds, variance_f2, label="F1 Score Variance")
ax2.set_xlabel("Threshold", fontsize=16)
ax2.set_ylabel("Value", fontsize=16)
ax2.set_title("Deviation Curve on Llama3", fontsize=16)
plt.savefig(os.path.join(eval_dir, "deviation.pdf"), dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()
# # Save outputs
# output_df = pd.DataFrame(
#     {
#         "Threshold": thresholds,
#         "Precision1": precision1,
#         "Recall1": recall1,
#         "F11": f1_score1,
#         "Precision2": precision2,
#         "Recall2": recall2,
#         "F12": f1_score2,
#         "Precision3": precision3,
#         "Recall3": recall3,
#         "F13": f1_score3,
#     }
# )
#
# output_df.to_csv(os.path.join(eval_dir, "comparison.csv"), index=False)

