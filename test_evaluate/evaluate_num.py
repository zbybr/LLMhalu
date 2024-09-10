import os

import matplotlib.pyplot as plt
import pandas as pd

dataset_path = "overall_outputs.csv"
# eval_dir = "evaluation_metric/final/overall"
df = pd.read_csv(dataset_path, encoding="latin-1")


def get_metric_score(score, threshold, ground_truth):
    ground_truth = ground_truth.strip().lower()
    label = "yes" if score >= threshold else "no"

    tp = 1 if ground_truth == "yes" and label == "yes" else 0
    fp = 1 if ground_truth == "no" and label == "yes" else 0
    tn = 1 if ground_truth == "no" and label == "no" else 0
    fn = 1 if ground_truth == "yes" and label == "no" else 0

    return tp, fp, tn, fn


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


thresholds = [0.6, 0.65, 0.7, 0.75, 0.8]
mutations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

our_precision = []
our_recall = []
our_f1_score = []
our_precision1 = []
our_recall1 = []
our_f1_score1 = []
our_precision2 = []
our_recall2 = []
our_f1_score2 = []
our_precision3 = []
our_recall3 = []
our_f1_score3 = []
our_precision4 = []
our_recall4 = []
our_f1_score4 = []
our_precision5 = []
our_recall5 = []
our_f1_score5 = []
our_precision6 = []
our_recall6 = []
our_f1_score6 = []
our_precision7 = []
our_recall7 = []
our_f1_score7 = []
our_precision8 = []
our_recall8 = []
our_f1_score8 = []
our_precision9 = []
our_recall9 = []
our_f1_score9 = []
our_precision_syn = []
our_recall_syn = []
our_f1_score_syn = []
our_precision_ant = []
our_recall_ant = []
our_f1_score_ant = []
for threshold in thresholds:
    print(f"Threshold: {threshold}")
    our_scores = []
    our_scores2 = []
    our_scores4 = []
    our_scores6 = []
    our_scores8 = []
    our_scores1 = []
    our_scores3 = []
    our_scores5 = []
    our_scores7 = []
    our_scores9 = []
    synscores = []
    antscores = []

    for index, row in df.iterrows():
        score = row["score"]
        score2 = row["score2"]
        score4 = row["score4"]
        score6 = row["score6"]
        score8 = row["score8"]
        score1 = row["score1"]
        score3 = row["score3"]
        score5 = row["score5"]
        score7 = row["score7"]
        score9 = row["score9"]
        scoresyn = row["scoresyn"]
        scoreant = row["scoreant"]

        halu = row["hallucination_check"]  # Replace this key with ground truths

        row_score = get_metric_score(score, threshold, halu)
        row_score1 = get_metric_score(score1, threshold, halu)
        row_score2 = get_metric_score(score2, threshold, halu)
        row_score3 = get_metric_score(score3, threshold, halu)
        row_score4 = get_metric_score(score4, threshold, halu)
        row_score5 = get_metric_score(score5, threshold, halu)
        row_score6 = get_metric_score(score6, threshold, halu)
        row_score7 = get_metric_score(score7, threshold, halu)
        row_score8 = get_metric_score(score8, threshold, halu)
        row_score9 = get_metric_score(score9, threshold, halu)

        row_syn = get_metric_score(scoresyn, threshold, halu)
        row_ant = get_metric_score(scoreant, threshold, halu)

        our_scores.append(row_score)
        our_scores1.append(row_score1)
        our_scores2.append(row_score2)
        our_scores3.append(row_score3)
        our_scores4.append(row_score4)
        our_scores5.append(row_score5)
        our_scores6.append(row_score6)
        our_scores7.append(row_score7)
        our_scores8.append(row_score8)
        our_scores9.append(row_score9)
        synscores.append(row_syn)
        antscores.append(row_ant)

    precision, recall, f1_score = calculate_metrics(our_scores1)
    print(f"MetaQA&2 Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    our_precision1.append(precision)
    our_recall1.append(recall)
    our_f1_score1.append(f1_score)

    precision, recall, f1_score = calculate_metrics(our_scores2)
    print(f"MetaQA&2 Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    our_precision2.append(precision)
    our_recall2.append(recall)
    our_f1_score2.append(f1_score)

    precision, recall, f1_score = calculate_metrics(our_scores3)
    print(f"MetaQA&2 Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    our_precision3.append(precision)
    our_recall3.append(recall)
    our_f1_score3.append(f1_score)

    precision, recall, f1_score = calculate_metrics(our_scores4)
    print(f"MetaQA&4 Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    print("================================================================")
    our_precision4.append(precision)
    our_recall4.append(recall)
    our_f1_score4.append(f1_score)

    precision, recall, f1_score = calculate_metrics(our_scores5)
    print(f"MetaQA&2 Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    our_precision5.append(precision)
    our_recall5.append(recall)
    our_f1_score5.append(f1_score)

    precision, recall, f1_score = calculate_metrics(our_scores6)
    print(f"MetaQA&6 Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    print("================================================================")
    our_precision6.append(precision)
    our_recall6.append(recall)
    our_f1_score6.append(f1_score)

    precision, recall, f1_score = calculate_metrics(our_scores7)
    print(f"MetaQA&6 Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    print("================================================================")
    our_precision7.append(precision)
    our_recall7.append(recall)
    our_f1_score7.append(f1_score)

    precision, recall, f1_score = calculate_metrics(our_scores8)
    print(f"MetaQA&8 Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    print("================================================================")
    our_precision8.append(precision)
    our_recall8.append(recall)
    our_f1_score8.append(f1_score)

    precision, recall, f1_score = calculate_metrics(our_scores9)
    print(f"MetaQA&8 Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    print("================================================================")
    our_precision9.append(precision)
    our_recall9.append(recall)
    our_f1_score9.append(f1_score)

    precision, recall, f1_score = calculate_metrics(our_scores)
    print(f"MetaQA&10 Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    print("================================================================")
    our_precision.append(precision)
    our_recall.append(recall)
    our_f1_score.append(f1_score)

    # precision, recall, f1_score = calculate_metrics(synscores)
    # print(f"MetaQA&SYN Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    # print("================================================================")
    # our_precision_syn.append(precision)
    # our_recall_syn.append(recall)
    # our_f1_score_syn.append(f1_score)
    #
    # precision, recall, f1_score = calculate_metrics(antscores)
    # print(f"MetaQA&ANT Score\nPrecision: {precision}, Recall: {recall}, F1: {f1_score}")
    # print("================================================================")
    # our_precision_ant.append(precision)
    # our_recall_ant.append(recall)
    # our_f1_score_ant.append(f1_score)

# # Ensure the plot directory exists
# os.makedirs(eval_dir, exist_ok=True)

# # Plot F1 score curve
# plt.plot(mutations, precision0, label="MetaQA&0")
# plt.plot(mutations, our_f1_score2, label="MetaQA&0.2")
# plt.plot(mutations, our_f1_score4, label="MetaQA&0.4")
# plt.plot(mutations, our_f1_score6, label="MetaQA&0.6")
# plt.plot(mutations, our_f1_score8, label="MetaQA&0.8")
# plt.xlabel("Number of mutations")
# plt.ylabel("F1 Score")
# plt.title("F1 Score Curve")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(eval_dir, "f1_score_curve.png"))
# plt.close()

# # Plot Precision for thresholds
# plt.plot(mutations, our_precision2, label="MetaQA&2")
# plt.plot(mutations, our_precision4, label="MetaQA&4")
# plt.plot(mutations, our_precision6, label="MetaQA&6")
# plt.plot(mutations, our_precision8, label="MetaQA&8")
# plt.plot(mutations, our_precision, label="MetaQA")
# plt.xlabel("Number of mutations")
# plt.ylabel("Precision")
# plt.title("Precision Curve")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(eval_dir, "precision_curve.png"))
# plt.close()

# # Plot Recall for thresholds
# plt.plot(mutations, our_recall2, label="MetaQA&2")
# plt.plot(mutations, our_recall4, label="MetaQA&4")
# plt.plot(mutations, our_recall6, label="MetaQA&6")
# plt.plot(mutations, our_recall8, label="MetaQA&8")
# plt.plot(mutations, our_recall, label="MetaQA")
# plt.xlabel("Number of mutations")
# plt.ylabel("Recall")
# plt.title("Recall Curve")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(eval_dir, "recall_curve.png"))
# plt.close()

# plt.plot(thresholds, our_f1_score_syn, label="MetaQA&SYN")
# plt.plot(thresholds, our_f1_score_ant, label="MetaQA&ANT")
# plt.plot(thresholds, our_f1_score, label="MetaQA")
# plt.xlabel("Threshold")
# plt.ylabel("F1 Score")
# plt.title("F1 Score Curve")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(eval_dir, "f1_score_curve_1.png"))
# plt.close()
#
# # Plot Precision for thresholds
# plt.plot(thresholds, our_precision_syn, label="MetaQA&SYN")
# plt.plot(thresholds, our_precision_ant, label="MetaQA&ANT")
# plt.plot(thresholds, our_precision, label="MetaQA")
# plt.xlabel("Threshold")
# plt.ylabel("Precision")
# plt.title("Precision Curve")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(eval_dir, "precision_curve_1.png"))
# plt.close()
#
# # Plot Recall for thresholds
# plt.plot(thresholds, our_recall_syn, label="MetaQA&SYN")
# plt.plot(thresholds, our_recall_ant, label="MetaQA&ANT")
# plt.plot(thresholds, our_recall, label="MetaQA")
# plt.xlabel("Threshold")
# plt.ylabel("Recall")
# plt.title("Recall Curve")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(eval_dir, "recall_curve_1.png"))
# plt.close()

precision0 = [our_precision1[0], our_precision2[0], our_precision3[0], our_precision4[0], our_precision5[0],
              our_precision6[0], our_precision7[0], our_precision8[0], our_precision9[0], our_precision[0]]
precision2 = [our_precision1[1], our_precision2[1], our_precision3[1], our_precision4[1], our_precision5[1],
              our_precision6[1], our_precision7[1], our_precision8[1], our_precision9[1], our_precision[1]]
precision4 = [our_precision1[2], our_precision2[2], our_precision3[2], our_precision4[2], our_precision5[2],
              our_precision6[2], our_precision7[2], our_precision8[2], our_precision9[2], our_precision[2]]
precision6 = [our_precision1[3], our_precision2[3], our_precision3[3], our_precision4[3], our_precision5[3],
              our_precision6[3], our_precision7[3], our_precision8[3], our_precision9[3], our_precision[3]]
precision8 = [our_precision1[4], our_precision2[4], our_precision3[4], our_precision4[4], our_precision5[4],
              our_precision6[4], our_precision7[4], our_precision8[4], our_precision9[4], our_precision[4]]

f1_score0 = [our_f1_score1[0], our_f1_score2[0], our_f1_score3[0], our_f1_score4[0], our_f1_score5[0], our_f1_score6[0],
             our_f1_score7[0], our_f1_score8[0], our_f1_score9[0], our_f1_score[0]]
f1_score2 = [our_f1_score1[1], our_f1_score2[1], our_f1_score3[1], our_f1_score4[1], our_f1_score5[1], our_f1_score6[1],
             our_f1_score7[1], our_f1_score8[1], our_f1_score9[1], our_f1_score[1]]
f1_score4 = [our_f1_score1[2], our_f1_score2[2], our_f1_score3[2], our_f1_score4[2], our_f1_score5[2], our_f1_score6[2],
             our_f1_score7[2], our_f1_score8[2], our_f1_score9[2], our_f1_score[2]]
f1_score6 = [our_f1_score1[3], our_f1_score2[3], our_f1_score3[3], our_f1_score4[3], our_f1_score5[3], our_f1_score6[3],
             our_f1_score7[3], our_f1_score8[3], our_f1_score9[3], our_f1_score[3]]
f1_score8 = [our_f1_score1[4], our_f1_score2[4], our_f1_score3[4], our_f1_score4[4], our_f1_score5[4], our_f1_score6[4],
             our_f1_score7[4], our_f1_score8[4], our_f1_score9[4], our_f1_score[4]]


fig = plt.figure(figsize=(16, 6))

# Plot Precision for mutations
ax1 = fig.add_subplot(1, 2, 1)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.plot(mutations, precision0, label=r'$\theta = 0.6$', marker='*', linewidth='1')
ax1.plot(mutations, precision2, label=r'$\theta = 0.65$', marker='^', linewidth='1')
ax1.plot(mutations, precision4, label=r'$\theta = 0.7$', marker='v', linewidth='1')
ax1.plot(mutations, precision6, label=r'$\theta = 0.75$', marker='o', linewidth='1')
ax1.plot(mutations, precision8, label=r'$\theta = 0.8$', marker='.', linewidth='1')
ax1.set_xlabel("Number of mutations", fontsize=16)
ax1.set_ylabel("Precision", fontsize=16)
ax1.set_title("Precision Curve", fontsize=16)
ax1.legend(bbox_to_anchor=(1, -0.15), ncol=5, loc='upper center', borderaxespad=0)

# Plot f1_score for mutations
ax2 = fig.add_subplot(1, 2, 2)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.plot(mutations, f1_score0, label="MetaQA&0.6", marker='*', linewidth='1')
ax2.plot(mutations, f1_score2, label="MetaQA&0.65", marker='^', linewidth='1')
ax2.plot(mutations, f1_score4, label="MetaQA&0.7", marker='v', linewidth='1')
ax2.plot(mutations, f1_score6, label="MetaQA&0.75", marker='o', linewidth='1')
ax2.plot(mutations, f1_score8, label="MetaQA&0.8", marker='.', linewidth='1')
ax2.set_xlabel("Number of mutations", fontsize=16)
ax2.set_ylabel("F1 score", fontsize=16)
ax2.set_title("F1 score Curve", fontsize=16)
plt.savefig("nummuts.pdf", dpi=300, bbox_inches='tight', pad_inches=0)

plt.close()
