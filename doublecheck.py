import pandas as pd
import matplotlib.pyplot as plt

dataset_path = "comparison.csv"
df = pd.read_csv(dataset_path, encoding='latin-1')


def get_score(score, threshold, ground_truth):
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
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

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
    for _, row in df.iterrows():
        score = float(row["scores"])
        halu = row["Hallucination check(Manually)"]
        score_s = float(row["Selfcheck Scores"][1:-1])

        row_score = get_score(score, threshold, halu)
        selfcheck_score = get_score(score_s, threshold, halu)

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

# Plot precision-recall curve
plt.plot(our_recall, our_precision, label='Our Model')
plt.plot(selfcheck_recall, selfcheck_precision, label='Self Check Model')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()

# Plot F1 score curve
plt.plot(thresholds, our_f1_score, label='Our Model')
plt.plot(thresholds, selfcheck_f1_score, label='Self Check Model')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score Curve')
plt.legend()
plt.grid(True)
plt.show()

# Save outputs
output_df = pd.DataFrame({
    "Threshold": thresholds,
    "Our Precision": our_precision,
    "Our Recall": our_recall,
    "Our F1": our_f1_score,
    "Selfcheck Precision": selfcheck_precision,
    "Selfcheck Recall": selfcheck_recall,
    "Selfcheck F1": selfcheck_f1_score
})

output_df.to_csv("metrics_comparison.csv", index=False)