from openai import OpenAI
import pandas as pd

dataset_path = "comparison.csv"
df = pd.read_csv(dataset_path, encoding='latin-1')

total = 0 # total hallucination samples
true_s = 0 # true positive from selfcheckgpt
true = 0 # true positive
check = 0 # TP+FP
check_s = 0 # TP+FP

threshold = 0.5

for _, row in df.iterrows():
    score = float(row["scores"])
    halu = row["Hallucination check(Manually)"]
    score_s = float(row["Selfcheck Scores"][1:-1])
    if 'Yes' in halu:
        total += 1
        if score_s >= threshold:
            true_s += 1
        if score >= threshold:
            true += 1
    if score >= threshold:
        check += 1
    if score_s >= threshold:
        check_s += 1


print("hallucination rate: ")
print(str(total) + " / " + "100")
print(str(100.0 * total / 100.0) + "\n")

print("selfcheckgpt precision rate: ")
print(str(true_s) + " / " + str(check_s))
print(str(100.0 * true_s / check_s) + "\n")

print("selfcheckgpt recall rate: ")
print(str(true_s) + " / " + str(total))
print(str(100.0 * true_s / total) + "\n")

print("precision rate: ")
print(str(true) + " / " + str(check))
print(str(100.0 * true / check) + "\n")

print("recall rate: ")
print(str(true) + " / " + str(total))
print(str(100.0 * true / total) + "\n")

