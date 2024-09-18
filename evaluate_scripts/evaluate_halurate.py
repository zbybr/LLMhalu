import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


eval_dir = "evaluation_metric/final/overall/"

model_list = ['GPT-4o', 'GPT-3.5', 'Llama3-8B', 'Mistral-7B']
dataset_list = ['TruthfulQA-Enhanced', 'HotpotQA', 'FreshQA']

gpt4_overall = []
dataset_path = "output/final/gpt4_outputs_truthfulqa1.3.csv"
df = pd.read_csv(dataset_path, encoding="latin-1")
total = 0
halu = 0
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
    check = row["hallucination_check"]
    if check == 'yes':
        halu += 1
    total += 1
gpt4_overall.append(1.0 * halu / total)

dataset_path = "output/final/gpt4_outputs_hotpotqa.csv"
df = pd.read_csv(dataset_path, encoding="latin-1")
total = 0
halu = 0
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
    check = row["hallucination_check"]
    if check == 'yes':
        halu += 1
    total += 1
gpt4_overall.append(1.0 * halu / total)

dataset_path = "output/final/gpt4_outputs_freshqa.csv"
df = pd.read_csv(dataset_path, encoding="latin-1")
total = 0
halu = 0
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
    check = row["hallucination_check"]
    if check == 'yes':
        halu += 1
    total += 1
gpt4_overall.append(1.0 * halu / total)

gpt3_overall = []
dataset_path = "output/final/gpt3_outputs_truthfulqa1.3.csv"
df = pd.read_csv(dataset_path, encoding="latin-1")
total = 0
halu = 0
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
    check = row["hallucination_check"]
    if check == 'yes':
        halu += 1
    total += 1
gpt3_overall.append(1.0 * halu / total)

dataset_path = "output/final/gpt3_outputs_hotpotqa.csv"
df = pd.read_csv(dataset_path, encoding="latin-1")
total = 0
halu = 0
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
    check = row["hallucination_check"]
    if check == 'yes':
        halu += 1
    total += 1
gpt3_overall.append(1.0 * halu / total)

dataset_path = "output/final/gpt3_outputs_freshqa.csv"
df = pd.read_csv(dataset_path, encoding="latin-1")
total = 0
halu = 0
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
    check = row["hallucination_check"]
    if check == 'yes':
        halu += 1
    total += 1
gpt3_overall.append(1.0 * halu / total)

llama3_overall = []
dataset_path = "output/final/llama3_outputs_truthfulqa1.3.csv"
df = pd.read_csv(dataset_path, encoding="latin-1")
total = 0
halu = 0
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
    check = row["hallucination_check"]
    if check == 'yes':
        halu += 1
    total += 1
llama3_overall.append(1.0 * halu / total)

dataset_path = "output/final/llama3_outputs_hotpotqa.csv"
df = pd.read_csv(dataset_path, encoding="latin-1")
total = 0
halu = 0
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
    check = row["hallucination_check"]
    if check == 'yes':
        halu += 1
    total += 1
llama3_overall.append(1.0 * halu / total)

dataset_path = "output/final/llama3_outputs_freshqa.csv"
df = pd.read_csv(dataset_path, encoding="latin-1")
total = 0
halu = 0
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
    check = row["hallucination_check"]
    if check == 'yes':
        halu += 1
    total += 1
llama3_overall.append(1.0 * halu / total)

mistral_overall = []
dataset_path = "output/final/mistral_outputs_truthfulqa1.3.csv"
df = pd.read_csv(dataset_path, encoding="latin-1")
total = 0
halu = 0
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
    check = row["hallucination_check"]
    if check == 'yes':
        halu += 1
    total += 1
mistral_overall.append(1.0 * halu / total)

dataset_path = "output/final/mistral_outputs_hotpotqa.csv"
df = pd.read_csv(dataset_path, encoding="latin-1")
total = 0
halu = 0
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
    check = row["hallucination_check"]
    if check == 'yes':
        halu += 1
    total += 1
mistral_overall.append(1.0 * halu / total)

dataset_path = "output/final/mistral_outputs_freshqa.csv"
df = pd.read_csv(dataset_path, encoding="latin-1")
total = 0
halu = 0
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
    check = row["hallucination_check"]
    if check == 'yes':
        halu += 1
    total += 1
mistral_overall.append(1.0 * halu / total)

data = [gpt4_overall, gpt3_overall, llama3_overall, mistral_overall]
data_transposed = np.transpose(data)
fig, ax = plt.subplots(figsize=(6, 8))
sns.heatmap(pd.DataFrame(np.round(data_transposed, 2), columns=model_list, index=dataset_list), annot=True, vmax=0.8, vmin=0.2,
            square=True, cmap='YlGnBu', cbar_kws={'shrink': 0.5})
ax.set_title('Hallucination Rate Heatmap Overview', fontsize=14)

plt.savefig(eval_dir + 'overallhalu.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
