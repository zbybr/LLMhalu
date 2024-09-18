import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

dataset_path = "output/final/overall_outputs1.csv"
eval_dir = "evaluation_metric/final/overall/"

scores = []
selfcheck_scores = []
if __name__ == "__main__":
    df = pd.read_csv(dataset_path, encoding="latin-1")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
        if row['hallucination_check'] == 'no':
            continue
        scores.append(row['score'])
        selfcheck_scores.append(row['selfcheck_score'])

    bar_width = 0.4
    bins = np.arange(0, 1.2, 0.1)

    counts1, _ = np.histogram(scores, bins=bins)
    counts2, _ = np.histogram(selfcheck_scores, bins=bins)

    r1 = np.arange(len(counts1))
    r2 = [x + bar_width for x in r1]

    plt.bar(r1, counts1, width=bar_width, color='orange', edgecolor='none', label='MetaQA', alpha=0.7)
    plt.bar(r2, counts2, width=bar_width, color='cyan', edgecolor='none', label='SelfCheckGPT', alpha=0.7)

    plt.xticks([r + bar_width / 2 for r in range(len(counts1))], np.round(bins[:-1], 2))
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.xlabel('Hallucination Score', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.legend()
    plt.savefig(os.path.join(eval_dir, "haluscore.pdf"), dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()

