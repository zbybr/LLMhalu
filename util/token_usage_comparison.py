import json
from collections import defaultdict

import pandas as pd


def get_total(data, key):
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    for d in data:
        prompt_tokens += d[key]["prompt_tokens"]
        completion_tokens += d[key]["completion_tokens"]
        total_tokens += d[key]["total_tokens"]

    return prompt_tokens, completion_tokens, total_tokens


def calculate_token_comparison(data):
    base_tokens = get_total(data, "base_token_usage")
    selfcheck_tokens = get_total(data, "selfcheck_token_usage")
    previous_synant_tokens = get_total(data, "previous_synant_token_usage")
    current_synant_tokens = get_total(data, "current_synant_token_usage")

    return [base_tokens, selfcheck_tokens, previous_synant_tokens, current_synant_tokens]


data_path = "D:\\Projects\\LLMhalu\\gpt3\\data\\gpt3_outputs_truthfulqa1.2_token_usage_seed77.csv"
df = pd.read_csv(data_path)

data = []

for _, row in df.iterrows():
    row_data = {
        'base_token_usage': json.loads(row["base_token_usage"].replace("'", "\"")),
        'selfcheck_token_usage': json.loads(row["selfcheck_token_usage"].replace("'", "\"")),
        'previous_synant_token_usage': json.loads(row["previous_synant_token_usage"].replace("'", "\"")),
        'current_synant_token_usage': json.loads(row["current_synant_token_usage"].replace("'", "\""))
    }
    data.append(row_data)

comparison_data = calculate_token_comparison(data[0:5])

# Use the first tuple as the baseline
baseline = comparison_data[0]


# Function to calculate the percentage increase
def calculate_percentage_increase(baseline, current):
    return [(current[i] - baseline[i]) / baseline[i] * 100 for i in range(len(baseline))]


# Calculate the percentage increases
percentage_increases = [calculate_percentage_increase(baseline, current) for current in comparison_data[1:]]


print(comparison_data)

# Print the results
for i, increases in enumerate(percentage_increases, start=1):
    print(f"Tuple {i + 1}:")
    print(f"  Prompt tokens increase: {increases[0]:.2f}%")
    print(f"  Completion tokens increase: {increases[1]:.2f}%")
    print(f"  Total tokens increase: {increases[2]:.2f}%")
    print()