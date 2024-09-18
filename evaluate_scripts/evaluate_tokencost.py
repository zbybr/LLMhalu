import os
import ast
import pandas as pd

data_dir = "output/final"
dataset_path = "output/final/gpt3_outputs_truthfulqa1.3_token_usage_seed77_100samples.csv"
dataset_path1 = "output/final/gpt4_outputs_truthfulqa1.3_token_usage_seed77.csv"
dataset_path2 = "output/final/llama3_outputs_truthfulqa_token_usage_seed77.csv"
dataset_path3 = "output/final/mistral_outputs_truthfulqa_token_usage_seed77.csv"
eval_dir = "evaluation_metric/final/overall"
df = pd.read_csv(dataset_path, encoding="latin-1")
df1 = pd.read_csv(dataset_path1, encoding="latin-1")
df2 = pd.read_csv(dataset_path2, encoding="latin-1")
df3 = pd.read_csv(dataset_path3, encoding="latin-1")

gpt3_base_tokencost = []
gpt3_self_tokencost = []
gpt3_meta_tokencost = []
for index, row in df.iterrows():
    base_tokencost = ast.literal_eval(row['base_token_usage'])['total_tokens']
    meta_tokencost = ast.literal_eval(row['metaqa_token_usage'])['total_tokens']
    self_tokencost = ast.literal_eval(row['selfcheck_token_usage'])['total_tokens']

    df.at[index, 'base_total_token'] = base_tokencost
    df.at[index, 'self_total_token'] = self_tokencost
    df.at[index, 'meta_total_token'] = meta_tokencost

    gpt3_base_tokencost.append(base_tokencost)
    gpt3_self_tokencost.append(self_tokencost)
    gpt3_meta_tokencost.append(meta_tokencost)

gpt4_base_tokencost = []
gpt4_self_tokencost = []
gpt4_meta_tokencost = []
for index, row in df1.iterrows():
    base_tokencost = ast.literal_eval(row['base_token_usage'])['total_tokens']
    meta_tokencost = ast.literal_eval(row['metaqa_token_usage'])['total_tokens']
    self_tokencost = ast.literal_eval(row['selfcheck_token_usage'])['total_tokens']

    df.at[index, 'base_total_token'] = base_tokencost
    df.at[index, 'self_total_token'] = self_tokencost
    df.at[index, 'meta_total_token'] = meta_tokencost

    gpt4_base_tokencost.append(base_tokencost)
    gpt4_self_tokencost.append(self_tokencost)
    gpt4_meta_tokencost.append(meta_tokencost)

llama3_base_tokencost = []
llama3_self_tokencost = []
llama3_meta_tokencost = []
for index, row in df2.iterrows():
    base_tokencost = ast.literal_eval(row['base_token_usage'])['total_tokens']
    meta_tokencost = ast.literal_eval(row['metaqa_token_usage'])['total_tokens']
    self_tokencost = ast.literal_eval(row['selfcheck_token_usage'])['total_tokens']

    df.at[index, 'base_total_token'] = base_tokencost
    df.at[index, 'self_total_token'] = self_tokencost
    df.at[index, 'meta_total_token'] = meta_tokencost

    llama3_base_tokencost.append(base_tokencost)
    llama3_self_tokencost.append(self_tokencost)
    llama3_meta_tokencost.append(meta_tokencost)

mistral_base_tokencost = []
mistral_self_tokencost = []
mistral_meta_tokencost = []
for index, row in df3.iterrows():
    base_tokencost = ast.literal_eval(row['base_token_usage'])['total_tokens']
    meta_tokencost = ast.literal_eval(row['metaqa_token_usage'])['total_tokens']
    self_tokencost = ast.literal_eval(row['selfcheck_token_usage'])['total_tokens']

    df.at[index, 'base_total_token'] = base_tokencost
    df.at[index, 'self_total_token'] = self_tokencost
    df.at[index, 'meta_total_token'] = meta_tokencost

    mistral_base_tokencost.append(base_tokencost)
    mistral_self_tokencost.append(self_tokencost)
    mistral_meta_tokencost.append(meta_tokencost)

# Save outputs
output_df = pd.DataFrame(
    {
        "gpt3_base_tokenusage": gpt3_base_tokencost,
        "gpt3_self_tokenusage": gpt3_self_tokencost,
        "gpt3_meta_tokenusage": gpt3_meta_tokencost,
        "gpt4_base_tokenusage": gpt4_base_tokencost,
        "gpt4_self_tokenusage": gpt4_self_tokencost,
        "gpt4_meta_tokenusage": gpt4_meta_tokencost,
        "llama3_base_tokenusage": llama3_base_tokencost,
        "llama3_self_tokenusage": llama3_self_tokencost,
        "llama3_meta_tokenusage": llama3_meta_tokencost,
        "mistral_base_tokenusage": mistral_base_tokencost,
        "mistral_self_tokenusage": mistral_self_tokencost,
        "mistral_meta_tokenusage": mistral_meta_tokencost,
    }
)

output_df.to_csv(os.path.join(eval_dir, "overall_tokencost.csv"), index=False)
