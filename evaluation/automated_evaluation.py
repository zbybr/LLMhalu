import argparse
import sys
import warnings

sys.path.append("/home/mdafifal.mamun/research/LLMhalu/llama3")

import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

from llama3 import Llama3

warnings.filterwarnings("ignore")


system_prompt = (
    "Given two sentences, your job is to tell whether or not they are same."
    "Reply SAME if the sentences mean same thing, NOT SAME if they are different in meaning."
    "Do not reply anything else."
)


def check_similarity_by_llama3(model, base_sentence, ref_sentence):
    question = f"Sentence 1: {base_sentence}\nSentence 2: {ref_sentence}"
    model_response = model.invoke(system_prompt, question)

    if model_response.lower().strip() == "same":
        return (model_response, True)

    return (model_response, False)


def check_similarity_by_opnai(openai_client, base_sentence, ref_sentence):
    question = f"Sentence 1: {base_sentence}\nSentence 2: {ref_sentence}"
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )

    model_response = response.choices[0].message.content

    if model_response.lower().strip() == "same":
        return (model_response, True)

    return (model_response, False)


def check_similarity_by_cosine_sim(model, base_sentence, ref_sentence, threshold=0.8):
    embeddings1 = model.encode(base_sentence, convert_to_tensor=True)
    embeddings2 = model.encode(ref_sentence, convert_to_tensor=True)

    cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)

    print(cosine_similarity)

    if (cosine_similarity >= threshold).any():
        return True
    else:
        return False


print("Starting evaluation pipeline...")
# plm = SentenceTransformer("bert-base-nli-mean-tokens")
llama3_client = Llama3("meta-llama/Meta-Llama-3-8B-Instruct")
openai_client = OpenAI()

# Argument parsing
parser = argparse.ArgumentParser(
    description="Check similarity between sentences using different models."
)
parser.add_argument(
    "--eval_type", type=str, help="Evaluation type (e.g. llama3, gpt3, etc.)"
)
parser.add_argument("--dataset", type=str, help="Dataset key")
args = parser.parse_args()

eval_type = args.eval_type
dataset = args.dataset

input_data = f"/home/mdafifal.mamun/research/LLMhalu/{eval_type}/data/{eval_type}_outputs_{dataset}.csv"
output_data = f"/home/mdafifal.mamun/research/LLMhalu/{eval_type}/data/{eval_type}_outputs_check_auto_{dataset}.csv"

print(f"Evaluation input: {input_data}")
print(f"Evaluation output: {output_data}")

df = pd.read_csv(input_data)

conditions = {
    (True, True): ("No", "No"),
    (False, False): ("Yes", "No"),
    (True, False): ("Not Sure", "Yes"),
    (False, True): ("Not Sure", "Yes"),
}

for index, row in df.iterrows():
    question = row["Question"]
    answer = row["base_response"]
    best_answer = row["Best Answer"]

    correct_answers = [best_answer]
    correct_answers.extend(row["Correct Answers"].split(";"))

    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"All Correct Answers: {correct_answers}")

    not_sure_flag = False

    df.loc[index, "not_sure_ca_index"] = -1
    df.loc[index, "matched_ca_index"] = (
        -1
    )  # 0 for best answe, any other number os for correct answer index

    for ca_index, correct_answer in enumerate(correct_answers):
        llama_response, llama_sim = check_similarity_by_llama3(
            llama3_client, answer, correct_answer
        )
        openai_reponse, openai_sim = check_similarity_by_opnai(
            openai_client, answer, correct_answer
        )

        # Get the corresponding values from the dictionary
        auto_hallucination_check, manual_check_required = conditions[
            (llama_sim, openai_sim)
        ]  # type: ignore

        print("=========================")
        print(llama_sim, openai_sim)
        print(f"Correct Answer: {correct_answer}")
        print(f"LLM Answer: {answer}")
        print("LLaMA Similarity:", llama_response)
        print("GPT Similarity:", openai_reponse)
        print(f"Auto Hallucination Check: {auto_hallucination_check}")
        print(f"Manual Check Required: {manual_check_required}")

        if (llama_sim, openai_sim) in [(True, False), (False, True)]:
            not_sure_flag = True
            df.loc[index, "not_sure_ca_index"] = ca_index

        # If not sure flag was set for a previous correct answer, we may want to recheck
        if (llama_sim, openai_sim) == (False, False) and not_sure_flag:
            print("Skipping as Not Sure flag is true previously")
            continue

        # Update the DataFrame
        df.loc[index, "auto_hallucination_check"] = auto_hallucination_check
        df.loc[index, "manual_check_required"] = manual_check_required

        # If the fact matches for both llm response, break
        if (llama_sim, openai_sim) == (True, True):
            df.loc[index, "matched_ca_index"] = ca_index
            df.loc[index, "not_sure_ca_index"] = -1
            break


df.to_csv(output_data)
