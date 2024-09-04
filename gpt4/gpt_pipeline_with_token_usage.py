import argparse
import sys
from pathlib import Path

import pandas as pd
import spacy
from dotenv import load_dotenv
from openai import OpenAI
from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt
from tqdm import tqdm

sys.path.append("/home/mdafifal.mamun/research/LLMhalu/")

import llm_prompts.prompts as prompts

# from util import token_counter


load_dotenv()

nlp = spacy.load("en_core_web_sm")

parser = argparse.ArgumentParser(description="GPT pipeline.")
parser.add_argument(
    "--dataset_path",
    default="/home/mdafifal.mamun/research/LLMhalu/datasets/TruthfulQA1.3.csv",
    type=str,
    help="Dataset path",
)
args = parser.parse_args()

# Constants
dataset_path = args.dataset_path
dataset_name = str(Path(dataset_path).stem).lower()
SEED = 77

INPUT_DATA = dataset_path
OUTPUT_DATA = f"/home/mdafifal.mamun/research/LLMhalu/gpt4/data/gpt4_outputs_{dataset_name}_token_usage_seed{SEED}.csv"

# Initializing Llama3 pipeline
print("Preparing GPT pipeline...")
print(f"Input dataset: {INPUT_DATA}")
print(f"Output dataset: {OUTPUT_DATA}")

GPT_MODEL_KEY = "gpt-4o"

selfcheck_prompt = SelfCheckAPIPrompt(model=GPT_MODEL_KEY)

META_SYNONYM_GENERATION_PROMPT = prompts.META_SYNONYM_GENERATION_PROMPT
META_ANTONYM_GENERATION_PROMPT = prompts.META_ANTONYM_GENERATION_PROMPT
META_SINGLE_SYNONYM_GENERATION_PROMPT = prompts.META_SINGLE_SYNONYM_GENERATION_PROMPT
META_SINGLE_ANTONYM_GENERATION_PROMPT = prompts.META_SINGLE_ANTONYM_GENERATION_PROMPT
FACT_VERIFICATION_PROMPT = prompts.FACT_VERIFICATION_PROMPT


def extract_numbered_list(text):
    return [
        line.strip()
        for line in text.split("\n")
        if line.strip().startswith(tuple(str(i) + "." for i in range(10)))
    ]


# TODO This function need to be integrated to selfcheck api library for token usage comparison
def add_token_usage(base_usage, new_usage):
    for key in new_usage:
        if key in base_usage:
            base_usage[key] += new_usage[key]

    return base_usage


def get_gpt_response(prompt, question, temperature=0.0):
    gpt_model = OpenAI()
    response = gpt_model.chat.completions.create(
        model=GPT_MODEL_KEY,
        temperature=temperature,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )

    token_usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return response.choices[0].message.content, token_usage


if __name__ == "__main__":
    df = pd.read_csv(INPUT_DATA).sample(100, random_state=SEED)

    print("Generating Responses...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
        question = row["Question"]
        print(f"Question: {question}")

        # Generate base response
        system_prompt = prompts.SYSTEM_PROMPT
        base_response, base_token_usage = get_gpt_response(system_prompt, question)

        sentences = [sent.text.strip() for sent in nlp(base_response).sents]

        # Generate samples for selfcheck
        generated_samples = []

        selfchkgpt_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for i in range(10):
            base_response, token_usage = get_gpt_response(
                system_prompt, question, temperature=0.5
            )
            print(f"Sample {i}: {base_response}")

            generated_samples.append(base_response)
            selfchkgpt_token_usage = add_token_usage(
                selfchkgpt_token_usage, token_usage
            )

        sent_scores_prompt, token_usage = selfcheck_prompt.predict(
            sentences=sentences,
            sampled_passages=generated_samples,
            verbose=True,
        )
        # print(token_usage)

        selfchkgpt_token_usage = add_token_usage(selfchkgpt_token_usage, token_usage)

        # For exception handling
        if len(sent_scores_prompt) > 1:
            print("Selfcheck score: ", sent_scores_prompt)
            print("Found exception!! Considering only the first sentence.")
            sent_scores_prompt = sent_scores_prompt[0]

        # Generate synonyms and synonym responses
        qa_pair = f"Question: {question} Answer: {base_response}"
        previous_syno_anto_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        resp, token_usage = get_gpt_response(
            META_SYNONYM_GENERATION_PROMPT, qa_pair, temperature=0.5
        )
        previous_syno_anto_usage = add_token_usage(
            previous_syno_anto_usage, token_usage
        )

        synonyms = extract_numbered_list(resp)
        syn_responses = []

        for syn in synonyms:
            syn_resp, token_usage = get_gpt_response(
                FACT_VERIFICATION_PROMPT, syn, temperature=0.0
            )
            previous_syno_anto_usage = add_token_usage(
                previous_syno_anto_usage, token_usage
            )
            syn_responses.append(syn_resp)

        # Generate antonyms and antonym responses
        resp, token_usage = get_gpt_response(
            META_ANTONYM_GENERATION_PROMPT, qa_pair, temperature=0.5
        )
        previous_syno_anto_usage = add_token_usage(
            previous_syno_anto_usage, token_usage
        )

        antonyms = extract_numbered_list(resp)
        ant_responses = []

        for ant in antonyms:
            ant_resp, token_usage = get_gpt_response(
                FACT_VERIFICATION_PROMPT, ant, temperature=0.0
            )
            previous_syno_anto_usage = add_token_usage(
                previous_syno_anto_usage, token_usage
            )
            ant_responses.append(ant_resp)

        # Print responses
        print(f"Response: {base_response}")
        print(f"Generated samples: {generated_samples}")
        print(f"Synonyms:\n{synonyms}")
        print(f"Synonym Responses:\n{syn_responses}")
        print(f"Antonyms:\n{antonyms}")
        print(f"Antonym Responses:\n{ant_responses}")

        # Update DataFrame with responses
        df.loc[index, "base_response"] = base_response
        df.loc[index, "generated_samples"] = ";".join(generated_samples)
        df.loc[index, "synonyms"] = ";".join(synonyms)
        df.loc[index, "synonym_responses"] = ";".join(syn_responses)
        df.loc[index, "antonyms"] = ";".join(antonyms)
        df.loc[index, "antonym_responses"] = ";".join(ant_responses)
        df.loc[index, "generated_samples"] = ";".join(generated_samples)
        df.loc[index, "selfcheck_score"] = sent_scores_prompt
        df.loc[index, "base_token_usage"] = str(base_token_usage)
        df.loc[index, "selfcheck_token_usage"] = str(selfchkgpt_token_usage)
        df.loc[index, "metaqa_token_usage"] = str(previous_syno_anto_usage)

        print("===================================\n")
        df.to_csv(OUTPUT_DATA)

    # Save output data
    df.to_csv(OUTPUT_DATA)
    print("Output saved.")
