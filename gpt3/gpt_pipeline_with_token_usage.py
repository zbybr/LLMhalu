import argparse
from pathlib import Path

import pandas as pd
import spacy
from dotenv import load_dotenv
from openai import OpenAI
from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt
from tqdm import tqdm

import sys
sys.path.append("D:\\Projects\\LLMhalu")

import llm_prompts.prompts as prompts
from util import token_counter


load_dotenv()

nlp = spacy.load("en_core_web_sm")

parser = argparse.ArgumentParser(description="GPT pipeline.")
parser.add_argument("--dataset_path", default="D:\\Projects\\LLMhalu\\TruthfulQA1.2.csv", type=str, help="Dataset path")
args = parser.parse_args()

# Constants
dataset_path = args.dataset_path
dataset_name = str(Path(dataset_path).stem).lower()
SEED = 77

INPUT_DATA = dataset_path
OUTPUT_DATA = (
    f"data/gpt3_outputs_{dataset_name}_token_usage_seed{SEED}.csv"
)

# Initializing Llama3 pipeline
print("Preparing GPT pipeline...")
print(f"Input dataset: {INPUT_DATA}")
print(f"Output dataset: {OUTPUT_DATA}")

GPT_MODEL_KEY = "gpt-3.5-turbo"

selfcheck_prompt = SelfCheckAPIPrompt()

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
        "total_tokens": response.usage.total_tokens
    }

    return response.choices[0].message.content, token_usage


if __name__ == "__main__":
    df = pd.read_csv(INPUT_DATA).sample(10, random_state=SEED)

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
            base_response, token_usage = get_gpt_response(system_prompt, question, temperature=0.5)
            print(f"Sample {i}: {base_response}")

            generated_samples.append(base_response)
            selfchkgpt_token_usage = add_token_usage(selfchkgpt_token_usage, token_usage)

        sent_scores_prompt = selfcheck_prompt.predict(
            sentences=sentences,
            sampled_passages=generated_samples,
            verbose=True,
        )

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

        resp, token_usage = get_gpt_response(META_SYNONYM_GENERATION_PROMPT, qa_pair, temperature=0.5)
        previous_syno_anto_usage = add_token_usage(previous_syno_anto_usage, token_usage)

        synonyms = extract_numbered_list(resp)
        syn_responses = []

        for syn in synonyms:
            syn_resp, token_usage = get_gpt_response(FACT_VERIFICATION_PROMPT, syn, temperature=0.0)
            previous_syno_anto_usage = add_token_usage(previous_syno_anto_usage, token_usage)
            syn_responses.append(syn_resp)

        # Generate antonyms and antonym responses
        resp, token_usage = get_gpt_response(META_ANTONYM_GENERATION_PROMPT, qa_pair, temperature=0.5)
        previous_syno_anto_usage = add_token_usage(previous_syno_anto_usage, token_usage)

        antonyms = extract_numbered_list(resp)
        ant_responses = []

        for ant in antonyms:
            ant_resp, token_usage = get_gpt_response(FACT_VERIFICATION_PROMPT, ant, temperature=0.0)
            previous_syno_anto_usage = add_token_usage(previous_syno_anto_usage, token_usage)
            ant_responses.append(ant_resp)

        # Generate single synonyms
        single_synonyms = []
        single_synonym_responses = []
        current_syno_anto_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        for i in range(5):
            syn, token_usage = get_gpt_response(META_SINGLE_SYNONYM_GENERATION_PROMPT, qa_pair, temperature=0.7)
            current_syno_anto_usage = add_token_usage(current_syno_anto_usage, token_usage)
            resp, token_usage = get_gpt_response(FACT_VERIFICATION_PROMPT, syn, temperature=0.0)
            current_syno_anto_usage = add_token_usage(current_syno_anto_usage, token_usage)
            single_synonyms.append(syn)
            single_synonym_responses.append(resp)


        # Generate single antonyms
        single_antonyms = []
        single_antonym_responses = []
        for i in range(5):
            ant, token_usage = get_gpt_response(META_SINGLE_ANTONYM_GENERATION_PROMPT, qa_pair, temperature=0.7)
            current_syno_anto_usage = add_token_usage(current_syno_anto_usage, token_usage)
            resp, token_usage = get_gpt_response(FACT_VERIFICATION_PROMPT, ant, temperature=0.0)
            current_syno_anto_usage = add_token_usage(current_syno_anto_usage, token_usage)
            single_antonyms.append(ant)
            single_antonym_responses.append(resp)

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
        df.loc[index, "single_synonyms"] = ";".join(single_synonyms)
        df.loc[index, "single_synonym_responses"] = ";".join(single_synonym_responses)
        df.loc[index, "antonyms"] = ";".join(antonyms)
        df.loc[index, "antonym_responses"] = ";".join(ant_responses)
        df.loc[index, "single_antonyms"] = ";".join(single_antonyms)
        df.loc[index, "single_antonym_responses"] = ";".join(single_antonym_responses)
        df.loc[index, "generated_samples"] = ";".join(generated_samples)
        df.loc[index, "selfcheck_score"] = sent_scores_prompt
        df.loc[index, "base_token_usage"] = str(base_token_usage)
        df.loc[index, "selfcheck_token_usage"] = str(selfchkgpt_token_usage)
        df.loc[index, "previous_synant_token_usage"] = str(previous_syno_anto_usage)
        df.loc[index, "current_synant_token_usage"] = str(current_syno_anto_usage)

        print("===================================\n")

    # Save output data
    df.to_csv(OUTPUT_DATA)
    print("Output saved.")
