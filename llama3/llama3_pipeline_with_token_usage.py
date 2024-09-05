import argparse
import sys

sys.path.append("/home/mdafifal.mamun/research/LLMhalu")

import pandas as pd
import spacy
import torch
from dotenv import load_dotenv
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt
from tqdm import tqdm

import llm_prompts.prompts as prompts
from llama3 import Llama3

load_dotenv()

nlp = spacy.load("en_core_web_sm")

# Constants
SEED = 77
temperature = 0.1
INPUT_DATA = "/home/mdafifal.mamun/research/LLMhalu/TruthfulQA1.3.csv"
OUTPUT_DATA = f"/home/mdafifal.mamun/research/LLMhalu/llama3/data/llama3_outputs_truthfulqa_token_usage_seed{SEED}.csv"

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

selfcheck_prompt = SelfCheckLLMPrompt(MODEL_ID, device)

META_SYNONYM_GENERATION_PROMPT = prompts.META_SYNONYM_GENERATION_PROMPT
META_ANTONYM_GENERATION_PROMPT = prompts.META_ANTONYM_GENERATION_PROMPT
META_SINGLE_SYNONYM_GENERATION_PROMPT = prompts.META_SINGLE_SYNONYM_GENERATION_PROMPT
META_SINGLE_ANTONYM_GENERATION_PROMPT = prompts.META_SINGLE_ANTONYM_GENERATION_PROMPT
FACT_VERIFICATION_PROMPT = prompts.FACT_VERIFICATION_PROMPT


# TODO This function need to be integrated to selfcheck api library for token usage comparison
def add_token_usage(base_usage, new_usage):
    for key in new_usage:
        if key in base_usage:
            base_usage[key] += new_usage[key]

    return base_usage


def extract_numbered_list(text):
    return [
        line.strip()
        for line in text.split("\n")
        if line.strip().startswith(tuple(str(i) + "." for i in range(10)))
    ]


# Initializing Llama3 pipeline
print("Preparing Llama3 pipeline...")
llama3_model = Llama3(MODEL_ID)


def run_pipeline(df, output_path):
    print("Generating Responses...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
        question = row["Question"]
        print(f"Question: {question}")

        # Generate base response
        system_prompt = prompts.SYSTEM_PROMPT
        base_response, base_token_usage = llama3_model.invoke(
            system_prompt=system_prompt,
            question=question,
            temperature=temperature,
            return_token_usage=True,
        )

        sentences = [sent.text.strip() for sent in nlp(base_response).sents]

        # Generate samples for selfcheck
        selfchkgpt_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        generated_samples = []
        for i in range(10):
            base_response, token_usage = llama3_model.invoke(
                system_prompt=system_prompt, question=question, return_token_usage=True
            )
            selfchkgpt_token_usage = add_token_usage(
                selfchkgpt_token_usage, token_usage
            )
            print(f"Sample {i}: {base_response}")
            generated_samples.append(base_response)

        sent_scores_prompt, token_usage = selfcheck_prompt.predict(
            sentences=sentences,
            sampled_passages=generated_samples,
            verbose=True,
        )
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

        response, token_usage = llama3_model.invoke(
            system_prompt=META_SYNONYM_GENERATION_PROMPT,
            question=qa_pair,
            temperature=temperature,
            return_token_usage=True,
        )
        previous_syno_anto_usage = add_token_usage(
            previous_syno_anto_usage, token_usage
        )

        synonyms = extract_numbered_list(response)
        syn_responses = []

        for syn in synonyms:
            response, token_usage = llama3_model.invoke(
                system_prompt=FACT_VERIFICATION_PROMPT,
                question=syn,
                temperature=0.1,
                return_token_usage=True,
            )
            syn_responses.append(syn)
            previous_syno_anto_usage = add_token_usage(
                previous_syno_anto_usage, token_usage
            )

        # Generate antonyms and antonym responses
        response, token_usage = llama3_model.invoke(
            system_prompt=META_ANTONYM_GENERATION_PROMPT,
            question=qa_pair,
            temperature=temperature,
            return_token_usage=True,
        )
        previous_syno_anto_usage = add_token_usage(
            previous_syno_anto_usage, token_usage
        )

        antonyms = extract_numbered_list(response)
        ant_responses = []

        for ant in antonyms:
            response, token_usage = llama3_model.invoke(
                system_prompt=FACT_VERIFICATION_PROMPT,
                question=ant,
                temperature=0.1,
                return_token_usage=True,
            )
            ant_responses.append(ant)
            previous_syno_anto_usage = add_token_usage(
                previous_syno_anto_usage, token_usage
            )

        # Update DataFrame with responses
        df.loc[index, "base_response"] = base_response

        df.loc[index, "generated_samples"] = ";".join(generated_samples)
        df.loc[index, "selfcheck_score"] = sent_scores_prompt

        df.loc[index, "synonyms"] = ";".join(synonyms)
        df.loc[index, "synonym_responses"] = ";".join(syn_responses)
        df.loc[index, "antonyms"] = ";".join(antonyms)
        df.loc[index, "antonym_responses"] = ";".join(ant_responses)
        df.loc[index, "base_token_usage"] = str(base_token_usage)
        df.loc[index, "selfcheck_token_usage"] = str(selfchkgpt_token_usage)
        df.loc[index, "metaqa_token_usage"] = str(previous_syno_anto_usage)

        print("===================================\n")

        df.to_csv(output_path)
        print(f"Output saved at index {index}.")

    # Save output data
    df.to_csv(output_path)
    print("Output saved.")


if __name__ == "__main__":
    SAMPLES = 100
    TOTAL_RUNS = 1

    df = pd.read_csv(INPUT_DATA).sample(SAMPLES, random_state=SEED)
    run_pipeline(df, OUTPUT_DATA)
