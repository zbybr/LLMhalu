import sys

sys.path.append("/home/mdafifal.mamun/research/LLMhalu/")


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from dotenv import load_dotenv
from openai import OpenAI
from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt
from tqdm import tqdm

import llm_prompts.prompts as prompts

load_dotenv()

NUM_SAMPLES = 200
SEED = 42

nlp = spacy.load("en_core_web_sm")

parser = argparse.ArgumentParser(description="GPT pipeline.")
parser.add_argument("--dataset_path", type=str, help="Dataset path", default=None)
args = parser.parse_args()

# Constants
dataset_path = args.dataset_path
dataset_name = str(Path(dataset_path).stem).lower()

INPUT_DATA = dataset_path
OUTPUT_DATA = (
    f"/home/mdafifal.mamun/research/LLMhalu/gpt3/data/gpt3_outputs_{dataset_name}.csv"
)

print("Preparing GPT pipeline...")
print(f"Input dataset: {INPUT_DATA}")
print(f"Output dataset: {OUTPUT_DATA}")
print(f"Random state: {SEED}")

GPT_MODEL_KEY = "gpt-3.5-turbo"

selfcheck_prompt = SelfCheckAPIPrompt()

META_SYNONYM_GENERATION_PROMPT = prompts.META_SYNONYM_GENERATION_PROMPT

META_ANTONYM_GENERATION_PROMPT = prompts.META_ANTONYM_GENERATION_PROMPT

FACT_VERIFICATION_PROMPT = prompts.FACT_VERIFICATION_PROMPT


def extract_numbered_list(text):
    return [
        line.strip()
        for line in text.split("\n")
        if line.strip().startswith(tuple(str(i) + "." for i in range(10)))
    ]


def get_gpt_response(prompt, question):
    gpt_model = OpenAI()
    response = gpt_model.chat.completions.create(
        model=GPT_MODEL_KEY,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    df = pd.read_excel(INPUT_DATA, sheet_name="freshqa")
    df = df[df["fact_type"] == "never-changing"]

    print("Generating Responses...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
        question = row["question"]
        print(f"Question: {question}")

        # Generate base response
        system_prompt = prompts.SYSTEM_PROMPT
        base_response = get_gpt_response(system_prompt, question)

        sentences = [sent.text.strip() for sent in nlp(base_response).sents]

        # Generate samples for selfcheck
        generated_samples = []
        for i in range(10):
            base_response = get_gpt_response(system_prompt, question)
            print(f"Sample {i}: {base_response}")
            generated_samples.append(base_response)

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
        resp = get_gpt_response(META_SYNONYM_GENERATION_PROMPT, qa_pair)
        synonyms = extract_numbered_list(resp)
        syn_responses = [
            get_gpt_response(FACT_VERIFICATION_PROMPT, syn) for syn in synonyms
        ]

        # Generate antonyms and antonym responses
        resp = get_gpt_response(META_ANTONYM_GENERATION_PROMPT, qa_pair)
        antonyms = extract_numbered_list(resp)
        ant_responses = [
            get_gpt_response(FACT_VERIFICATION_PROMPT, ant) for ant in antonyms
        ]

        # Print responses
        print(f"Response: {base_response}")
        print(f"Generated samples: {generated_samples}")
        print(f"Synonyms:\n{synonyms}")
        print(f"Synonym Responses:\n{syn_responses}")
        print(f"Antonyms:\n{antonyms}")
        print(f"Antonym Responses:\n{ant_responses}")

        # Update DataFrame with responses
        df.loc[index, "base_response"] = base_response
        df.loc[index, "synonyms"] = ";".join(synonyms)
        df.loc[index, "synonym_responses"] = ";".join(syn_responses)
        df.loc[index, "generated_samples"] = ";".join(generated_samples)
        df.loc[index, "antonyms"] = ";".join(antonyms)
        df.loc[index, "antonym_responses"] = ";".join(ant_responses)
        df.loc[index, "generated_samples"] = ";".join(generated_samples)
        df.loc[index, "selfcheck_score"] = sent_scores_prompt

        print("===================================\n")

    # Save output data
    df.to_csv(OUTPUT_DATA, index=False)
    print("Output saved.")
