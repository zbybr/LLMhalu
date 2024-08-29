import argparse
from pathlib import Path
import pandas as pd
import spacy
from dotenv import load_dotenv
from openai import OpenAI
from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt
from tqdm import tqdm
import sys
sys.path.append("/Users/afif/Work/projects/LLMhalu")

import llm_prompts.prompts as prompts


load_dotenv()

nlp = spacy.load("en_core_web_sm")

parser = argparse.ArgumentParser(description="GPT pipeline.")
parser.add_argument("--dataset_path", type=str, help="Dataset path")
args = parser.parse_args()

# Constants
dataset_path = args.dataset_path
dataset_name = str(Path(dataset_path).stem).lower()

INPUT_DATA = dataset_path
OUTPUT_DATA = (
    f"gpt4/data/gpt4_outputs_{dataset_name}.csv"
)

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

    return response.choices[0].message.content


if __name__ == "__main__":
    df = pd.read_csv(INPUT_DATA).sample(2, random_state=42)

    print("Generating Responses...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
        question = row["Question"]
        print(f"Question: {question}")

        # Generate base response
        system_prompt = prompts.SYSTEM_PROMPT
        base_response = get_gpt_response(system_prompt, question)

        sentences = [sent.text.strip() for sent in nlp(base_response).sents]

        # Generate samples for selfcheck
        generated_samples = []
        for i in range(10):
            base_response = get_gpt_response(system_prompt, question, temperature=0.5)
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
        resp = get_gpt_response(META_SYNONYM_GENERATION_PROMPT, qa_pair, temperature=0.5)
        synonyms = extract_numbered_list(resp)
        syn_responses = [
            get_gpt_response(FACT_VERIFICATION_PROMPT, syn, temperature=0.0) for syn in synonyms
        ]

        # Generate antonyms and antonym responses
        resp = get_gpt_response(META_ANTONYM_GENERATION_PROMPT, qa_pair, temperature=0.5)
        antonyms = extract_numbered_list(resp)
        ant_responses = [
            get_gpt_response(FACT_VERIFICATION_PROMPT, ant, temperature=0.0) for ant in antonyms
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
        df.loc[index, "generated_samples"] = ";".join(generated_samples)
        df.loc[index, "synonyms"] = ";".join(synonyms)
        df.loc[index, "synonym_responses"] = ";".join(syn_responses)
        df.loc[index, "antonyms"] = ";".join(antonyms)
        df.loc[index, "antonym_responses"] = ";".join(ant_responses)
        df.loc[index, "generated_samples"] = ";".join(generated_samples)
        df.loc[index, "selfcheck_score"] = sent_scores_prompt

        print("===================================\n")

    # Save output data
    df.to_csv(OUTPUT_DATA)
    print("Output saved.")
