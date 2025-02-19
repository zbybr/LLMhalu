import sys
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

client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key='sk-cV7MZTd97HxBFHYvQ6pSjrKM8o26hFLv9rJbbqYY9XN1xMxU',
)

nlp = spacy.load("en_core_web_sm")

parser = argparse.ArgumentParser(description="GPT pipeline.")
parser.add_argument("--dataset_path", type=str, help="Dataset path", default=None)
args = parser.parse_args()

# Constants
INPUT_DATA = "testsamples.csv"
OUTPUT_DATA = "selfcheckgpttest.csv"

print("Preparing GPT pipeline...")
print(f"Input dataset: {INPUT_DATA}")
print(f"Output dataset: {OUTPUT_DATA}")

GPT_MODEL_KEY = "gpt-3.5-turbo"

selfcheck_prompt = SelfCheckAPIPrompt()

def extract_numbered_list(text):
    return [
        line.strip()
        for line in text.split("\n")
        if line.strip().startswith(tuple(str(i) + "." for i in range(10)))
    ]


def get_gpt_response(prompt, question, temperature=0.0):
    response = client.chat.completions.create(
        model=GPT_MODEL_KEY,
        temperature=temperature,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    df = pd.read_csv(INPUT_DATA)

    print("Generating Responses...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
        question = row["Question"]
        print(f"Question: {question}")

        # Generate base response
        system_prompt = prompts.SYSTEM_PROMPT
        base_response = get_gpt_response(system_prompt, question)

        sentences = [sent.text.strip() for sent in nlp(base_response).sents]

        # Generate samples for selfcheck
        generated_samples_0 = []
        for i in range(10):
            base_response = get_gpt_response(system_prompt, question, temperature=0.0)
            print(f"Sample {i}: {base_response}")
            generated_samples_0.append(base_response)

        generated_samples_5 = []
        for i in range(10):
            base_response = get_gpt_response(system_prompt, question, temperature=0.5)
            print(f"Sample {i}: {base_response}")
            generated_samples_5.append(base_response)

        generated_samples_10 = []
        for i in range(10):
            base_response = get_gpt_response(system_prompt, question, temperature=1.0)
            print(f"Sample {i}: {base_response}")
            generated_samples_10.append(base_response)

        sent_scores_prompt_0 = selfcheck_prompt.predict(
            sentences=sentences,
            sampled_passages=generated_samples_0,
            verbose=True,
        )

        sent_scores_prompt_5 = selfcheck_prompt.predict(
            sentences=sentences,
            sampled_passages=generated_samples_5,
            verbose=True,
        )

        sent_scores_prompt_10 = selfcheck_prompt.predict(
            sentences=sentences,
            sampled_passages=generated_samples_10,
            verbose=True,
        )
        # For exception handling
        if len(sent_scores_prompt_0) > 1:
            print("Selfcheck score: ", sent_scores_prompt_0)
            print("Found exception!! Considering only the first sentence.")
            sent_scores_prompt_0 = sent_scores_prompt_0[0]
        if len(sent_scores_prompt_5) > 1:
            print("Selfcheck score: ", sent_scores_prompt_5)
            print("Found exception!! Considering only the first sentence.")
            sent_scores_prompt_5 = sent_scores_prompt_5[0]
        if len(sent_scores_prompt_10) > 1:
            print("Selfcheck score: ", sent_scores_prompt_10)
            print("Found exception!! Considering only the first sentence.")
            sent_scores_prompt_10 = sent_scores_prompt_10[0]

        # Print responses
        print(f"Response: {base_response}")
        print(f"Socre1: {sent_scores_prompt_0}")
        print(f"Score2: {sent_scores_prompt_5}")
        print(f"Score3: {sent_scores_prompt_10}")

        # Update DataFrame with responses
        df.loc[index, "base_response"] = base_response
        df.loc[index, "generated_samples"] = ";".join(generated_samples_0)
        df.loc[index, "selfcheck_score"] = sent_scores_prompt_0
        df.loc[index, "generated_samples"] = ";".join(generated_samples_5)
        df.loc[index, "selfcheck_score"] = sent_scores_prompt_5
        df.loc[index, "generated_samples"] = ";".join(generated_samples_10)
        df.loc[index, "selfcheck_score"] = sent_scores_prompt_10

        print("===================================\n")

    # Save output data
    df.to_csv(OUTPUT_DATA)
    print("Output saved.")
