import sys

sys.path.append("/home/mdafifal.mamun/research/LLMhalu")

import pandas as pd
import spacy
import torch
from dotenv import load_dotenv
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt
from tqdm import tqdm

import llm_prompts.prompts as prompts
from mistral import Mistral

load_dotenv()

nlp = spacy.load("en_core_web_sm")

# Constants
INPUT_DATA = "/home/mdafifal.mamun/research/LLMhalu/TruthfulQA1.3.csv"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# selfcheck_prompt = SelfCheckLLMPrompt(MODEL_ID, device)

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


# Initializing Mistral pipeline
print("Preparing Mistral pipeline...")
mistral_model = Mistral(MODEL_ID, temperature=0.1)
# mistral_model = mistral_model.cuda()


def run_pipeline(df, output_path):
    print("Generating Responses...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
        question = row["Question"]
        print(f"Question: {question}")

        # Generate base response
        system_prompt = prompts.SYSTEM_PROMPT
        base_response = mistral_model.invoke(
            system_prompt=system_prompt, question=question
        )

        # sentences = [sent.text.strip() for sent in nlp(base_response).sents]

        # Generate samples for selfcheck
        # generated_samples = []
        # for i in range(10):
        #     base_response = llama3_model.invoke(
        #         system_prompt=system_prompt, question=question
        #     )
        #     print(f"Sample {i}: {base_response}")
        #     generated_samples.append(base_response)

        # sent_scores_prompt = selfcheck_prompt.predict(
        #     sentences=sentences,
        #     sampled_passages=generated_samples,
        #     verbose=True,
        # )

        # # For exception handling
        # if len(sent_scores_prompt) > 1:
        #     print("Selfcheck score: ", sent_scores_prompt)
        #     print("Found exception!! Considering only the first sentence.")
        #     sent_scores_prompt = sent_scores_prompt[0]

        # Generate synonyms and synonym responses
        qa_pair = f"Question: {question} Answer: {base_response}"
        synonyms = extract_numbered_list(
            mistral_model.invoke(
                system_prompt=META_SYNONYM_GENERATION_PROMPT, question=qa_pair
            )
        )

        syn_responses = [
            mistral_model.invoke(system_prompt=FACT_VERIFICATION_PROMPT, question=syn)
            for syn in synonyms
        ]

        # Generate antonyms and antonym responses
        antonyms = extract_numbered_list(
            mistral_model.invoke(
                system_prompt=META_ANTONYM_GENERATION_PROMPT, question=qa_pair
            )
        )
        ant_responses = [
            mistral_model.invoke(system_prompt=FACT_VERIFICATION_PROMPT, question=ant)
            for ant in antonyms
        ]

        # Print responses
        print(f"Response: {base_response}")
        # print(f"Generated samples: {generated_samples}")
        print(f"Synonyms:\n{synonyms}")
        print(f"Synonym Responses:\n{syn_responses}")
        print(f"Antonyms:\n{antonyms}")
        print(f"Antonym Responses:\n{ant_responses}")

        # Update DataFrame with responses
        df.loc[index, "base_response"] = base_response

        # df.loc[index, "generated_samples"] = ";".join(generated_samples)
        # df.loc[index, "selfcheck_score"] = sent_scores_prompt

        df.loc[index, "synonyms"] = ";".join(synonyms)
        df.loc[index, "synonym_responses"] = ";".join(syn_responses)
        df.loc[index, "antonyms"] = ";".join(antonyms)
        df.loc[index, "antonym_responses"] = ";".join(ant_responses)

        print("===================================\n")

        df.to_csv(output_path)
        print(f"Output saved at index {index}.")

    # Save output data
    df.to_csv(output_path)
    print("Output saved.")


if __name__ == "__main__":
    SEED = 42
    SAMPLES = 5
    TOTAL_RUNS = 1

    df = pd.read_csv(INPUT_DATA).sample(SAMPLES, random_state=SEED)

    for i in range(TOTAL_RUNS):
        print(f"Run: {i+1}/{TOTAL_RUNS}")
        OUTPUT_DATA = f"/home/mdafifal.mamun/research/LLMhalu/llama3/data/final_responses/truthfulqaq.3_temp0.1.csv"
        run_pipeline(df, OUTPUT_DATA)
