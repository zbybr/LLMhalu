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
INPUT_DATA = "/home/mdafifal.mamun/research/LLMhalu/TruthfulQA1.3.csv"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
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
        base_response = llama3_model.invoke(
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
            llama3_model.invoke(
                system_prompt=META_SYNONYM_GENERATION_PROMPT, question=qa_pair
            )
        )
        syn_responses = [
            llama3_model.invoke(system_prompt=FACT_VERIFICATION_PROMPT, question=syn)
            for syn in synonyms
        ]

        # Generate antonyms and antonym responses
        antonyms = extract_numbered_list(
            llama3_model.invoke(
                system_prompt=META_ANTONYM_GENERATION_PROMPT, question=qa_pair
            )
        )
        ant_responses = [
            llama3_model.invoke(system_prompt=FACT_VERIFICATION_PROMPT, question=ant)
            for ant in antonyms
        ]

        # # Generate single synonyms
        # single_synonyms = []
        # single_synonym_responses = []
        # for i in range(5):
        #     syn = llama3_model.invoke(
        #         META_SINGLE_SYNONYM_GENERATION_PROMPT, qa_pair, temperature=0.7
        #     )
        #     resp = llama3_model.invoke(FACT_VERIFICATION_PROMPT, syn, temperature=0.1)
        #     print(f"Generated Single Synonym {i + 1}: {syn}, Response: {resp}")
        #     single_synonyms.append(syn)
        #     single_synonym_responses.append(resp)

        # # Generate single antonyms
        # single_antonyms = []
        # single_antonym_responses = []
        # for i in range(5):
        #     ant = llama3_model.invoke(
        #         META_SINGLE_ANTONYM_GENERATION_PROMPT, qa_pair, temperature=0.7
        #     )
        #     resp = llama3_model.invoke(FACT_VERIFICATION_PROMPT, ant, temperature=0.1)
        #     print(f"Generated Single Antonym {i + 1}: {ant}, Response: {resp}")
        #     single_antonyms.append(ant)
        #     single_antonym_responses.append(resp)

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

    # Save output data
    df.to_csv(output_path)
    print("Output saved.")


if __name__ == "__main__":
    SEED = 42
    SAMPLES = 100
    TOTAL_RUNS = 3

    df = pd.read_csv(INPUT_DATA).sample(SAMPLES, random_state=SEED)

    for i in range(TOTAL_RUNS):
        print(f"Run: {i+1}/{TOTAL_RUNS}")
        OUTPUT_DATA = f"/home/mdafifal.mamun/research/LLMhalu/llama3/data/final_responses/truthfulqa/std/llama3_outputs_truthfulqa1.3_{SAMPLES}samples_seed{SEED}_run{i+1}.csv"
        run_pipeline(df, OUTPUT_DATA)
