import argparse
import time
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

load_dotenv()

nlp = spacy.load("en_core_web_sm")
SEED = 77

# Initializing Llama3 pipeline
print("Preparing GPT pipeline...")

GPT_MODEL_KEY = "gpt-3.5-turbo-0613"
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


def run_pipeline(input_path, output_path):
    print(f"Input dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Output dataset: {output_path}")
    print("Generating Responses...")

    every_five_response_flag = 0

    try:
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
            question = row["Question"]
            print(f"Question: {question}")
            every_five_response_flag += 1

            # Generate base response
            system_prompt = prompts.SYSTEM_PROMPT
            base_response = get_gpt_response(system_prompt, question)

            sentences = [sent.text.strip() for sent in nlp(base_response).sents]

            # Generate samples for selfcheck
            selfcheck_time = time.time()
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
            selfcheck_time = time.time() - selfcheck_time

            # For exception handling
            if len(sent_scores_prompt) > 1:
                print("Selfcheck score: ", sent_scores_prompt)
                print("Found exception!! Considering only the first sentence.")
                sent_scores_prompt = sent_scores_prompt[0]

            # # Generate synonyms and synonym responses
            # prev_approach_time = time.time()
            qa_pair = f"Question: {question} Answer: {base_response}"
            # resp = get_gpt_response(META_SYNONYM_GENERATION_PROMPT, qa_pair, temperature=0.5)
            # synonyms = extract_numbered_list(resp)
            # syn_responses = [
            #     get_gpt_response(FACT_VERIFICATION_PROMPT, syn, temperature=0.0) for syn in synonyms
            # ]
            #
            # # Generate antonyms and antonym responses
            # resp = get_gpt_response(META_ANTONYM_GENERATION_PROMPT, qa_pair, temperature=0.5)
            # antonyms = extract_numbered_list(resp)
            # ant_responses = [
            #     get_gpt_response(FACT_VERIFICATION_PROMPT, ant, temperature=0.0) for ant in antonyms
            # ]
            # prev_approach_time = time.time() - prev_approach_time

            # Generate single synonyms
            new_approach_time = time.time()
            single_synonyms = []
            single_synonym_responses = []
            for i in range(5):
                syn = get_gpt_response(META_SINGLE_SYNONYM_GENERATION_PROMPT, qa_pair, temperature=0.7)
                resp = get_gpt_response(FACT_VERIFICATION_PROMPT, syn, temperature=0.0)
                single_synonyms.append(syn)
                single_synonym_responses.append(resp)

            # Generate single antonyms
            single_antonyms = []
            single_antonym_responses = []
            for i in range(5):
                ant = get_gpt_response(META_SINGLE_ANTONYM_GENERATION_PROMPT, qa_pair, temperature=0.7)
                resp = get_gpt_response(FACT_VERIFICATION_PROMPT, ant, temperature=0.0)
                single_antonyms.append(ant)
                single_antonym_responses.append(resp)
            new_approach_time = time.time() - new_approach_time

            # Print responses
            print(f"Response: {base_response}")
            print(f"Generated samples: {generated_samples}")
            print(f"Single Synonyms:\n{single_synonyms}")
            print(f"Synonym Responses:\n{single_synonym_responses}")
            print(f"Single Antonyms:\n{single_antonyms}")
            print(f"Antonym Responses:\n{single_antonym_responses}")

            # Update DataFrame with responses
            df.loc[index, "base_response"] = base_response
            df.loc[index, "generated_samples"] = ";".join(generated_samples)
            # df.loc[index, "synonyms"] = ";".join(synonyms)
            # df.loc[index, "synonym_responses"] = ";".join(syn_responses)
            df.loc[index, "single_synonyms"] = ";".join(single_synonyms)
            df.loc[index, "single_synonym_responses"] = ";".join(single_synonym_responses)
            # df.loc[index, "antonyms"] = ";".join(antonyms)
            # df.loc[index, "antonym_responses"] = ";".join(ant_responses)
            df.loc[index, "single_antonyms"] = ";".join(single_antonyms)
            df.loc[index, "single_antonym_responses"] = ";".join(single_antonym_responses)
            df.loc[index, "generated_samples"] = ";".join(generated_samples)
            df.loc[index, "selfcheck_score"] = sent_scores_prompt
            df.loc[index, "selfcheck_time"] = selfcheck_time
            # df.loc[index, "prev_approach_time"] = prev_approach_time
            df.loc[index, "new_approach_time"] = new_approach_time

            print("===================================\n")

            if every_five_response_flag >= 5:
                every_five_response_flag = 0
                df.to_csv(output_path)
                print(f"Output saved at index {index}.")

    except Exception as e:
        print(e)
        print("Error occurred. Saving output...")

        df.to_csv(output_path)
        print("Output saved.")

    # Save output data
    df.to_csv(output_path)
    print("Output saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT pipeline.")
    parser.add_argument("--dataset_path", type=str, help="Dataset path")
    args = parser.parse_args()

    # Constants
    dataset_path = args.dataset_path
    dataset_name = str(Path(dataset_path).stem).lower()

    output_path = (
        f"gpt3/final_responses/truthfulqa/gpt3_outputs_{dataset_name}_run1.csv"
    )

    run_pipeline(dataset_path, output_path)
