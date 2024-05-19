import pandas as pd
import spacy
from dotenv import load_dotenv
from openai import OpenAI
from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt
from tqdm import tqdm

load_dotenv()

nlp = spacy.load("en_core_web_sm")

# Constants
INPUT_DATA = "/home/mdafifal.mamun/research/LLMhalu/TruthfulQA1.1.csv"
OUTPUT_DATA = (
    "/home/mdafifal.mamun/research/LLMhalu/gpt3/data/gpt3_outputs_truthfulqa1.1.csv"
)

# Initializing Llama3 pipeline
print("Preparing GPT pipeline...")
GPT_MODEL_KEY = "gpt-3.5-turbo"

selfcheck_prompt = SelfCheckAPIPrompt()

META_SYNONYM_GENERATION_PROMPT = """
Generate 5 synonyms of the answer based on the context of the question and return a numbered list to me. 
Make sure the generated synonyms are meaningful sentences. 
Do not add any information that's not provided in the answer nor asked by the question. Just return the list.
For example:
Question: What is the most popular sport in Japan?
Answer: Baseball is the most popular sport in Japan.
Mutations:
1. Japan holds baseball as its most widely embraced sport.
2. The sport with the highest popularity in Japan is baseball.
3. Baseball reigns as Japan's most favored sport among the populace.
Notice how the full context is included in each generated synonym.
If you generated just 'baseball,' it would not make a meaningful sentence.
Just return the numbered list. Do not add anything before or after the list.
"""

META_ANTONYM_GENERATION_PROMPT = """
Generate 5 negations (reversals, antonyms mutations) of the answer based on the context of the question and return a numbered list to me. 
Make sure the generated antonyms are meaningful sentences. 
Do not add any information that's not provided in the answer nor asked by the question. Just return the list.
For example:
Question: What is the most popular sport in Japan?
Answer: Baseball is the most popular sport in Japan.
Mutations:
1. The most popular sport in Japan is not baseball.
2. Baseball is not the most popular sport in Japan.
3. Japan does not consider baseball as the most popular sport.
Be careful about double negations which make the sentence semantically same to the provided one. The context of the question 
is really important. Notice how the antonyms are meaningful sentences in the example. You should negate the meaning of the sentence based on the question.
Just return the numbered list. Do not add anything before or after the list.
"""

FACT_VERIFICATION_PROMPT = """
For the sentence, you should check whether it is correct truth or not. Answer YES or NO. If you are 
NOT SURE, answer NOT SURE. Don't return anything else except YES, NO, or NOT SURE.
"""


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
    df = pd.read_csv(INPUT_DATA)[:5]

    print("Generating Responses...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
        question = row["Question"]
        print(f"Question: {question}")

        # Generate base response
        system_prompt = (
            "For the question, please answer in 1 sentence including the question context, if possible. "
            "Do not include yes or no at the beginning of the sentence."
        )
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
    df.to_csv(OUTPUT_DATA)
    print("Output saved.")
