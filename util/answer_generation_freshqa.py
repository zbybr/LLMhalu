import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

GPT_MODEL_KEY = "gpt-3.5-turbo-0613"

SYSTEM_PROMPT = """Given a question and a very short answer, 
your task is to extend the answer to a complete sentence with appropriate context from the question.
DO NOT generate more than one sentence and no additional details that are not provided in the short answer."""


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


df = pd.read_csv("D:\\Projects\\LLMhalu\\datasets\\freshqa.csv")
# df = df.sample(3, random_state=42)

for index, row in tqdm(df.iterrows(), total=len(df)):
    tqdm.write(f"Processing sample {index}/{len(df)}")
    question = row["question"]

    for i in range(0, 10):
        answer = row[f"answer_{i}"]
        if answer is not np.nan:
            qa_pair = f"Question: {question}\nShort Answer: {answer}"
            print(qa_pair)

            response = get_gpt_response(SYSTEM_PROMPT, question=qa_pair, temperature=0.0)
            df.loc[index, f"extended_answer_{i}"] = response

            print("Response:", response)


df.to_csv("D:\\Projects\\LLMhalu\\datasets\\freshqa_extended.csv")