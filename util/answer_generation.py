import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

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


df = pd.read_csv("D:\\Projects\\LLMhalu\\HotpotQA.csv")

for _, row in df.sample(5, random_state=122).iterrows():
    question = row["Questions"]
    answer = row["Answers"]

    qa_pair = f"Question: {question}\nShort Answer: {answer}"
    print(qa_pair)

    response = get_gpt_response(SYSTEM_PROMPT, question=qa_pair, temperature=0.0)
    print("Response:", response)
    print("\n\n")
