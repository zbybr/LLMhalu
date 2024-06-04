import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import csv

from tqdm import tqdm

load_dotenv()

client = OpenAI(
    # base_url='https://api.openai-proxy.org/v1',
    # api_key='sk-cV7MZTd97HxBFHYvQ6pSjrKM8o26hFLv9rJbbqYY9XN1xMxU',
)

dataset_path = "selfcheck_output.csv"
df = pd.read_csv(dataset_path, encoding='latin-1')

questions = []
answers = []
score = []
synonyms = []
antonyms = []
synonym_responses = []
antonym_responses = []

for _, row in df.iterrows():
    question = row["Questions"]
    answer = row["Answers"]

    questions.append(question)
    answers.append(answer)


for question, answer in tqdm(zip(questions, answers), total=len(questions), desc="Processing question"):
    print(f"Base Question: {question}")
    print(f"Base Answer: {answer}")

    # Synonyms
    prompt = """Please generate 5 synonyms of the answer based on the context of the question and return a numbered list to me. 
        The context of the question is really important. So, be careful.
        Do not add any information that's not in the provided in the answer nor asked by the question. Just return the list.
        """

    qa_pair = (f"Question: {question}"
               f"Answer: {answer}")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": qa_pair},
        ],
    )
    res = response.choices[0].message.content

    lst = res.splitlines()
    synonyms.append(";".join(lst))

    sum = 0.0
    synonym_llm_responses = []
    for ss in lst:
        prompt = ("For the sentence, you should check whether it is correct truth or not, answer YES or NO, if you are "
                  "NOT SURE, answer NOT SURE. Don't return anything else except YES, NO or NOT SURE.")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k-0613",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": ss},
            ],
        )
        synonym_llm_response = response.choices[0].message.content
        synonym_llm_responses.append(synonym_llm_response)
        print(f"Mutated Synonym: {ss}\nSynonym Response: {synonym_llm_response}")

    synonym_responses.append(";".join(synonym_llm_responses))

    # Antonyms
    prompt = """Please generate 5 negations(reversals, antonyms mutations) of the answer based on the context of the question 
    and return a list to  me. 
    Be careful about double negations which make the sentence semantically same to the provided one. The context of the question
    is really important. So, be careful. Also, you should negate the meaning of the sentence based on the question. 
    Do not add any information that's not in the provided in the answer nor asked by the question.
    """

    qa_pair = (f"Question: {question}"
               f"Answer: {answer}")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k-0613",
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": qa_pair},
        ],
    )
    res = response.choices[0].message.content

    lst = res.splitlines()
    antonyms.append(";".join(lst))

    antonym_llm_responses = []
    for ss in lst:
        prompt = ("For the sentence, you should check whether it is correct truth or not, answer YES or NO, if you are "
                  "NOT SURE, answer NOT SURE. Don't return anything else except YES, NO or NOT SURE.")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k-0613",
            temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": ss},
            ],
        )
        antonym_llm_response = response.choices[0].message.content
        antonym_llm_responses.append(antonym_llm_response)
        print(f"Mutated Antonym: {ss}\nAntonym Response: {antonym_llm_response}")

    antonym_responses.append(";".join(antonym_llm_responses))

output_df = pd.DataFrame({
    "question": questions,
    "answer": answers,
    "synonyms": synonyms,
    "synonym_responses": synonym_responses,
    "antonyms": antonyms,
    "antonym_responses": antonym_responses
})

output_df.to_csv("output_mutations.csv")
