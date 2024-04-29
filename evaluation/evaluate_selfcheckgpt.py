from typing import List, Any

import numpy as np
import pandas as pd
import spacy
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt
from spacy import Language
from tqdm import tqdm

load_dotenv()
SEED = 42
np.random.seed(SEED)


def get_base_response(question: str, model: str = "gpt-3.5-turbo"):
    llm = ChatOpenAI(model=model)
    template = PromptTemplate.from_template("Answer the question in one sentence. No addition details.\n{question}")

    chain = template | llm | StrOutputParser()
    response = chain.invoke({"question": question})

    return response


def read_csv(file_path: str):
    return pd.read_csv(file_path)


def generate_score(selfcheckgpt: Any, base_sentences: List[str], correct_answer_samples: List[str]) -> List[float]:
    scores = selfcheckgpt.predict(
        sentences=base_sentences,
        sampled_passages=correct_answer_samples,
        verbose=True,
    )

    return scores


def process_dataset(df: pd.DataFrame, nlp: Language, selfcheckgpt: Any, num_samples: int = 1) -> pd.DataFrame:
    sampled_df = df.sample(n=num_samples)

    questions = sampled_df["Question"].tolist()
    correct_answers = sampled_df["Correct Answers"].tolist()
    best_answers = sampled_df["Best Answer"].tolist()
    base_responses = []
    scores = []

    for question, correct_answer in tqdm(zip(questions, correct_answers), total=len(questions), desc="Processing row"):
        base_response = get_base_response(question)

        base_sentences = [sent.text.strip() for sent in nlp(base_response).sents]
        correct_samples = correct_answer.split(";")
        hallu_score = generate_score(selfcheckgpt, base_sentences, correct_samples)

        base_responses.append(base_response)
        scores.append(hallu_score)

    return pd.DataFrame({
        'Question': questions,
        'Correct Answers': correct_answers,
        'Best Answer': best_answers,
        'LLM Response': base_responses,
        'Hallu Score': scores
    })


model="gpt-3.5-turbo"
dataset_path = "D:\\Projects\\LLMhalu\\TruthfulQA.csv"
output_path = "TruthfulQA-selfcheckgpt.csv"
df = read_csv(dataset_path)
nlp = spacy.load("en_core_web_sm")
selfcheckgpt = SelfCheckAPIPrompt(client_type="openai", model=model)

processed_df = process_dataset(df, nlp, selfcheckgpt, num_samples=100)
processed_df.to_csv(output_path)
