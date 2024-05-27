from pprint import pprint

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from llm_prompts.prompts import *

load_dotenv()


def generate_mutations(question: str, num_variation: int):
    llm = ChatOpenAI()
    prompt_template = PromptTemplate.from_template(MUTATED_QUESTION_GENERATION_PROMPT)
    meta_question_generation_chain = prompt_template | llm | StrOutputParser()
    meta_question_response = meta_question_generation_chain.invoke(
        {
            "num_variation": num_variation,
            "question": question
        }
    )

    meta_verification_questions = meta_question_response.split("\n")

    return meta_verification_questions


if __name__ == "__main__":
    df = pd.read_csv("TruthfulQA_100-samples_42-seed.csv")

    questions = []
    mutations = []

    for _, row in df.iterrows():
        question = row["Question"]
        print("Question:", question)
        num_variation = 5

        meta_questions = generate_mutations(question=question, num_variation=num_variation)
        print("Mutations")
        pprint(meta_questions)
        print()

        questions.append(question)
        mutations.append(";".join(meta_questions))

    output_df = pd.DataFrame({
        "questions": questions,
        "mutations": mutations
    })

    output_df.to_csv("question_mutations.csv")
