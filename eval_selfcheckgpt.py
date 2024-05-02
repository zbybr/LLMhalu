import pandas as pd
import spacy
import torch
import csv
from openai import OpenAI
from dotenv import load_dotenv
from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt

load_dotenv()

nlp = spacy.load("en_core_web_sm")

client = OpenAI()

# prompts = ["When responding, please formulate your answer in a single sentence while encompassing the context of the question.", "Kindly provide your response within a single sentence, ensuring it encapsulates the context of the question without starting with 'yes' or 'no'.", "Could you craft your answer in one sentence, incorporating the context of the question, without starting with 'yes' or 'no'?", "I'd appreciate it if your answer could be expressed in a single sentence, including the context of the question but avoiding starting with 'yes' or 'no'.", "It would be helpful if your response could be contained in a single sentence, covering the context of the question while refraining from starting with 'yes' or 'no'."]

questions = []
answers = []
best_answers = []
correct_answers = []
wrong_answers = []
generated_samples = []
scores = []

df = pd.read_csv("TruthfulQA_100-samples_42-seed.csv")

for _, row in df.iterrows():
    question = row["Question"]
    best_answer = row["Best Answer"]
    correct_answer = row["Correct Answers"]
    wrong_answer = row["Incorrect Answers"]

    questions.append(question)
    best_answers.append(best_answer)
    correct_answers.append(correct_answer)
    wrong_answers.append(wrong_answer)

    prompt = ("For the question, please answer in 1 sentence including the question context, if it is possible, "
              "do not include yes or no at the first of the sentence.")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )
    answer = response.choices[0].message.content

    answers.append(answer)

    passage = answer
    sentences = [sent.text.strip() for sent in nlp(passage).sents]

    samples = []

    for i in range(5):
        prompt = ("For the question, please answer within 1 sentence including the question context, if it is "
                  "possible, do not include yes or no at the first of the sentence.")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
        )
        sample = response.choices[0].message.content
        samples.append(sample)

    generated_samples.append(
        ";".join(samples)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selfcheck_prompt = SelfCheckAPIPrompt(client_type="openai", model="gpt-3.5-turbo")

    sent_scores_prompt = selfcheck_prompt.predict(
        sentences=sentences,
        sampled_passages=samples,
        verbose=True,
    )

    scores.append(sent_scores_prompt)

data = {
    'Questions': questions,
    'Answers': answers,
    'Best Answers': best_answers,
    'Correct Answers': correct_answers,
    'Incorrect Answers': wrong_answers,
    'Generated Samples': generated_samples,
    'Selfcheck Scores': scores
}

# Create DataFrame
df = pd.DataFrame(data)
df.to_csv("selfcheck_output.csv")

