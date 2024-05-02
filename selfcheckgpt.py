import spacy
import torch
import csv
from openai import OpenAI
from dotenv import load_dotenv
from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt

load_dotenv()

nlp = spacy.load("en_core_web_sm")

client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key='sk-cV7MZTd97HxBFHYvQ6pSjrKM8o26hFLv9rJbbqYY9XN1xMxU',
)

# prompts = ["When responding, please formulate your answer in a single sentence while encompassing the context of the question.", "Kindly provide your response within a single sentence, ensuring it encapsulates the context of the question without starting with 'yes' or 'no'.", "Could you craft your answer in one sentence, incorporating the context of the question, without starting with 'yes' or 'no'?", "I'd appreciate it if your answer could be expressed in a single sentence, including the context of the question but avoiding starting with 'yes' or 'no'.", "It would be helpful if your response could be contained in a single sentence, covering the context of the question while refraining from starting with 'yes' or 'no'."]

questions = []
answers = []
bestanswers = []
correctanswers = []
wronganswers = []
scores = []
with open('test.csv', 'r', encoding='utf-8-sig') as cf:
    cr = csv.reader(cf)
    for line in cr:
        questions.append(line[0].strip())
        bestanswers.append(line[1].strip())
        correctanswers.append(line[2].strip())
        wronganswers.append(line[3].strip())

for question in questions:
    print(question)
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
    print(answer)

    passage = answer
    sentences = [sent.text.strip() for sent in nlp(passage).sents]

    samples = []

    for i in range(5):
        prompt = ("For the question, please answer within 3 sentences including the question context, if it is "
                  "possible, do not include yes or no at the first of the sentence.")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
        )
        sample = response.choices[0].message.content
        print(sample)
        samples.append(sample)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selfcheck_prompt = SelfCheckAPIPrompt(client_type="openai", model="gpt-3.5-turbo")

    sent_scores_prompt = selfcheck_prompt.predict(
        sentences=sentences,
        sampled_passages=samples,
        verbose=True,
    )

    print(sent_scores_prompt)

    scores.append(sent_scores_prompt)

rows = zip(questions, answers, scores)
with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
