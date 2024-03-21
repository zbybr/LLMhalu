from openai import OpenAI
import csv

client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key='sk-cV7MZTd97HxBFHYvQ6pSjrKM8o26hFLv9rJbbqYY9XN1xMxU',
)

questions = []
for line in open('MyQA.csv', 'r'):
    questions.append(line.strip())

answers = []
converses = []
converseanswers = []
for i in range(len(questions)):
    prompt = ("For each question you need to give a conclusion in one sentence and don't return YES or No. For "
              "example, Q: Where is the capital of China, A: The capital of China is Beijing.")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": questions[i]},
        ],
    )
    print(response.choices[0].message.content)
    answer = response.choices[0].message.content
    answers.append(answer)

for i in range(len(answers)):
    prompt = ('For each proposition, please return the form of their negative proposition question. For example,'
              'for the proposition: "The Himalaya is the highest mountain peak in the world and K2 is the second '
              'highest peak", your answer should be: "The Himalaya is not the highest mountain peak and K2 is not the '
              'second highest peak, is that a correct truth?" Just return the sentence and dont return anything else.')
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": answers[i]},
        ],
    )
    print(response.choices[0].message.content)
    converse = response.choices[0].message.content
    converses.append(converse)

for i in range(len(converses)):
    prompt = ("For each question you need to give a conclusion in one sentence and please answer YES or NO first, "
              "if you are not sure, answer NS first.")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": converses[i]},
        ],
    )
    print(response.choices[0].message.content)
    converseanswer = response.choices[0].message.content
    converseanswers.append(converseanswer)


rows = zip(questions, answers, converses, converseanswers)
with open('OutputGPT3.5.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
