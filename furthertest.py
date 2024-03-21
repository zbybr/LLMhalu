from openai import OpenAI
import csv
import random

client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key='sk-cV7MZTd97HxBFHYvQ6pSjrKM8o26hFLv9rJbbqYY9XN1xMxU',
)

range_start = 1
range_end = 817

num_samples = 20

selected_numbers = random.sample(range(range_start, range_end + 1), num_samples)


questions = []
i = 0
for line in open('MyQA.csv', 'r'):
    if i in selected_numbers:
        questions.append(line.strip())
    i += 1

answers = []
pos_mutations = []
neg_mutations = []
for i in range(len(questions)):
        prompt = "For each question, you need to give me 5 synonymous mutations of it."
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": questions[i]},
            ],
        )
        print(response.choices[0].message.content)
        pos_mutations.append(response.choices[0].message.content)
        prompt = "For each question, you need to give me 5 negative mutations of it."
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": questions[i]},
            ],
        )
        print(response.choices[0].message.content)
        neg_mutations.append(response.choices[0].message.content)

rows = zip(questions, pos_mutations, neg_mutations)
with open('Mutations.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
