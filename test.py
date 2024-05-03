from openai import OpenAI
import csv

client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key='sk-cV7MZTd97HxBFHYvQ6pSjrKM8o26hFLv9rJbbqYY9XN1xMxU',
)

questions = []
answers = []
score = []
with open('comparison.csv', 'r', encoding='utf-8-sig') as cf:
    cr = csv.reader(cf)
    for line in cr:
        questions.append(line[1].strip())
        answers.append(line[6])

for i in range(len(questions)):
    if i == 0:
        continue
    print(questions[i])
    print(answers[i])

    prompt = "Please generate 5 synonym mutations of the sentence, and return a python list to me."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k-0613",
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": answers[i]},
        ],
    )
    res = response.choices[0].message.content
    print(res)

    lst = res.splitlines()
    sum = 0.0
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
        res = response.choices[0].message.content
        print(res)

    prompt = ("Please generate 5 negations(reversals, antonyms mutations) of the sentence, and return a python list to "
              "me.")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k-0613",
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": answers[i]},
        ],
    )
    res = response.choices[0].message.content

    lst = res.splitlines()

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
        res = response.choices[0].message.content
        print(res)
