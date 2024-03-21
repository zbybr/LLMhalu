from openai import OpenAI
import csv

client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key='sk-cV7MZTd97HxBFHYvQ6pSjrKM8o26hFLv9rJbbqYY9XN1xMxU',
)

questions = []
for line in open('MyQA.csv', 'r'):
    questions.append(line.strip())
corrects = []
with open('answer.csv', 'r') as cf:
    cr = csv.reader(cf)
    for line in cr:
        corrects.append(line[0] + '; ' + line[1])

answers = []
dcanswers = []
janswers = []
for i in range(len(questions)):
    prompt = "For each question you need to give a conclusion in one sentence."
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": questions[i]},
        ],
    )
    print(response.choices[0].message.content)
    answer = response.choices[0].message.content
    answers.append(answer)

for i in range(len(questions)):
    nquestion = answers[i] + ",is that a correct truth?"

    dcprompt = ("For this question, you have to answer 'YES' or 'NO' at the beginning of your answer. If you cannot "
                "answer the question, answer 'NS'(Not Sure) at the beginning of your answer")
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {"role": "system", "content": dcprompt},
            {"role": "user", "content": nquestion},
        ],
    )
    print(response.choices[0].message.content)
    dcanswer = response.choices[0].message.content
    dcanswers.append(dcanswer)

for i in range(len(questions)):
    jprompt = ("You are provided two paragraphs A and B, you have to decide whether the two paragraphs are talking "
               "about the same meaning, answer 'YES' or 'NO' only.")

    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {"role": "system", "content": jprompt},
            {"role": "user", "content": "A:" + answers[i] + '\n' + 'B:' + corrects[i] + '\n'},
        ],
    )
    print(response.choices[0].message.content)
    janswer = response.choices[0].message.content
    janswers.append(janswer)

rows = zip(questions, answers, dcanswers, corrects, janswers)
with open('outputt.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
