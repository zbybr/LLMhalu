from openai import OpenAI
import csv

client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key='sk-cV7MZTd97HxBFHYvQ6pSjrKM8o26hFLv9rJbbqYY9XN1xMxU',
)

answers = []
corrects = []
with open('OutputGPT3.5.csv', 'r') as cf:
    cr = csv.reader(cf)
    for line in cr:
        answers.append(line[1].strip())
        corrects.append(line[4].strip())
judges = []
for i in range(len(answers)):
    prompt = ("You are provided two paragraphs A and B, there are many sentences in B, all of them are the possible "
              "correct answers, you should decide whether sentence A is corrct or not, answer 'YES' or 'NO' only.")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "A:" + answers[i] + '\n' + 'B:' + corrects[i] + '\n'},
        ],
    )
    print(response.choices[0].message.content)
    judge = response.choices[0].message.content
    judges.append(judge)

rows = zip(judges)
with open('outputt.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
