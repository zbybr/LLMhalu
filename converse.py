from openai import OpenAI
import csv

client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key='sk-cV7MZTd97HxBFHYvQ6pSjrKM8o26hFLv9rJbbqYY9XN1xMxU',
)

converses = []
with open('Converses.csv', 'r') as cf:
    cr = csv.reader(cf)
    for line in cr:
        converses.append(line[1].strip())

# converses = []
# for i in range(len(preanswers)):
#     prompt = ("For each proposition, please return the form of their negative proposition question. For example, "
#               "for the proposition: The Himalaya is the highest mountain peak in the world and K2 is the second "
#               "highest peak, your answer should be: The Himalaya is not the highest mountain peak and K2 is not the "
#               "second highest peak, is that a correct truth?")
#     response = client.chat.completions.create(
#         model="gpt-4",
#         temperature=0,
#         messages=[
#             {"role": "system", "content": prompt},
#             {"role": "user", "content": preanswers[i]},
#         ],
#     )
#     print(response.choices[0].message.content)
#     converse = response.choices[0].message.content
#     converses.append(converse)

converseanswers = []
for i in range(len(converses)):
    prompt = ('For each question you need to answer YES or NO first, if you are not sure, answer NS first. Then give '
              'conclusion in one sentence. Please notice there is a "not" in sentence, so answer carefully')
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": converses[i]},
        ],
    )
    print(response.choices[0].message.content)
    converseanswer = response.choices[0].message.content
    converseanswers.append(converseanswer)


rows = zip(converses, converseanswers)
with open('outputt.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
