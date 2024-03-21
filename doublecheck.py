from openai import OpenAI
import csv

client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key='sk-cV7MZTd97HxBFHYvQ6pSjrKM8o26hFLv9rJbbqYY9XN1xMxU',
)

total = 0
check = 0
with open('outputt.csv', 'r') as cf:
    cr = csv.reader(cf)
    for line in cr:
        if 'YES' in line[6]:
            total += 1

print("hallucination rate: ")
print(str(total) + " / " + "100")
print(str(100.0 * total / 100.0) + "\n")

total = 0
check = 0
with open('outputt.csv', 'r') as cf:
    cr = csv.reader(cf)
    for line in cr:
        if 'YES' in line[7]:
            total += 1
            if 'YES' in line[6]:
                check += 1

print("precision: ")
print(str(check) + " / " + str(total))
print(str(100.0 * check / total) + "\n")

total = 0
check = 0
with open('outputt.csv', 'r') as cf:
    cr = csv.reader(cf)
    for line in cr:
        if 'YES' in line[6]:
            total += 1
            if 'YES' in line[7]:
                check += 1

print("recall: ")
print(str(check) + " / " + str(total))
print(str(100.0 * check / total) + "\n")

