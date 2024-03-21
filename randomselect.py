import random
import csv

range_start = 1
range_end = 817

num_samples = 200

selected_numbers = random.sample(range(range_start, range_end + 1), num_samples)

print(selected_numbers)

questions = []
answers = []
converses = []
conanswers = []
corrects = []
with open('OutputGPT3.5.csv', 'r') as cf:
    cr = csv.reader(cf)
    i = 0
    for line in cr:
        if i in selected_numbers:
            questions.append(line[0])
            answers.append(line[1])
            converses.append(line[2])
            conanswers.append(line[3])
            corrects.append(line[4])
        i += 1

rows = zip(questions, answers, converses, conanswers, corrects)
with open('outputt.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
