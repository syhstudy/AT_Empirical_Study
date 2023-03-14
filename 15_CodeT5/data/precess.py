import pandas as pd

df = pd.read_csv("BIKER_test.csv",)
title_list = df['title'].tolist()
answer_list = df['answer'].tolist()

data_list = []
for i in range(len(title_list)):
    title = title_list[i]
    answer = answer_list[i]
    answer = answer.replace('[', '')
    answer = answer.replace(']', '')
    answer = answer.replace('\'', '')
    answer = answer.replace(', ', ' ')
    answer = answer.replace('"', '')
    if (len(answer.split()) == 1):
        data_list.append([title, answer])

df = pd.DataFrame(data_list, columns=['title', 'answer'])
df.to_csv("test.csv", index=False)