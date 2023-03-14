import pandas as pd

df = pd.read_csv("dev.csv",)
title_list = df['title'].tolist()
answer_list = df['answer'].tolist()

data_list = []
for i in range(len(title_list)):
    title = title_list[i]
    title = title
    answer = answer_list[i]
    answer = answer.replace('.', ' . ')
    data_list.append([title, answer])

df = pd.DataFrame(data_list, columns=['title', 'answer'])
df.to_csv("dev_token.csv", index=False)