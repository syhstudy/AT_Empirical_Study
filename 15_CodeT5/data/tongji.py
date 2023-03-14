import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby

from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("D:\model\codet5-base")

df = pd.read_csv("Bash_test.csv")
code_list = df['code'].tolist()
new_list = code_list
# for idx in range(len(code_list)):
#     code = code_list[idx]
#     code = code.replace('[', '')
#     code = code.replace(']', '')
#     code = code.replace('\'', '')
#     code = code.replace(', ', ' ')
#     code = code.replace('"', '')
#     new_list.append(code)

print(len(new_list))
# NL_RX : NL 20; REGEX 50; AST 50
ast_len_list = [len(tokenizer.tokenize(ast)) for ast in new_list]
# ast_len_list = [len(str(ast).split()) for ast in new_list]

commutes = pd.Series(ast_len_list)


commutes.plot.hist(grid=True, bins=25, rwidth=0.9,
                   color='#3B64AD')
# plt.title('Commute Times for 1,000 Commuters')
plt.xlabel('Code Sequence Length')
plt.ylabel('Count')
plt.grid(axis='y', alpha=0.75)
# plt.savefig('description_len.png', dpi = 1000)
plt.show()


def getBili(num, demo_list):
    s = 0
    for i in range(len(demo_list)):
        if(demo_list[i] < num):
            s += 1
    print('<'+str(num)+'比例为'+str(s/len(demo_list)))

from numpy import *
code_len_list = ast_len_list
b = mean(code_len_list)
c = median(code_len_list)
counts = np.bincount(code_len_list)
d = np.argmax(counts)
print('平均值'+str(b))
print('众数'+str(d))
print('中位数'+str(c))

getBili(16,code_len_list)
getBili(32,code_len_list)
getBili(48,code_len_list)
getBili(64,code_len_list)