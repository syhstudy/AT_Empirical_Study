from nlgeval import compute_metrics
import pandas as pd

df = pd.read_csv("test_hyp.csv", header=None)
hyp_list = df[0].tolist()

df = pd.read_csv("test_ref.csv", header=None)
ref_list = df[0].tolist()

acc = 0
for i in range(len(hyp_list)):
    hyp = hyp_list[i]
    ref = ref_list[i]
    if hyp == ref:
        acc += 1
print("准确率(acc)：",acc/len(hyp_list))

metrics_dict = compute_metrics(hypothesis="test_hyp.csv",
                               references=["test_ref.csv"], no_skipthoughts=True,
                               no_glove=True)