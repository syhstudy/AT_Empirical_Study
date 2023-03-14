from nlgeval import compute_metrics

# metrics_dict = compute_metrics(hypothesis="./result/BASHEXPLAINER_hyp.csv",
#                                references=["./result/BASHEXPLAINER_ref.csv"], no_skipthoughts=True,
#                                no_glove=True)

# metrics_dict = compute_metrics(hypothesis="./result1/BASHEXPLAINER_hyp.csv",
#                                references=["./result1/BASHEXPLAINER_ref.csv"], no_skipthoughts=True,
#                                no_glove=True)

metrics_dict = compute_metrics(hypothesis="./pretrained_model/first_stage/valid_hyp.csv",
                               references=["./pretrained_model/first_stage/valid_ref.csv"], no_skipthoughts=True,
                               no_glove=True)