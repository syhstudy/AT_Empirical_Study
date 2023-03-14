# AT_Empirical_Study
An Empirical Study of Adversarial Training in Code Comment Generation

Practical Guidelines
==========================================
normPDG first performs $L_1$ normalization on the vectors and then applies $L_2$ normalization to the generated vectors.
normPDG has the advantage of  using $L_1$ normalization to reduce the effect of large values on the vectors, and then applying $L_2$ normalization to ensure that the resulting vectors have a consistent length and sum to 1. 
Therefore, normPDG can improve the stability of the normalization process while retaining the advantages of $L_1$ and $L_2$ normalization.

The experimental results in this paper can verify the feasibility of normPDG:

<img src="https://user-images.githubusercontent.com/93321396/224938336-42ca251f-11d9-4495-80ac-8ab470c7cd3a.png" width = "700" />

Specifically, compared with the PDG, normPDG can improve the performance by 5.01\%, 5.67\%, 2.76\%, and 2.99\% for BLEU-3, BLEU-4, METEOR, and ROUGE-L respectively.
