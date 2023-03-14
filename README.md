# AT_Empirical_Study
An Empirical Study of Adversarial Training in Code Comment Generation

normPDG first performs $L_1$ normalization on the vectors and then applies $L_2$ normalization to the generated vectors.
normPDG has the advantage of  using $L_1$ normalization to reduce the effect of large values on the vectors, and then applying $L_2$ normalization to ensure that the resulting vectors have a consistent length and sum to 1. 
Therefore, normPDG can improve the stability of the normalization process while retaining the advantages of $L_1$ and $L_2$ normalization.
