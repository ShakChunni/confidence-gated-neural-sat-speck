# My glossary of project terms

Notes updated: 27 October 2026 (Perth).

I put this glossary together to explain the terms we use most often in the report. The locations refer to our report draft of 26 October.

For the first use, I looked at the main text and figures, leaving out the title, abstract and contents pages. Sometimes the report uses a related phrase before the exact term; I have shown both where that matters. Main-text PDF page numbers are five higher than the printed page numbers.

| Term | Plain-language meaning | First report section / location |
|---|---|---|
| SPECK32/64 | A version of SPECK with 32-bit blocks and a 64-bit master key. We test versions with fewer rounds than the full cipher. | Section 1.1, p. 1 |
| SIMON32/64 | A version of SIMON with 32-bit blocks and a 64-bit master key. Its round function differs from SPECK, so similar remaining depth does not mean similar solving time. | Section 1.1, p. 1 |
| Round key | A key word used in one round of encryption. The key schedule derives these from the master key; the words we rank are 16 bits long. | Section 1.1, p. 1 |
| Input difference | The XOR difference we choose between two plaintexts in a pair. This is how we construct related pairs for the distinguisher. | Section 4.2, p. 4 |
| DDT | A difference distribution table gives probabilities for output differences under a chosen input difference and model. Our sampled SPECK tables and model-based SIMON distributions should be described separately. | “Difference tables”: Section 1.1, p. 1; expanded DDT abbreviation: Section 2.2, p. 3 |
| Neural distinguisher | A trained network that scores whether data looks like a selected cipher-pair distribution. The score can guide our ranking, but it is not proof of a key. | Section 1.1, p. 1 |
| Ranking | Putting candidate keys or key words in score order. A better order may reach the useful guess earlier, but producing that order also takes time. | Section 1.1, p. 1 |
| Containment | The true candidate chain is somewhere in the queue we kept. We may still run out of time before reaching it. | Section 1.1, p. 1 |
| Queue | The ordered list of candidate key-word chains sent for completion. Our pipeline uses summed stage ranks and stated rules for ties. | Section 1.1, p. 1, including Figure 1 |
| SAT solver | A program that finds a Boolean assignment satisfying all the encoded constraints, or establishes that none exists. We still check a returned key against our acceptance rule. | Section 1.1, p. 1 (“exact solver”); SAT labelled in Figure 1 |
| CryptoMiniSat | The SAT solver family used in our main completion experiments. Its XOR handling and settings can change the cost of solving. | Section 4.2, p. 4 |
| Gaussian elimination | A way to solve systems of linear XOR equations. Some solver settings use it, but turning it on does not always save time. | Section 4.2, p. 4 (Gauss-Jordan); Gaussian-elimination wording follows in Section 4.4 |
| UNSAT | The tested formula and assumptions have no satisfying assignment. This is different from a timeout and does not rule out every other candidate chain. | Section 1.1, Figure 1, p. 1 |
| Fresh verification blocks | Extra plaintext/ciphertext observations that the ranker and solver did not see. We use them to check the proposed master key, alongside equality with the assigned key. | Section 1.1, p. 1; precise acceptance rule in Section 4.2 |
| PAR-2 | Average time with a penalty for failures: each failure counts as twice the time limit. With our 300-second limit, a failure contributes 600 seconds. | Section 4.2, pp. 4-5; Equation (1) |
| Preparation time | Work before the first completion attempt, including partial decryption, scoring, queue work and formula construction. Loading is included where the campaign rules charge it. | Section 1.1, Figure 1, p. 1; operational definition in Section 5.3.4 |
| Rejection cost | Time used to rule out a wrong candidate. Our cost model also accounts for gaps between attempts; slow rejection can keep the solver from reaching the right guess. | Section 1.1, p. 1 and Figure 1; named component in Section 5.3.4 |
| Affordable depth k* | An estimate of how many wrong candidates we can pass before finishing the correct one would exceed the budget. Equation (3) uses `(300 - P - s) / r` and compares it with the true zero-based position. | “Affordable search depth”: Section 1.3, p. 2; symbol and formula: Section 5.3.5, p. 12 |
| Exhaustive search | Trying every assignment of the remaining unknown schedule bits and checking the resulting key. After some words are fixed, this can be much smaller than all 2^64 master keys. | Section 1.3, p. 2; implementation in Section 5.3.8 |
| Prediction interval | A model-based range for a future result, such as a recovery count on fresh keys. It is not a guarantee or the same as a confidence interval for a population parameter. | Section 4.4, p. 6 (90% count intervals); explicit “prediction intervals” in Section 5.3.6 |

I based the definitions on our threat model, acceptance rule and Equations (1)-(3). My [background notes](BACKGROUND_NOTES.md) explain the literature behind them. The [wording check](BACKGROUND_CHECK.md) lists places where I think the report needs a clearer limit or explanation.
