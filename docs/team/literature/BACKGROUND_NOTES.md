# My background notes

Notes updated: 27 October 2026 (Perth).

I wrote these notes to explain the ideas behind our project and check how we describe earlier work. I used Sections 2 and 4-6 of our report draft of 26 October and the sources linked below. Reference numbers match [my audit](REFERENCE_AUDIT.md).

## Neural differential cryptanalysis

A neural differential distinguisher learns to tell selected ciphertext-pair data apart from comparison data. For key ranking, we partly decrypt the observations with a guessed trailing round key and use the network's scores to order the guesses. Gohr shows how this can improve key ranks and support reduced-round SPECK recovery. It does not establish a break of the full cipher. [Gohr 2019, reference 13](https://link.springer.com/chapter/10.1007/978-3-030-26951-7_6).

The network is only part of the attack. Bao et al. improve the classical neutral-bit part of neural-aided SPECK attacks and compare neural and differential distinguishers on SIMON. I would not attribute the whole gain to the network architecture. [Bao et al. 2022, reference 4](https://link.springer.com/chapter/10.1007/978-3-031-22963-3_11).

Zhang, Wang, and Wang use Inception-inspired convolutions, larger receptive fields and knowledge distillation. Their work also makes an important distinction for our report: the number of rounds a distinguisher handles is different from the number reached by a complete attack. [Zhang et al. 2024, reference 24](https://cic.iacr.org/p/1/3/13).

The 2026 survey identifies comparison and scaling as open problems. For our work, that means stating the data, hardware, time limit and success rule clearly. It does not mean that earlier papers ignored runtime. I see our question as whether a useful ranking leaves enough time to recover and verify the key. [Gérault et al. 2026, reference 12](https://cic.iacr.org/p/3/2/21).

## Classical difference tables

Differential cryptanalysis studies how input differences pass through cipher rounds. The resulting patterns can help rank key guesses. Classical SIMON and SPECK attacks existed before neural ranking: Abed et al. studied reduced-round attacks, while Biryukov, Roy, and Velichkov improved trails and differentials through automated search. These papers give our background; they do not directly measure the tables in our repository. [Abed et al. 2015, reference 1](https://link.springer.com/chapter/10.1007/978-3-662-46706-0_27); [Biryukov et al. 2015, reference 8](https://link.springer.com/chapter/10.1007/978-3-662-46706-0_28).

I think the classical comparison needs care. Benamira et al. study what the SPECK network learns and build non-neural alternatives. A difference-only table may therefore miss information that a stronger classical method could use. [Benamira et al. 2021, reference 7](https://link.springer.com/chapter/10.1007/978-3-030-77870-5_28).

Meet-in-the-filter and dynamic counting improve classical recovery using differential structure and time-data trade-offs. Its 11-round result uses its own assumptions. We cannot turn its complexity figures directly into seconds on our 300-second experiment. [Biryukov et al. 2023, reference 9](https://link.springer.com/chapter/10.1007/978-3-031-33488-7_6).

Our table descriptions also need to stay precise. A sparse sampled SPECK table, a sampled table storing all 2^32 entries, and a SIMON distribution calculated from a model are different things. Storing every entry does not remove sampling error. Evaluating a Markov model exactly does not make it the exact distribution for a fixed key. I have followed Sections 4.3 and 4.5 of our report and [timeline 06](../../timeline/06_simon.md) here.

## SAT-based key recovery

SAT cryptanalysis turns encryption observations into Boolean constraints. A satisfying assignment gives a key that fits those constraints. We still need to check that it passes our full acceptance rule. Massacci and Marraro explain the approach through DES; their work is a foundation, rather than a promise of the same speed on SPECK. [Massacci and Marraro 2000, reference 17](https://link.springer.com/article/10.1023/A:1006326723002).

How we write the formula matters. Soos, Nohl, and Castelluccia adapt both the solver and the representation, including native XOR relations. This can help, but our results depend on the solver settings. I would not say Gaussian elimination is always faster. [Soos et al. 2009, reference 21](https://link.springer.com/chapter/10.1007/978-3-642-02777-2_24).

Andrzejczak and Dudzic study SIMON and SPECK with and without key-expansion equations. Without the schedule, the solver can find unrelated round keys. That is not automatically the real master key. This is why retaining the schedule and checking the returned key matters in our work. [Andrzejczak and Dudzic 2019, reference 2, Section III.A](https://www.infocommunications.hu/documents/169298/4647244/InfocomJ_2019_4_1_Andrzejczak.pdf).

SAT4CryptoBench gives a common set of cryptographic SAT tasks and compares different uses of learning. Classification, solver guidance and BASIN's bitstring-based predictions produce different kinds of answers. Their correctness requirements also differ. I use this as motivation for careful comparisons, not as proof that our pipeline gains a speed advantage. [Zheng et al. 2025, reference 26](https://papers.neurips.cc/paper_files/paper/2025/hash/69d97a6493fbf016fff0a751f253ad18-Abstract-Datasets_and_Benchmarks_Track.html).

## Why a shortlist is not enough

The distinction I find most useful is between having the right candidate in the queue and finishing its recovery. Our report separates containment, dispatch, completion and verification. A slow wrong candidate can use the remaining time before the solver reaches the right one. Ranking itself may also take too long. A timeout is not UNSAT, and a high score is not a verified key. These points are set out in Sections 4.2 and 5.3.4-5.3.5.

For a contained candidate at zero-based position `k`, the simple cost estimate is `P + k*r + s`. Here `P` is preparation, `r` is the typical cost of rejecting and moving past a wrong candidate, and `s` is correct-candidate completion. As an example, `P=60`, `r=2`, `s=2` and `k=150` give 362 seconds. That is over the 300-second limit even though the right candidate is present. The estimated affordable depth is `(300-60-2)/2 = 119`. I use these numbers only to explain the rule; they are not another experiment.

Our contribution is to compare the complete procedures under stated conditions and test a cost-based choice on fresh keys. The true position is known during evaluation, not to an attacker choosing a method for a new target. I would keep explanations using observed positions separate from forecasts made before the keys were drawn. Different hardware, formulas or queue rules would need new cost measurements. [Timeline 09](../../timeline/09_queue_cost_rule.md); [timeline 11](../../timeline/11_brute_force_completer.md); report Sections 5.3.6, 5.3.8 and 6.2.
