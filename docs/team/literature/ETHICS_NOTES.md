# My ethics notes on our attack model

Notes updated: 27 October 2026 (Perth).

I have focused on the neural attack model and how we ran and reported the research. These notes follow Sections 4.2, 4.6 and 6.2-6.5 of our report draft of 26 October.

## How the work could be used

Cryptanalysis can help us understand weaknesses, but the same ideas can also support attacks. Our experiments use reduced-round SPECK32/64 and SIMON32/64 with generated keys and synthetic chosen-plaintext data. No deployed service, production key or third-party account is part of this evaluation.

I would keep that limit clear in the report and presentation. These results do not show a break of full-round SPECK or SIMON. Any use outside this research setting would need its own permission and scope.

## A score is not proof of a key

The neural model ranks guesses. It does not prove that its highest-scoring guess is correct. A SAT assignment also depends on whether we encoded the problem properly. Our acceptance rule checks the returned master key against the assigned key and two fresh blocks that were kept out of ranking and solving. The later native-solver studies also check all 128 known blocks.

For me, this is the main reliability point. Containment, timeout, UNSAT and verified recovery must stay separate. Running out of time does not prove that no key exists. We remain responsible for the encoding, the checks and how we explain the outcome.

## Fair comparisons and honest reporting

Fairness in this project means giving methods a comparable test. Paired methods need the stated keys, data, time limit and solver conditions. Each method has its own solver state, and every assigned key stays in the denominator, including failures.

We also need to separate online work from training and table construction. Some campaigns use different clock rules, so I would explain those differences instead of pooling their results. Published weights and methods need credit. The negative B1 and confidence-gate results matter too, along with the reasons for compromised runs. I used the report and [project timeline](../../timeline/README.md) for this review; I have not independently rerun those campaigns.

## Data and leakage

Our experiment keys, plaintexts and ciphertexts were generated for the study. They are not customer or personal records. That reduces privacy concerns, but synthetic data can still leak between training, development and evaluation if we handle the splits poorly.

Ground-truth positions are for analysis, not inputs given to the ranker. Fresh verification blocks must also stay out of ranking and solving. I would keep these checks separate from claims about repository access controls or production security, which we did not test. The details are in Section 6.4 of the report and [DATASETS.md](../../DATASETS.md).

## Resource use

We measured time, not energy or emissions. A 300-second online limit does not establish a smaller carbon footprint, since training, table construction, memory and hardware also have costs. The 4 GiB table and the network weights have different resource needs; online speed alone does not compare their full impact.

Reusing suitable assets and testing a small run first could reduce repeated work. I would describe those as practical suggestions, not measured environmental savings. Broader economic and deployment benefits were not established either.

Page 2 of `final_report_structure.pdf` asks us to discuss ethics and AI principles, data security and sustainability. I have addressed those topics here through the model's reliability, fair evaluation, data handling and resource limits.
