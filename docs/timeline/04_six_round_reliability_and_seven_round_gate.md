# 04. Prospective reliability tests and the seven-round solver gate

**Dates:** 4 to 10 September 2026

## What we set out to do

Test the frozen G1 reliability model on new keys, and then check whether the
exact solver could handle seven rounds before we trained anything new for it.

## What we ran

1. **G1 development.** 192 development keys, five folds split by key, so no
   key is in both the training and the test side of a fold.
2. **G1 prospective test.** 64 new keys, run once with the frozen model.
3. **G2.** A richer model with 14 features about the shape of the score
   distribution and its stability. It was developed on 512 keys and then
   tested once on 256 new keys.
4. **Seven-round gate.** 64 new keys. The solver was given the true last round
   key and 10 s. If it could not recover keys even with that help, seven-round
   ranking would be pointless. The pass mark was at least 58 of 64.

## What we found

**G1 (64 keys).** ROC-AUC 0.776 for predicting a top-16 failure, against 0.480
for the old scalar. The paired interval crossed zero. A plain fixed search of
32 candidates recovered 64/64. The G1 routing policy recovered 63/64.

**G2 (256 keys).** ROC-AUC 0.851 against 0.719 for G1, paired p = 0.027. So G2
predicts failure better. But its threshold did not transfer. It was chosen to
give at most 2% error among trusted keys on development data. On new keys the
error was 5.2% (11 of 211 trusted keys failed).

**Seven-round gate: RED.** 0/64 recovered and all 64 timed out, even with the
true last round key supplied.

## What we decided and why

- **Six-round reliability work is closed.** Discrimination improved, but the
  threshold did not hold on new keys, and the fixed search was already as good.
  We did not build a G3 or retune the threshold.
- **The finding becomes part of the paper.** Failure prediction, threshold
  transport and recovery are three separate properties. Improving one did not
  give the others.
- **The gate result stays as recorded.** No setting was changed after the
  outcome was seen.

We then ran checks on the same opened keys to understand the RED result,
without using them to choose anything. With the full true master key fixed,
the seven-round model was satisfiable in 0.0005 s, so the encoding was fine.
The same keys solved at six rounds in 0.4 to 1.3 s. Giving 120 s still failed.
So the wall came from solver search growth between six and seven rounds. This
led to the next stage.

## Questions and answers

**If G2 has AUC 0.85, why do you call it a failure?**
AUC measures ranking of keys by risk. Using G2 needs a fixed threshold. The
threshold set on development data let through more than twice the planned
error rate on new keys, and routing never beat the plain fixed search.

**Why did you stop working on reliability?**
The prospective tests answered the question. Another model on the same design
would be a search for a better number, not a new test.

**Was the seven-round gate set up to fail?**
No. It was fixed before the run, with the true last round key given to the
solver. It measured that exact solver setting. The next stage shows that other
settings and more conditioning change the picture.

**What does "split by key" mean for folds?**
All rows from one master key stay in the same fold. Otherwise the model could
learn a key's quirks in training and be tested on the same key.
