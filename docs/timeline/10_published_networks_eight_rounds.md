# 10. Published networks at seven and eight rounds

**Dates:** 10 to 11 October 2026

## What we set out to do

Our own network lost to a classical table at seven rounds. The fair next
question was whether a well-known published network, Gohr's, does better, and
whether ranking pays at eight rounds, where classical tables are weaker.

## Probe first (demo keys)

While probing we found a defect in our batched GPU (MPS) path for the new
networks. Large batches gave wrong scores. Our production B1 used a different
batch size and matched the CPU to within 5.3e-6, so earlier results were not
affected. After the fix:

- Gohr's seven-round network put the true last key in its top 32 for 112 of
  128 demo keys at eight rounds;
- no sparse seven-round table we tried did so for any of 32 keys;
- Gohr's six-round network had the true key in its top 8 for 32/32 keys at
  seven rounds.

## First campaign (32 fresh keys)

Protocol, predictions and pass rules were committed before the keys existed.

| Arm | Stage 1 | Recovered of 32 |
|---|---|---:|
| Seven rounds | Gohr six-round network | 32 |
| Seven rounds | sparse six-round table | 30 |
| Seven rounds | our B1 | 19 |
| Eight rounds | Gohr seven-round network | 26 |
| Eight rounds | sparse seven-round table | 0 |

All five arms landed inside their predicted ranges. At eight rounds the network
won 26 to 0. After the run we checked why the table got zero. On every key, none
of the 64 decrypted pairs hit a stored entry, so every candidate scored zero.
That table had no signal at eight rounds, and the win was against a ranker with
nothing to offer.

## Second campaign: a complete table

The eight-round attack fixes three round keys. Stage 1 keeps 32 candidates,
then classical six- and five-round tables keep 8 and 2 below each, so the
queue has 512 entries. To make the test fair, we built a full 2^32-entry
seven-round table for Stage 1 on a Google Colab A100 GPU: 2^42 samples, 811 s,
stored as 4 GiB. On demo keys it put
the true key in the top 32 for 22 of 32 keys, against 28 for Gohr's network.

On 32 new fresh keys:

| Arm | Recovered of 32 |
|---|---:|
| Gohr seven-round network | 24 |
| Complete seven-round table | 18 |

The network recovered 6 keys the table missed, and the table none the network
missed (McNemar p = 0.031, Holm-adjusted 0.063). Across the four eight-round arms
of the two campaigns, on separate keys, every true candidate that reached the
queue was recovered (26, 0, 24 and 18, so 68 of 68), and none
outside the queue were.

## What we decided and why

- The seven-round "classical beats neural" finding applies to our B1, not to
  neural ranking in general. Gohr's network matched the table at seven rounds.
- At eight rounds, rejecting a wrong candidate is cheap, so containment decides
  recovery. The network's edge came from putting the true key in the queue more
  often.
- With 32 keys the eight-round gap was not significant after correction. That
  was the reason for a larger replication later (stage 12).

## Questions and answers

**Why not report the 26 to 0 as the headline?**
The table it beat scored every candidate zero. The complete-table run is the
fair version, and we report both.

**Why did you need a cloud GPU?**
The table needed 2^42 samples. That would take too long on the laptop. The
attack itself still ran on the laptop.

**Did the MPS defect affect earlier results?**
No. Production B1 used 65,536-row batches, which gave correct scores. We
checked CPU against MPS on the same inputs.

**Why does containment decide everything at eight rounds?**
With three round keys fixed, a wrong candidate is rejected in about 0.25 s,
so the solver gets through the whole 512-entry queue in time. If the true chain
is in the queue, it is found.
