# D-058: measured reference-encryption calibration

Classification: ENGINEERING. Authorized by the author's request on 2026-10-09
to perform measured encryption-cost calibration. This protocol is fixed before
any calibration timing. It changes no attack method, outcome or cutoff.

## Question and estimand

Measure local, single-thread native reference block throughput E (blocks/s),
then express the existing 300-second allowance as 300E reference encryptions
and log2(300E). Measure SPECK32/64 at R=7 and R=22, and SIMON32/64 at R=10 and
R=32. Full-round reference rates and study-round rates remain separate.

For each round profile, measure two explicitly distinct scopes:

- `expanded_key`: one fixed key, schedule generated before timing; independent
  plaintext blocks encrypted using that schedule. Primary reference rate.
- `schedule_charged`: a different public fixture key per block; schedule expansion
  and encryption both timed. Secondary sensitivity, not the conventional rate.

Both include input reads, output writes, scalar batch loops and clock checks.
Allocation, fixture generation, process launch, warm-up and final checksum are
excluded. This is warm-buffer bulk throughput, not dependent-block latency or
the fastest possible SIMD/GPU implementation.

## Frozen workload and environment

See `encryption_cost_calibration_manifest_20261009.json`. There are eight profiles
and seven repetitions per profile. Each invocation warms up for at least 0.5 s
then measures complete batches for at least 2.0 s. The eight-profile order is
rotated by one position on successive repetitions. Preserve all 56 samples;
report median, IQR and min/max. These are descriptive spreads, not confidence
intervals. Each invocation has a 15 s subprocess timeout; the timing campaign
has a 240 s wall ceiling. No rate-based stopping, trimming or parameter tuning.

Batch size is 65,536 independent blocks. Plaintexts are the unique 32-bit values
`(i * 0x9e3779b1 + 0x65656877) mod 2^32`, split high word first. Per-block keys
are the 16-bit high-to-low words of SplitMix64 outputs from public fixture state
`0xe5c0202610096002`. This is a performance fixture, not an attack dataset,
training family or protected seed. The fixed key is `(1918,1110,0908,0100)` hex.
Preallocated plaintext/output/key buffers total 1 MiB; no bulk schedule cache.

Compile two separate C11 translation units, without LTO, with explicit O3 and
disabled loop/SLP vectorization. The externally visible batch function is
noinline. The driver cannot remove or hoist its opaque output-writing calls;
every output contributes to a checked post-timing 64-bit digest.
The digest starts at 0xcbf29ce484222325 and folds each high-to-low packed output
word `(a<<16)|b` by XOR then multiplication by 0x100000001b3 modulo 2^64. It is
an output-consumption check, not a cryptographic integrity hash; files use SHA-256.
Record compiler command, source/object/executable hashes, CPU model/core counts, OS, one-thread
policy, process priority/QoS, power mode and thermal observations. No CPU affinity
or fixed frequency is claimed. Power observations are retained, including drift.
On Darwin the measured main thread explicitly requests USER_INITIATED (25) QoS;
nice must be zero. This fixes a foreground-computation profile without using
the higher USER_INTERACTIVE (33) class recorded by some earlier attack workers.
The classes are not claimed identical. The policy is checked before timing.
Darwin process background state must also be zero. The compiled driver object's
undefined symbols must include the external batch call, retained as a receipt.
Word precision is uint16 with uint32 rotate intermediates and uint64 counters/
digest; clock is CLOCK_MONOTONIC represented in double seconds. Native peak RSS
must remain below 64 MiB. Array memory is separate from whole-process peak RSS.
No concurrent outcome-bearing attack is permitted. Record the process check.
Scan known repository runners/workers and native solvers before each sample;
record host load. This guards against known concurrent attacks, not arbitrary
external CPU use or an unobservable process-start race. Observer calls are
bounded by the remaining campaign allowance. Pin actual imported Python paths
as well as native inputs before compilation; recheck identities after gates and
during timing. Bind every sample to the same frozen execution identity.

## Correctness gates and independent review

Before timing, verify both official full-round known-answer vectors and all
round prefixes on deterministic edge cases against the scalar Python reference.
Compare round keys as well as ciphertexts. SIMON also uses the separate clean-room
oracle. SPECK's independent anchors are its official vector and existing reduced
regression fixtures, not a claim of an additional independent implementation.
Verify all 65,536 fixture outputs in each of the eight profiles via a full-batch
digest computed independently in Python. Require the same expected digest for
every timed sample. Check distinct plaintexts/keys and unsigned wraparound.

Check malformed
arguments, invalid rounds, timeout handling, duplicate/missing samples, nonfinite
rates and drifted profile/source identities. A KAT, schedule, output digest,
sample-grid or identity mismatch invalidates the calibration. No result is
selected for favorable throughput. Failed invocations remain recorded.

## Interpretation and evidence outputs

The conversion is a local elapsed-time normalization. It does not count SAT
conflicts, instructions, FLOPs or heterogeneous CPU/GPU work, establish a
hardware-independent attack complexity, or prove 300 s is uniquely justified.
Apply one declared reference rate to every method within a cipher/round profile.
Historical attacks were not run alongside this calibration; frequency, thermal
state and implementation optimization can differ. Keep original seconds.

Export raw samples, correctness receipt, compiler/host observations, frozen run
identity, summary JSON/CSV and SHA-256 index under
`docs/encryption_cost_evidence_20261009/`; report in
`docs/paper/06_ENCRYPTION_COST_CALIBRATION_REPORT.md`. Binary/objects are local
runtime artifacts with recorded hashes and reproducible compile commands.

Extract cumulative recoveries at 100/200/300 s from the already recorded D-031,
SIMON replacement and D-056 executions, with all assigned keys and input hashes.
These describe the recorded policies, not new shorter-deadline experiments.
Keep 128 attack plaintext queries plus two fresh verification blocks distinct
from computation, training, table generation and asset storage. The learned
models and classical tables are offline assets; their actual provenance/resource
records remain separate and missing costs are not set to zero.

## Sources

The cipher definitions/KAT provenance is the original Beaulieu et al. paper,
[ePrint 2013/404](https://eprint.iacr.org/2013/404), identified by the
[official project repository](https://github.com/nsacyber/simon-speck).
Its later implementation guide covers 64/128-bit block variants, so we do not
claim it supplies the 32-bit code used here. The executed equations, word order
and regression fixtures are pinned from this repository's reviewed SPECK/SIMON
references. The C implementation is newly written for this measurement.
