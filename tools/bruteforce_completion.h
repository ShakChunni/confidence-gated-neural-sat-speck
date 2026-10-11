#ifndef BRUTEFORCE_COMPLETION_H
#define BRUTEFORCE_COMPLETION_H
#include <stddef.h>
#include <stdint.h>

/*
 * Brute-force completion of a reduced-round SPECK32/64 or SIMON32/64 master key
 * from fixed trailing round keys and known plaintext/ciphertext blocks.
 *
 * Conventions (identical to tools/reference_encryption.{c,h}):
 *   cipher 0 = SPECK32/64 (rounds 4..22 for completion), cipher 1 = SIMON32/64
 *   (rounds 4..32 for completion). Keys are reference_key.words order.
 *   Blocks are (a, b): SPECK a = x (left), b = y; SIMON a = L, b = R.
 *   Round keys are chronological: round_keys[i] = k_i.
 *
 * Unknown space: the 4-m round keys k_{R-4}, ..., k_{R-m-1}. A guess g in
 * [0, 2^(16*(4-m))) packs them as u_t = k_{R-4+t} at bits [16t, 16t+16), i.e.
 * k_{R-4} is the least significant word. Trailing words are chronological:
 * trailing[t] = k_{R-m+t}. Every guess determines exactly one master key
 * (the window-to-master-key map is a bijection), so no master key is visited
 * twice.
 *
 * Return codes:
 *   bf_complete: 1 = found and fully verified (out_key written), 0 = no guess in
 *   range matched, BF_EINVAL = invalid argument, BF_EINCONSISTENT = the
 *   forward expansion of an inverted schedule failed to reproduce its window
 *   (an internal bug; should never occur).
 *   Helper functions: 0 = success, BF_EINVAL / BF_EINCONSISTENT on error.
 */
#define BF_EINVAL (-1)
#define BF_EINCONSISTENT (-2)

/* Forward key schedule via the verified reference expansion. */
int bf_expand(unsigned cipher, unsigned rounds, const uint16_t key[4],
              uint16_t round_keys[32]);

/* In-place encryption of one block under the master key (reference path). */
int bf_encrypt_block(unsigned cipher, unsigned rounds, const uint16_t key[4],
                     uint16_t block[2]);

/*
 * Invert the schedule from the window k_{R-4..R-1} (chronological) to the master
 * key, then check that the forward expansion reproduces the window.
 */
int bf_master_from_window(unsigned cipher, unsigned rounds,
                          const uint16_t window[4], uint16_t key[4]);

/*
 * Enumerate guesses g in [guess_lo, guess_hi) over the 16*(4-m) unknown bits
 * (see packing above). pt and ct hold 2*nblocks words (a0, b0, a1, b1, ...).
 * On a match, all nblocks are verified; the first matching guess in ascending
 * order is returned. *tested is the number of guesses tried.
 * found_guess may be NULL; when non-NULL it receives the matching guess.
 *
 * Caution: with few blocks, wrong keys can match by chance (about 2^(16*(4-m))
 * * 2^-32 per block-0 match). Callers must verify the recovered key on fresh
 * pairs before reporting it.
 */
int bf_complete(unsigned cipher, unsigned rounds, unsigned m,
                const uint16_t *trailing, size_t nblocks, const uint16_t *pt,
                const uint16_t *ct, uint64_t guess_lo, uint64_t guess_hi,
                uint16_t out_key[4], uint64_t *tested, uint64_t *found_guess);

#endif
