#include "bruteforce_completion.h"

#include "reference_encryption.h"

/*
 * Single-threaded, allocation-free completion. The forward schedule and the
 * block cipher are the verified functions in reference_encryption.c; this file
 * adds only the inverse key schedule and the enumeration loop.
 *
 * SPECK32/64 inversion (derived from the forward recurrence in
 * reference_expand, with l_{j+3} = (ror(l_j,7) + k_j) ^ j and
 * k_{j+1} = rol(k_j,2) ^ l_{j+3}):
 *   The schedule state at index j is S_j = (k_j; l_j, l_{j+1}, l_{j+2}); the
 *   forward step S_j -> S_{j+1} is a bijection on 64 bits. Four consecutive
 *   round keys k_{R-4..R-1} determine S_{R-1} exactly:
 *     l_{R-1} = k_{R-3} ^ rol(k_{R-4}, 2)
 *     l_R     = k_{R-2} ^ rol(k_{R-3}, 2)
 *     l_{R+1} = k_{R-1} ^ rol(k_{R-2}, 2)
 *   Backward step from S_{j+1} = (k; a, b, c) with c = l_{j+3}:
 *     k_j = ror(k ^ c, 2)
 *     l_j = rol(((c ^ j) - k_j) mod 2^16, 7)
 *     S_j = (k_j; l_j, a, b)
 *   Applied for j = R-2 down to 0; S_0 = (k_0; l_0, l_1, l_2) gives the master
 *   key words (l_2, l_1, l_0, k_0).
 *
 * SIMON32/64 inversion (forward: k_{i+4} = k_i ^ T(k_{i+3}, k_{i+1}) ^ 0xfffc
 * ^ z_i with T(x, y) = t ^ ror(t, 1), t = ror(x, 3) ^ y):
 *   k_i = k_{i+4} ^ T(k_{i+3}, k_{i+1}) ^ 0xfffc ^ z_i, for i = R-5 down to 0.
 *   The window k_{R-4..R-1} is the starting point; master key (k_3, k_2, k_1,
 *   k_0) in reference_key.words order.
 */

#define BF_SIMON_Z0 UINT64_C(0x19c3522fb386a45f)
#define BF_SIMON_C UINT16_C(0xfffc)

static uint16_t bf_rol16(uint16_t x, unsigned r) {
    return (uint16_t)(((uint32_t)x << r) | ((uint32_t)x >> (16U - r)));
}

static uint16_t bf_ror16(uint16_t x, unsigned r) {
    return (uint16_t)(((uint32_t)x >> r) | ((uint32_t)x << (16U - r)));
}

static int bf_valid_cipher_rounds(unsigned cipher, unsigned rounds, unsigned minimum) {
    if (cipher > 1U) {
        return 0;
    }
    unsigned maximum = cipher == 0U ? 22U : 32U;
    return rounds >= minimum && rounds <= maximum;
}

/* Preconditions: rounds >= 4 and the cipher is valid. */
static void bf_invert_window(unsigned cipher, unsigned rounds,
                             const uint16_t w[4], uint16_t mk[4]) {
    if (cipher == 0U) {
        uint16_t k = w[3];                                          /* k_{R-1} */
        uint16_t a = (uint16_t)(w[1] ^ bf_rol16(w[0], 2));          /* l_{R-1} */
        uint16_t b = (uint16_t)(w[2] ^ bf_rol16(w[1], 2));          /* l_R     */
        uint16_t c = (uint16_t)(w[3] ^ bf_rol16(w[2], 2));          /* l_{R+1} */
        for (int j = (int)rounds - 2; j >= 0; --j) {
            uint16_t kn = bf_ror16((uint16_t)(k ^ c), 2);
            uint16_t ln = bf_rol16((uint16_t)((uint16_t)(c ^ (uint16_t)j) - kn), 7);
            c = b;
            b = a;
            a = ln;
            k = kn;
        }
        /* S_0 = (k_0; l_0 = a, l_1 = b, l_2 = c). Words are (l_2, l_1, l_0, k_0). */
        mk[0] = c;
        mk[1] = b;
        mk[2] = a;
        mk[3] = k;
    } else {
        uint16_t k[32] = {0};
        for (unsigned t = 0; t < 4U; ++t) {
            k[rounds - 4U + t] = w[t];
        }
        for (int i = (int)rounds - 5; i >= 0; --i) {
            uint16_t t = (uint16_t)(bf_ror16(k[i + 3], 3) ^ k[i + 1]);
            t = (uint16_t)(t ^ bf_ror16(t, 1));
            uint16_t z = (uint16_t)((BF_SIMON_Z0 >> ((unsigned)i % 62U)) & 1U);
            k[i] = (uint16_t)(k[i + 4] ^ t ^ BF_SIMON_C ^ z);
        }
        /* reference_key.words[3 - i] = k_i. */
        mk[0] = k[3];
        mk[1] = k[2];
        mk[2] = k[1];
        mk[3] = k[0];
    }
}

int bf_expand(unsigned cipher, unsigned rounds, const uint16_t key[4],
              uint16_t round_keys[32]) {
    if (key == NULL || round_keys == NULL || !bf_valid_cipher_rounds(cipher, rounds, 1U)) {
        return BF_EINVAL;
    }
    reference_key rkey = {{key[0], key[1], key[2], key[3]}};
    reference_expand(cipher, rounds, &rkey, round_keys);
    return 0;
}

int bf_encrypt_block(unsigned cipher, unsigned rounds, const uint16_t key[4],
                     uint16_t block[2]) {
    if (key == NULL || block == NULL || !bf_valid_cipher_rounds(cipher, rounds, 1U)) {
        return BF_EINVAL;
    }
    uint16_t round_keys[32];
    reference_key rkey = {{key[0], key[1], key[2], key[3]}};
    reference_expand(cipher, rounds, &rkey, round_keys);
    reference_block out = reference_encrypt(
        cipher, rounds, (reference_block){block[0], block[1]}, round_keys);
    block[0] = out.a;
    block[1] = out.b;
    return 0;
}

int bf_master_from_window(unsigned cipher, unsigned rounds,
                          const uint16_t window[4], uint16_t key[4]) {
    if (window == NULL || key == NULL || !bf_valid_cipher_rounds(cipher, rounds, 4U)) {
        return BF_EINVAL;
    }
    uint16_t mk[4];
    bf_invert_window(cipher, rounds, window, mk);
    uint16_t round_keys[32];
    reference_key rkey = {{mk[0], mk[1], mk[2], mk[3]}};
    reference_expand(cipher, rounds, &rkey, round_keys);
    for (unsigned t = 0; t < 4U; ++t) {
        if (round_keys[rounds - 4U + t] != window[t]) {
            return BF_EINCONSISTENT;
        }
    }
    for (unsigned t = 0; t < 4U; ++t) {
        key[t] = mk[t];
    }
    return 0;
}

int bf_complete(unsigned cipher, unsigned rounds, unsigned m,
                const uint16_t *trailing, size_t nblocks, const uint16_t *pt,
                const uint16_t *ct, uint64_t guess_lo, uint64_t guess_hi,
                uint16_t out_key[4], uint64_t *tested, uint64_t *found_guess) {
    if (found_guess != NULL) {
        *found_guess = 0;
    }
    if (tested == NULL) {
        return BF_EINVAL;
    }
    *tested = 0;
    if (trailing == NULL || pt == NULL || ct == NULL || out_key == NULL) {
        return BF_EINVAL;
    }
    if (!bf_valid_cipher_rounds(cipher, rounds, 4U) || m < 1U || m > 3U || nblocks == 0U) {
        return BF_EINVAL;
    }
    const unsigned unknown = 4U - m;
    const uint64_t space = UINT64_C(1) << (16U * unknown);
    if (guess_lo > guess_hi || guess_hi > space) {
        return BF_EINVAL;
    }

    for (uint64_t g = guess_lo; g < guess_hi; ++g) {
        uint16_t w[4];
        for (unsigned t = 0; t < unknown; ++t) {
            w[t] = (uint16_t)(g >> (16U * t));
        }
        for (unsigned t = 0; t < m; ++t) {
            w[unknown + t] = trailing[t];
        }

        uint16_t mk[4];
        bf_invert_window(cipher, rounds, w, mk);
        uint16_t round_keys[32];
        reference_key rkey = {{mk[0], mk[1], mk[2], mk[3]}};
        reference_expand(cipher, rounds, &rkey, round_keys);
        for (unsigned t = 0; t < 4U; ++t) {
            if (round_keys[rounds - 4U + t] != w[t]) {
                *tested = g - guess_lo + 1U;
                return BF_EINCONSISTENT;
            }
        }

        reference_block b0 = reference_encrypt(
            cipher, rounds, (reference_block){pt[0], pt[1]}, round_keys);
        if (b0.a != ct[0] || b0.b != ct[1]) {
            continue;
        }
        int ok = 1;
        for (size_t i = 1; i < nblocks && ok; ++i) {
            reference_block c = reference_encrypt(
                cipher, rounds, (reference_block){pt[2 * i], pt[2 * i + 1]}, round_keys);
            ok = (c.a == ct[2 * i]) && (c.b == ct[2 * i + 1]);
        }
        if (ok) {
            for (unsigned t = 0; t < 4U; ++t) {
                out_key[t] = mk[t];
            }
            *tested = g - guess_lo + 1U;
            if (found_guess != NULL) {
                *found_guess = g;
            }
            return 1;
        }
    }
    *tested = guess_hi - guess_lo;
    return 0;
}
