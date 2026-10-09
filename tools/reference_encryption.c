#include "reference_encryption.h"

static uint16_t rol16(uint16_t x, unsigned r) {
    return (uint16_t)(((uint32_t)x << r) | ((uint32_t)x >> (16U - r)));
}
static uint16_t ror16(uint16_t x, unsigned r) {
    return (uint16_t)(((uint32_t)x >> r) | ((uint32_t)x << (16U - r)));
}

void reference_expand(unsigned cipher, unsigned rounds, const reference_key *key,
                      uint16_t round_keys[32]) {
    if (cipher == 0) {
        uint16_t l[34] = {key->words[2], key->words[1], key->words[0]};
        round_keys[0] = key->words[3];
        for (unsigned i = 0; i + 1 < rounds; ++i) {
            l[i + 3] = (uint16_t)((uint16_t)(ror16(l[i], 7) + round_keys[i]) ^ i);
            round_keys[i + 1] = (uint16_t)(rol16(round_keys[i], 2) ^ l[i + 3]);
        }
    } else {
        /* z0 chronological bits, read from LSB; no textual bit-order ambiguity. */
        const uint64_t z0 = UINT64_C(0x19c3522fb386a45f);
        for (unsigned i = 0; i < 4 && i < rounds; ++i)
            round_keys[i] = key->words[3 - i];
        for (unsigned i = 0; i + 4 < rounds; ++i) {
            uint16_t tmp = (uint16_t)(ror16(round_keys[i + 3], 3) ^ round_keys[i + 1]);
            tmp = (uint16_t)(tmp ^ ror16(tmp, 1));
            round_keys[i + 4] = (uint16_t)(round_keys[i] ^ tmp ^
                UINT16_C(0xfffc) ^ ((z0 >> (i % 62)) & 1));
        }
    }
}

reference_block reference_encrypt(unsigned cipher, unsigned rounds,
                                  reference_block input,
                                  const uint16_t round_keys[32]) {
    uint16_t a = input.a, b = input.b;
    if (cipher == 0) {
        for (unsigned i = 0; i < rounds; ++i) {
            a = (uint16_t)((uint16_t)(ror16(a, 7) + b) ^ round_keys[i]);
            b = (uint16_t)(rol16(b, 2) ^ a);
        }
    } else {
        for (unsigned i = 0; i < rounds; ++i) {
            uint16_t next = (uint16_t)(b ^ (rol16(a, 1) & rol16(a, 8)) ^
                                       rol16(a, 2) ^ round_keys[i]);
            b = a;
            a = next;
        }
    }
    return (reference_block){a, b};
}

__attribute__((noinline))
void reference_batch(unsigned cipher, unsigned rounds, unsigned charge_schedule,
                     size_t count, const reference_block *inputs,
                     const reference_key *keys, const uint16_t fixed_schedule[32],
                     reference_block *outputs) {
    for (size_t i = 0; i < count; ++i) {
        uint16_t schedule[32];
        const uint16_t *rk = fixed_schedule;
        if (charge_schedule) {
            reference_expand(cipher, rounds, &keys[i], schedule);
            rk = schedule;
        }
        outputs[i] = reference_encrypt(cipher, rounds, inputs[i], rk);
    }
}
