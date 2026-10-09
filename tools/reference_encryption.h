#ifndef REFERENCE_ENCRYPTION_H
#define REFERENCE_ENCRYPTION_H
#include <stddef.h>
#include <stdint.h>

typedef struct { uint16_t a, b; } reference_block;
typedef struct { uint16_t words[4]; } reference_key;

/* cipher 0: SPECK32/64; cipher 1: SIMON32/64. Words are high-to-low. */
void reference_expand(unsigned cipher, unsigned rounds, const reference_key *key,
                      uint16_t round_keys[32]);
reference_block reference_encrypt(unsigned cipher, unsigned rounds,
                                  reference_block input,
                                  const uint16_t round_keys[32]);
/* Separate translation unit, no LTO: every timed call must write outputs. */
void reference_batch(unsigned cipher, unsigned rounds, unsigned charge_schedule,
                     size_t count, const reference_block *inputs,
                     const reference_key *keys, const uint16_t fixed_schedule[32],
                     reference_block *outputs);
#endif
