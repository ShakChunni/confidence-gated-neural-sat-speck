"""Word-level SPECK32/64 regression values.

The round-key sequence is an independently calculated regression fixture from
the source equations. The primary source supplies the complete block-cipher
known-answer vector; it does not publish this intermediate-key table.
"""

MASTER_KEY = (0x1918, 0x1110, 0x0908, 0x0100)
PLAINTEXT = (0x6574, 0x694C)
CIPHERTEXT = (0xA868, 0x42F2)

# These reduced-round states were calculated independently from the verified
# round equation and the expected round-key sequence below.
EXPECTED_REDUCED_CIPHERTEXTS = (
    (1, (0x5316, 0xF627)),
    (2, (0x37DF, 0xEF40)),
    (7, (0xE786, 0x574E)),
    (11, (0xCC25, 0xAC1B)),
    (22, CIPHERTEXT),
)

EXPECTED_ROUND_KEYS = (
    0x0100,
    0x1512,
    0x617D,
    0x1458,
    0x6919,
    0x77E2,
    0x0C89,
    0xCCDB,
    0xEFEA,
    0x4E33,
    0x76F4,
    0x5976,
    0xEE8B,
    0xDB04,
    0x4617,
    0xF37E,
    0x87B4,
    0x8ECA,
    0xED9B,
    0x3A52,
    0x8229,
    0xED64,
)
