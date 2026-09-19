"""Word-level SIMON32/64 authoritative vector and internal regression fixtures.

The complete block known-answer test vector (MASTER_KEY, PLAINTEXT, CIPHERTEXT)
is authoritative and taken directly from:
    Ray Beaulieu, Douglas Shors, Jason Smith, Stefan Treatman-Clark, Bryan Weeks,
    and Louis Wingers. "The SIMON and SPECK Families of Lightweight Block Ciphers."
    Cryptology ePrint Archive, Report 2013/404, Appendix: Test Vectors.

The intermediate round keys (EXPECTED_ROUND_KEYS) and reduced-round ciphertexts
(EXPECTED_REDUCED_CIPHERTEXTS) are NOT published in the primary source; they are
repository regression fixtures calculated from the mathematical recurrences in the
specification to verify internal consistency across pipeline stages.
"""

# Authoritative Known Answer Test Vector (Beaulieu et al., 2013)
MASTER_KEY = (0x1918, 0x1110, 0x0908, 0x0100)
PLAINTEXT = (0x6565, 0x6877)
CIPHERTEXT = (0xC69B, 0xE9BB)

# Repository Regression Fixtures: Intermediate round keys
# Calculated independently from the key schedule recurrence for regression testing
EXPECTED_ROUND_KEYS = (
    0x0100,  # k0
    0x0908,  # k1
    0x1110,  # k2
    0x1918,  # k3
    0x71C3,  # k4
    0xB649,  # k5
    0x56D4,  # k6
    0xE070,  # k7
    0xF15A,  # k8
    0xC535,  # k9
    0xDD94,  # k10
    0x4010,  # k11
    0x250A,  # k12
    0x6F66,  # k13
    0xE96B,  # k14
    0x4BD8,  # k15
    0x0FE5,  # k16
    0x7C47,  # k17
    0xE0EF,  # k18
    0x3E21,  # k19
    0x065B,  # k20
    0x438C,  # k21
    0xF26A,  # k22
    0xB5C0,  # k23
    0x8609,  # k24
    0x9F8E,  # k25
    0xD8BF,  # k26
    0x09AC,  # k27
    0xE812,  # k28
    0x2710,  # k29
    0x2CAA,  # k30
    0x8D14,  # k31
)

# Repository Regression Fixtures: Reduced-round ciphertexts
# Calculated independently from the round function recurrence for regression testing
EXPECTED_REDUCED_CIPHERTEXTS = (
    (1, (0xBCA2, 0x6565)),
    (2, (0xBEE3, 0xBCA2)),
    (4, (0x5327, 0x37BA)),
    (8, (0x873B, 0x8FCF)),
    (16, (0x6FC2, 0x1587)),
    (24, (0x0D11, 0x6041)),
    (32, CIPHERTEXT),
)
