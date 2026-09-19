"""Independent clean-room reference oracle for SIMON32/64.

This module is intentionally completely decoupled from `neural_sat_speck.simon.*`.
It implements SIMON32/64 block encryption, decryption, and key expansion from
first principles directly from the Beaulieu et al. (2013) specification.
It serves as an independent second implementation for differential testing to
ensure the production codebase has zero self-testing circularities.
"""

from __future__ import annotations

# Exact 62-bit periodic sequence z0 defined in Beaulieu et al. (2013), Table 1
_Z0_BITS: tuple[int, ...] = (
    1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0,
    1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0,
)


def _oracle_rotl(x: int, n: int) -> int:
    shift = n % 16
    return (((x & 0xFFFF) << shift) | ((x & 0xFFFF) >> (16 - shift))) & 0xFFFF


def _oracle_rotr(x: int, n: int) -> int:
    shift = n % 16
    return (((x & 0xFFFF) >> shift) | ((x & 0xFFFF) << (16 - shift))) & 0xFFFF


def _oracle_f(x: int) -> int:
    return ((_oracle_rotl(x, 1) & _oracle_rotl(x, 8)) ^ _oracle_rotl(x, 2)) & 0xFFFF


def oracle_simon32_64_key_schedule(
    master_key: tuple[int, int, int, int],
    rounds: int = 32,
) -> tuple[int, ...]:
    """Expand a 64-bit master key (k3, k2, k1, k0) into round keys independently."""
    k3, k2, k1, k0 = master_key
    keys: list[int] = [k0 & 0xFFFF, k1 & 0xFFFF, k2 & 0xFFFF, k3 & 0xFFFF]
    for i in range(rounds - 4):
        tmp = _oracle_rotr(keys[i + 3], 3) ^ keys[i + 1]
        tmp = tmp ^ _oracle_rotr(tmp, 1)
        z_bit = _Z0_BITS[i % 62]
        c_i = 0xFFFC ^ z_bit
        k_next = (keys[i] ^ tmp ^ c_i) & 0xFFFF
        keys.append(k_next)
    return tuple(keys[:rounds])


def oracle_simon32_64_encrypt(
    plaintext: tuple[int, int],
    master_key: tuple[int, int, int, int],
    rounds: int = 32,
) -> tuple[int, int]:
    """Encrypt a 32-bit block (L, R) independently."""
    round_keys = oracle_simon32_64_key_schedule(master_key, rounds=rounds)
    left, right = plaintext[0] & 0xFFFF, plaintext[1] & 0xFFFF
    for k in round_keys:
        new_left = (right ^ _oracle_f(left) ^ k) & 0xFFFF
        new_right = left
        left, right = new_left, new_right
    return left, right


def oracle_simon32_64_decrypt(
    ciphertext: tuple[int, int],
    master_key: tuple[int, int, int, int],
    rounds: int = 32,
) -> tuple[int, int]:
    """Decrypt a 32-bit block (L, R) independently."""
    round_keys = oracle_simon32_64_key_schedule(master_key, rounds=rounds)
    left, right = ciphertext[0] & 0xFFFF, ciphertext[1] & 0xFFFF
    for k in reversed(round_keys):
        prev_left = right
        prev_right = (left ^ _oracle_f(right) ^ k) & 0xFFFF
        left, right = prev_left, prev_right
    return left, right
