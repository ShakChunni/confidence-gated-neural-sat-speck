"""Command-line interface for neural-sat-speck research artefact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neural_sat_speck.attacks.diversity_fallback import (
    PHASE6C_FALLBACK_BUDGET,
    PHASE6C_TRUSTED_BUDGET,
    choose_phase6c_route,
)
from neural_sat_speck.attacks.ranking import (
    ALL_FINAL_ROUND_SUBKEYS,
)
from neural_sat_speck.attacks.recovery import (
    independent_verification_pairs,
    verify_master_key,
)
from neural_sat_speck.confidence.calibration import (
    PlattCalibrator,
    top8_score_separation,
)
from neural_sat_speck.data.attack import (
    AttackInstanceConfig,
    generate_attack_instances,
)
from neural_sat_speck.data.seed_policy import get_cli_default_seed
from neural_sat_speck.speck.cipher import (
    SPECK32_64_KAT_CIPHERTEXT,
    SPECK32_64_KAT_KEY,
    SPECK32_64_KAT_PLAINTEXT,
    SPECK32_64_ROUND_COUNT,
    decrypt_block,
    encrypt_block,
)


def _cmd_kat(_args: argparse.Namespace) -> None:
    """Verify and print official SPECK32/64 Known-Answer Test."""
    ct = encrypt_block(
        SPECK32_64_KAT_PLAINTEXT,
        SPECK32_64_KAT_KEY,
        SPECK32_64_ROUND_COUNT,
    )
    pt = decrypt_block(ct, SPECK32_64_KAT_KEY, SPECK32_64_ROUND_COUNT)
    matches = ct == SPECK32_64_KAT_CIPHERTEXT and pt == SPECK32_64_KAT_PLAINTEXT

    print("SPECK32/64 Known-Answer Test (KAT):")
    print(f"  Key:        {[f'{x:#06x}' for x in SPECK32_64_KAT_KEY]}")
    print(f"  Plaintext:  (0x{SPECK32_64_KAT_PLAINTEXT[0]:04x}, 0x{SPECK32_64_KAT_PLAINTEXT[1]:04x})")
    print(f"  Ciphertext: (0x{ct[0]:04x}, 0x{ct[1]:04x})")
    print(f"  Expected:   (0x{SPECK32_64_KAT_CIPHERTEXT[0]:04x}, 0x{SPECK32_64_KAT_CIPHERTEXT[1]:04x})")
    print(f"  Verified:   {matches}")


def _cmd_generate_data(args: argparse.Namespace) -> None:
    """Generate synthetic attack instances."""
    cfg = AttackInstanceConfig(
        instance_count=args.instances,
        pair_count=args.pairs,
        rounds=args.rounds,
        seed=args.seed,
        split_role="test",
        input_difference=(0x0040, 0x0000),
    )
    collection = generate_attack_instances(cfg)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": collection.metadata,
        "instances": [
            {
                "instance_id": inst.instance_id,
                "master_key": list(inst.master_key),
                "final_round_subkey": inst.final_round_subkey,
                "pair_count": len(inst.pairs),
            }
            for inst in collection.instances
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {args.instances} attack instances saved to {out_path}")


def _cmd_explain_confidence(args: argparse.Namespace) -> None:
    """Explain confidence calibration and routing decision."""
    ranking_path = Path(args.ranking)
    data = json.loads(ranking_path.read_text(encoding="utf-8"))
    scores = data.get("candidate_scores") or data.get("ranking", {}).get("candidate_scores")
    order = data.get("candidate_order") or data.get("ranking", {}).get("candidate_order")

    if not scores or not order:
        print("Error: Invalid ranking file format", file=sys.stderr)
        return

    ranking_dict = {
        "ranking": {
            "candidate_values": list(ALL_FINAL_ROUND_SUBKEYS),
            "candidate_scores": list(float(s) for s in scores),
            "candidate_order": list(int(x) for x in order),
        }
    }
    feature = top8_score_separation(ranking_dict)

    # Standard frozen Phase 6C Platt calibrator
    calibrator = PlattCalibrator(
        feature_mean=4.12053918838501,
        feature_std=0.5482019782066345,
        intercept=0.5789959917731415,
        slope=0.07917109093865415,
        l2_lambda=1.0,
        fit_count=64,
        positive_count=41,
        iterations=10,
        gradient_tolerance=1e-10,
        step_tolerance=1e-12,
    )
    prob = calibrator.predict(feature)
    route = choose_phase6c_route(prob, args.threshold)

    print("Confidence Gating Analysis:")
    print(f"  Confidence Feature (Normalized Margin): {feature:.6f}")
    print(f"  Calibrated Top-8 Probability:          {prob * 100:.2f}%")
    print(f"  Selected Route:                        {route}")
    if route == "trusted_top8":
        print(f"  Action: Test Top-{PHASE6C_TRUSTED_BUDGET} candidates directly via incremental SAT.")
    else:
        print(f"  Action: Route to Bounded Diversity Fallback ({PHASE6C_FALLBACK_BUDGET} candidates).")


def _cmd_verify_key(args: argparse.Namespace) -> None:
    """Verify candidate master key with fresh independent pairs."""
    key = tuple(int(x, 0) if isinstance(x, str) else int(x) for x in args.key)
    rounds = args.rounds

    # Generate fresh verification pairs
    pairs = independent_verification_pairs(
        key,
        rounds=rounds,
        pair_count=args.pairs,
        seed=args.seed,
    )
    # Known pairs simulation
    known = pairs[: len(pairs) // 2]
    independent = pairs[len(pairs) // 2 :]

    ver = verify_master_key(key, known, independent, rounds)
    print(f"Cryptographic Verification for Key {[f'{x:#06x}' for x in key]} ({rounds} rounds):")
    print(f"  Known pairs verified:       {ver.known_pairs_verified}")
    print(f"  Independent pairs verified: {ver.independent_pairs_verified}")
    print(f"  Result:                     {'PASS (Sound Key)' if ver.verified else 'FAIL (Spurious Key)'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="neural-sat-speck",
        description="Learning When to Trust Neural Guidance: Confidence-Gated SAT Key Recovery for SPECK32/64",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # KAT
    subparsers.add_parser("kat", help="Verify official SPECK32/64 KAT")

    # Generate Data
    gen_parser = subparsers.add_parser("generate-data", help="Generate synthetic differential attack data")
    gen_parser.add_argument("--instances", type=int, default=10, help="Number of instances")
    gen_parser.add_argument("--pairs", type=int, default=64, help="Pairs per instance")
    gen_parser.add_argument("--rounds", type=int, default=6, help="Cipher rounds")
    gen_parser.add_argument("--seed", type=int, default=get_cli_default_seed(), help="Random seed (demo default: 999999)")
    gen_parser.add_argument("--output", type=str, default="data/sample_instances.json", help="Output path")

    # Explain Confidence
    conf_parser = subparsers.add_parser("explain-confidence", help="Explain confidence score and routing")
    conf_parser.add_argument("--ranking", type=str, required=True, help="Path to subkey ranking JSON")
    conf_parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")

    # Verify Key
    ver_parser = subparsers.add_parser("verify-key", help="Verify master key with fresh pairs")
    ver_parser.add_argument("--key", nargs=4, required=True, help="4-word 64-bit master key (e.g. 0x1918 0x1110 0x0908 0x0100)")
    ver_parser.add_argument("--rounds", type=int, default=6, help="Cipher rounds")
    ver_parser.add_argument("--pairs", type=int, default=4, help="Verification pairs count")
    ver_parser.add_argument("--seed", type=int, default=12345, help="Fresh verification seed")

    args = parser.parse_args()

    if args.command == "kat":
        _cmd_kat(args)
    elif args.command == "generate-data":
        _cmd_generate_data(args)
    elif args.command == "explain-confidence":
        _cmd_explain_confidence(args)
    elif args.command == "verify-key":
        _cmd_verify_key(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
