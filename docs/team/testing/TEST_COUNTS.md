# Test counts

Collected on Windows 11 with Python 3.12.14 on 31 October 2026 from commit `78c1bd27257a364bf822d105a298085c7b963c90` on `hamza/testing`. Collection uses `TEMP` and `TMP` set to a writable workspace directory. Counts are platform-specific; POSIX-only modules excluded by `tests/conftest.py` are not counted.

Full collection: **1,466 tests**; unit: **1,377**; integration: **59**. The remaining **30** are root-level tests. Independent collection commands and outputs follow the area breakdown.

| Area | Collected tests |
|---|---:|
| cipher | 141 |
| key schedule | 19 |
| SAT encoding | 119 |
| solver isolation | 79 |
| ranking and neural | 390 |
| statistics | 68 |
| brute-force completer | 24 |
| run guards | 223 |
| other | 314 |
| **Total unit tests** | **1377** |

Each collected unit file is assigned once, based on its test subject and module description. Mixed modules belong to one primary area; `test_simon_cipher.py` includes key-schedule tests but is counted wholly under cipher. Figure/evidence/data tests without a closer category are other. Platform-excluded files contribute no collected tests.

## cipher

- `tests/unit/test_cipher.py`
- `tests/unit/test_partial.py`
- `tests/unit/test_primitives.py`
- `tests/unit/test_simon_cipher.py`

```text
COMMAND: <work>/hamza-venv/Scripts/python.exe -m pytest --collect-only -q tests/unit/test_cipher.py tests/unit/test_partial.py tests/unit/test_primitives.py tests/unit/test_simon_cipher.py
tests/unit/test_cipher.py::test_one_round_encryption_matches_regression_state
tests/unit/test_cipher.py::test_one_round_decryption_is_the_inverse
tests/unit/test_cipher.py::test_full_known_answer_vector
tests/unit/test_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[1-expected0]
tests/unit/test_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[2-expected1]
tests/unit/test_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[7-expected2]
tests/unit/test_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[11-expected3]
tests/unit/test_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[22-expected4]
tests/unit/test_cipher.py::test_deterministic_random_round_trips
tests/unit/test_cipher.py::test_round_and_block_inputs_are_explicitly_masked
tests/unit/test_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block-0]
tests/unit/test_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block--1]
tests/unit/test_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block-23]
tests/unit/test_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block-0]
tests/unit/test_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block--1]
tests/unit/test_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block-23]
tests/unit/test_cipher.py::test_non_integer_round_counts_are_rejected[encrypt_block-1.0]
tests/unit/test_cipher.py::test_non_integer_round_counts_are_rejected[encrypt_block-True]
tests/unit/test_cipher.py::test_non_integer_round_counts_are_rejected[decrypt_block-1.0]
tests/unit/test_cipher.py::test_non_integer_round_counts_are_rejected[decrypt_block-True]
tests/unit/test_cipher.py::test_malformed_round_state_is_rejected
tests/unit/test_partial.py::test_true_final_round_key_recovers_the_pre_final_state
tests/unit/test_partial.py::test_wrong_final_round_key_does_not_recover_the_pre_final_state
tests/unit/test_partial.py::test_partial_decryption_matches_inverse_round_for_seeded_states
tests/unit/test_primitives.py::test_mask_word_wraps_to_16_bits[0-0]
tests/unit/test_primitives.py::test_mask_word_wraps_to_16_bits[1-1]
tests/unit/test_primitives.py::test_mask_word_wraps_to_16_bits[65535-65535]
tests/unit/test_primitives.py::test_mask_word_wraps_to_16_bits[65536-0]
tests/unit/test_primitives.py::test_mask_word_wraps_to_16_bits[-1-65535]
tests/unit/test_primitives.py::test_rotations_use_16_bit_words[4660-0-4660-4660]
tests/unit/test_primitives.py::test_rotations_use_16_bit_words[4660-4-9025-16675]
tests/unit/test_primitives.py::test_rotations_use_16_bit_words[4660-16-4660-4660]
tests/unit/test_primitives.py::test_rotations_use_16_bit_words[65535-7-65535-65535]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-0]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-1]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-2]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-7]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-15]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-16]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-31]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-0]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-1]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-2]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-7]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-15]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-16]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-31]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-0]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-1]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-2]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-7]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-15]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-16]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-31]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-0]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-1]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-2]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-7]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-15]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-16]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-31]
tests/unit/test_primitives.py::test_rotations_normalise_negative_counts
tests/unit/test_primitives.py::test_add_words_wraps
tests/unit/test_primitives.py::test_subtract_words_wraps
tests/unit/test_primitives.py::test_addition_and_subtraction_are_inverses[0-0]
tests/unit/test_primitives.py::test_addition_and_subtraction_are_inverses[1-2]
tests/unit/test_primitives.py::test_addition_and_subtraction_are_inverses[4660-65244]
tests/unit/test_primitives.py::test_addition_and_subtraction_are_inverses[65535-65535]
tests/unit/test_primitives.py::test_non_integer_word_inputs_are_rejected[mask_word]
tests/unit/test_primitives.py::test_non_integer_word_inputs_are_rejected[rotate_left]
tests/unit/test_primitives.py::test_non_integer_word_inputs_are_rejected[rotate_right]
tests/unit/test_primitives.py::test_boolean_inputs_are_rejected
tests/unit/test_simon_cipher.py::test_one_round_encryption_matches_regression_state
tests/unit/test_simon_cipher.py::test_one_round_decryption_is_the_inverse
tests/unit/test_simon_cipher.py::test_full_known_answer_vector
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[1-expected0]
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[2-expected1]
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[4-expected2]
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[8-expected3]
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[16-expected4]
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[24-expected5]
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[32-expected6]
tests/unit/test_simon_cipher.py::test_key_schedule_matches_all_expected_round_keys
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[1]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[2]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[3]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[4]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[5]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[6]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[7]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[8]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[9]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[10]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[11]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[12]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[13]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[14]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[15]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[16]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[17]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[18]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[19]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[20]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[21]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[22]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[23]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[24]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[25]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[26]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[27]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[28]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[29]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[30]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[31]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[32]
tests/unit/test_simon_cipher.py::test_partial_decrypt_final_round_matches_prior_round
tests/unit/test_simon_cipher.py::test_simon_pair_difference_needs_candidate_sensitive_descent
tests/unit/test_simon_cipher.py::test_direct_pair_ranking_difference_has_r_minus_two_then_r_minus_three_depth
tests/unit/test_simon_cipher.py::test_deterministic_random_round_trips
tests/unit/test_simon_cipher.py::test_independent_reference_oracle_kat
tests/unit/test_simon_cipher.py::test_independent_reference_oracle_differential_random_comparison
tests/unit/test_simon_cipher.py::test_round_and_block_inputs_are_explicitly_masked
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block-0]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block--1]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block-33]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block-100]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block-0]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block--1]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block-33]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block-100]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[encrypt_block-1.0]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[encrypt_block-True]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[encrypt_block-32]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[encrypt_block-None]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[decrypt_block-1.0]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[decrypt_block-True]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[decrypt_block-32]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[decrypt_block-None]
tests/unit/test_simon_cipher.py::test_malformed_round_state_is_rejected
tests/unit/test_simon_cipher.py::test_key_schedule_validation
tests/unit/test_simon_cipher.py::test_primitives_properties

141 tests collected in 11.30s

Exit code: 0
Elapsed seconds: 14.547

```

## key schedule

- `tests/unit/test_key_schedule.py`

```text
COMMAND: <work>/hamza-venv/Scripts/python.exe -m pytest --collect-only -q tests/unit/test_key_schedule.py
tests/unit/test_key_schedule.py::test_full_key_schedule_matches_regression_values
tests/unit/test_key_schedule.py::test_reduced_key_schedule_is_a_prefix[1]
tests/unit/test_key_schedule.py::test_reduced_key_schedule_is_a_prefix[2]
tests/unit/test_key_schedule.py::test_reduced_key_schedule_is_a_prefix[7]
tests/unit/test_key_schedule.py::test_reduced_key_schedule_is_a_prefix[11]
tests/unit/test_key_schedule.py::test_reduced_key_schedule_is_a_prefix[22]
tests/unit/test_key_schedule.py::test_key_schedule_is_deterministic
tests/unit/test_key_schedule.py::test_key_schedule_outputs_16_bit_words
tests/unit/test_key_schedule.py::test_key_schedule_constants_match_verified_parameters
tests/unit/test_key_schedule.py::test_master_key_words_are_explicitly_masked
tests/unit/test_key_schedule.py::test_invalid_round_counts_are_rejected[0]
tests/unit/test_key_schedule.py::test_invalid_round_counts_are_rejected[-1]
tests/unit/test_key_schedule.py::test_invalid_round_counts_are_rejected[23]
tests/unit/test_key_schedule.py::test_non_integer_round_count_is_rejected
tests/unit/test_key_schedule.py::test_boolean_round_count_is_rejected
tests/unit/test_key_schedule.py::test_wrong_master_key_length_is_rejected[master_key0]
tests/unit/test_key_schedule.py::test_wrong_master_key_length_is_rejected[master_key1]
tests/unit/test_key_schedule.py::test_wrong_master_key_length_is_rejected[master_key2]
tests/unit/test_key_schedule.py::test_non_integer_master_key_word_is_rejected

19 tests collected in 0.34s

Exit code: 0
Elapsed seconds: 2.141

```

## SAT encoding

- `tests/unit/test_backend_formula.py`
- `tests/unit/test_benchmark_formula_export.py`
- `tests/unit/test_dimacs.py`
- `tests/unit/test_pure_cnf.py`
- `tests/unit/test_pure_cnf_adversarial.py`
- `tests/unit/test_sat_encoding.py`
- `tests/unit/test_simon_sat.py`
- `tests/unit/test_xor_to_cnf.py`

```text
COMMAND: <work>/hamza-venv/Scripts/python.exe -m pytest --collect-only -q tests/unit/test_backend_formula.py tests/unit/test_benchmark_formula_export.py tests/unit/test_dimacs.py tests/unit/test_pure_cnf.py tests/unit/test_pure_cnf_adversarial.py tests/unit/test_sat_encoding.py tests/unit/test_simon_sat.py tests/unit/test_xor_to_cnf.py
tests/unit/test_backend_formula.py::test_formula_has_independent_reference_witness_and_chronological_assumptions[speck32_64-4-key0]
tests/unit/test_backend_formula.py::test_formula_has_independent_reference_witness_and_chronological_assumptions[simon32_64-5-key1]
tests/unit/test_backend_formula.py::test_unguided_build_never_reads_truth_or_verification_fields[speck32_64-4-key0]
tests/unit/test_backend_formula.py::test_unguided_build_never_reads_truth_or_verification_fields[simon32_64-5-key1]
tests/unit/test_backend_formula.py::test_observation_prefix_does_not_read_unselected_pairs
tests/unit/test_backend_formula.py::test_reference_key_conflicts_with_corrupted_known_ciphertext
tests/unit/test_backend_formula.py::test_speck_native_output_matches_existing_encoder_bytes
tests/unit/test_backend_formula.py::test_speck_conditioned_output_matches_existing_pinned_key_units
tests/unit/test_backend_formula.py::test_simon_native_output_matches_existing_recorder_bytes
tests/unit/test_backend_formula.py::test_simon_conditioned_output_matches_existing_assumption_serializer
tests/unit/test_backend_formula.py::test_simon_builder_never_constructs_solver
tests/unit/test_backend_formula.py::test_native_only_build_skips_pure_cnf_conversion
tests/unit/test_backend_formula.py::test_renderer_preserves_signed_rhs_and_empty_xor_semantics
tests/unit/test_backend_formula.py::test_long_xor_conversion_allocates_fresh_auxiliaries_repeatably
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[0-0-fields0]
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[3-0-fields1]
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[6-0-fields2]
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[2--1-fields3]
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[2-5-fields4]
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[2-1-fields5]
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[2-1-fields6]
tests/unit/test_backend_formula.py::test_invalid_rounds_and_observation_words_fail_closed
tests/unit/test_backend_formula.py::test_repeated_builds_are_deterministic_and_do_not_mutate_inputs
tests/unit/test_benchmark_formula_export.py::test_benchmark_formula_manifest_integrity
tests/unit/test_benchmark_formula_export.py::test_speck_subcritical_formula_solvable_by_cadical
tests/unit/test_dimacs.py::test_dimacs_export_and_solve_recovers_master_key
tests/unit/test_dimacs.py::test_dimacs_export_with_pinned_round_key
tests/unit/test_pure_cnf.py::test_speck_pure_cnf_export_solved_by_cadical_and_cms
tests/unit/test_pure_cnf.py::test_speck_pure_cnf_export_with_pinned_keys
tests/unit/test_pure_cnf_adversarial.py::test_empty_contradiction_survives_conversion[0]
tests/unit/test_pure_cnf_adversarial.py::test_empty_contradiction_survives_conversion[x 0]
tests/unit/test_pure_cnf_adversarial.py::test_empty_contradiction_survives_conversion[x0]
tests/unit/test_pure_cnf_adversarial.py::test_empty_formula_is_tautology
tests/unit/test_pure_cnf_adversarial.py::test_generic_empty_xor_preserves_both_rhs_values[False]
tests/unit/test_pure_cnf_adversarial.py::test_generic_empty_xor_preserves_both_rhs_values[True]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals0]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals1]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals2]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals3]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals4]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals5]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals6]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals7]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals8]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals0]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals1]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals2]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals3]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals4]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals5]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals6]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals7]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals8]
tests/unit/test_pure_cnf_adversarial.py::test_multiple_long_xors_use_disjoint_fresh_auxiliaries
tests/unit/test_pure_cnf_adversarial.py::test_declared_unused_variables_are_reserved
tests/unit/test_pure_cnf_adversarial.py::test_wrapped_and_multiple_records_preserve_semantics
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[1 0\n-1-header]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[-0-header]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\np cnf 1 1\n1 0\n-1-header]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 0\n1 0\n-1-record count]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 2\n1 0\n-1-record count]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\n1\n-1-unterminated]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\nx 1\n-1-unterminated]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\nx\n-1-unterminated]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 2\n1 x 1 0\n-1-unterminated]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\n2 0\n-1-range]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\nx -2 0\n-1-range]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 6 1\nx 1 2 3 4 5 6 0\n-5-match]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf -1 0\n-0-negative]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 -1\n-1-negative]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf one 0\n-1-noninteger]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1_0 0\n-10-noninteger]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\n\u0661 0\n-1-noninteger]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 10 1\n1_0 0\n-10-noninteger]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 0 trailing\n-1-expected header]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p other 1 0\n-1-expected header]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\nword 0\n-1-noninteger]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\nx 1 0 0\n-1-record count]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\n1 0 1\n-1-unterminated]
tests/unit/test_pure_cnf_adversarial.py::test_invalid_supplied_variable_count_rejected[True]
tests/unit/test_pure_cnf_adversarial.py::test_invalid_supplied_variable_count_rejected[False]
tests/unit/test_pure_cnf_adversarial.py::test_invalid_supplied_variable_count_rejected[-1]
tests/unit/test_pure_cnf_adversarial.py::test_invalid_supplied_variable_count_rejected[1.0]
tests/unit/test_pure_cnf_adversarial.py::test_invalid_supplied_variable_count_rejected[1]
tests/unit/test_pure_cnf_adversarial.py::test_invalid_supplied_variable_count_rejected[None]
tests/unit/test_sat_encoding.py::test_four_bit_modular_addition_truth_table
tests/unit/test_sat_encoding.py::test_native_xor_encoder_matches_truth_table
tests/unit/test_simon_sat.py::test_simon_sat_model_matches_reference_cipher_on_kat[1]
tests/unit/test_simon_sat.py::test_simon_sat_model_matches_reference_cipher_on_kat[2]
tests/unit/test_simon_sat.py::test_simon_sat_model_matches_reference_cipher_on_kat[4]
tests/unit/test_simon_sat.py::test_simon_sat_model_matches_reference_cipher_on_kat[6]
tests/unit/test_simon_sat.py::test_simon_sat_model_matches_reference_cipher_on_kat[8]
tests/unit/test_simon_sat.py::test_simon_sat_exact_master_key_recovery_with_trailing_assumptions
tests/unit/test_simon_sat.py::test_solver_isolation_between_models
tests/unit/test_simon_sat.py::test_conflicting_assumptions_yield_unsat
tests/unit/test_simon_sat.py::test_timeout_state_classification
tests/unit/test_simon_sat.py::test_master_key_verification_fails_closed
tests/unit/test_simon_sat.py::test_dimacs_export_and_native_cli_solver
tests/unit/test_simon_sat.py::test_simon_pure_cnf_export_solved_by_cadical_and_cms
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[False-1]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[False-2]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[False-3]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[False-4]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[False-5]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[True-1]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[True-2]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[True-3]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[True-4]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[True-5]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[False-4]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[False-5]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[False-6]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[False-7]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[True-4]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[True-5]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[True-6]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[True-7]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_edge_cases
tests/unit/test_xor_to_cnf.py::test_convert_xor_clauses_to_cnf

119 tests collected in 10.63s

Exit code: 0
Elapsed seconds: 13.312

```

## solver isolation

- `tests/unit/test_backend_solver_io.py`
- `tests/unit/test_conditioning_worker.py`
- `tests/unit/test_queue_order_worker.py`
- `tests/unit/test_solver_isolation.py`

```text
COMMAND: <work>/hamza-venv/Scripts/python.exe -m pytest --collect-only -q tests/unit/test_backend_solver_io.py tests/unit/test_conditioning_worker.py tests/unit/test_queue_order_worker.py tests/unit/test_solver_isolation.py
tests/unit/test_backend_solver_io.py::test_native_gauss_profiles_differ_only_by_matrix_limit
tests/unit/test_backend_solver_io.py::test_native_default_keeps_cms_defaults_and_pins_one_thread
tests/unit/test_backend_solver_io.py::test_cnf_profile_changes_only_input_representation_from_x
tests/unit/test_backend_solver_io.py::test_cadical_uses_quiet_mode_and_preserves_path_as_one_argument
tests/unit/test_backend_solver_io.py::test_solver_arguments_reject_unknown_profiles_and_missing_paths[unknown-/bin/cms-/bin/cadical-formula.cnf]
tests/unit/test_backend_solver_io.py::test_solver_arguments_reject_unknown_profiles_and_missing_paths[cms_native_default--/bin/cadical-formula.cnf]
tests/unit/test_backend_solver_io.py::test_solver_arguments_reject_unknown_profiles_and_missing_paths[cadical_cnf-/bin/cms--formula.cnf]
tests/unit/test_backend_solver_io.py::test_solver_arguments_reject_unknown_profiles_and_missing_paths[cms_native_default-/bin/cms-/bin/cadical-]
tests/unit/test_backend_solver_io.py::test_sat_model_can_span_lines_and_returns_raw_gauss_diagnostics
tests/unit/test_backend_solver_io.py::test_cadical_signed_dimacs_model_and_comments
tests/unit/test_backend_solver_io.py::test_empty_unsat_result_has_empty_assignment[cms]
tests/unit/test_backend_solver_io.py::test_empty_unsat_result_has_empty_assignment[cadical]
tests/unit/test_backend_solver_io.py::test_unknown_outcomes[cms-15-s INDETERMINATE\n]
tests/unit/test_backend_solver_io.py::test_unknown_outcomes[cadical-0-s UNKNOWN\n]
tests/unit/test_backend_solver_io.py::test_unknown_needs_real_exit_semantics_and_explicit_output[cms-0-s UNKNOWN\n]
tests/unit/test_backend_solver_io.py::test_unknown_needs_real_exit_semantics_and_explicit_output[cadical-0-]
tests/unit/test_backend_solver_io.py::test_exit_code_status_disagreement_is_error
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-v 1 0\n-required0-1-no explicit solver status]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\ns SATISFIABLE\nv 1 0\n-required1-1-repeated]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 1 0 2\n-required2-2-after terminal zero]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 1 0\nv -1 0\n-required3-1-after terminal zero]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 1 -1 0\n-required4-1-conflicting]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 1 1_0 0\n-required5-10-non-integer]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 4 0\n-required6-3-outside]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 1 -2\n-required7-2-truncated]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 1 0\n-required8-2-omits required]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[20-s UNSATISFIABLE\nv 0\n-required9-0-UNSAT result unexpectedly]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[0-s UNKNOWN\nv 1 0\n-required10-1-UNKNOWN result unexpectedly]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE extra\nv 1 0\n-required11-1-malformed status]
tests/unit/test_backend_solver_io.py::test_invalid_requested_variable_sets_are_rejected[required0-2]
tests/unit/test_backend_solver_io.py::test_invalid_requested_variable_sets_are_rejected[required1-2]
tests/unit/test_backend_solver_io.py::test_invalid_requested_variable_sets_are_rejected[required2-2]
tests/unit/test_backend_solver_io.py::test_invalid_requested_variable_sets_are_rejected[required3-2]
tests/unit/test_backend_solver_io.py::test_invalid_requested_variable_sets_are_rejected[required4--1]
tests/unit/test_backend_solver_io.py::test_unknown_engine_and_unexpected_exit_code_are_errors
tests/unit/test_conditioning_worker.py::test_production_solver_factory_uses_the_fixed_wrapper_options
tests/unit/test_conditioning_worker.py::test_worker_loads_identical_formula_and_changes_only_conditioning_interface[permanent_units-speck32_64]
tests/unit/test_conditioning_worker.py::test_worker_loads_identical_formula_and_changes_only_conditioning_interface[permanent_units-simon32_64]
tests/unit/test_conditioning_worker.py::test_worker_loads_identical_formula_and_changes_only_conditioning_interface[temporary_assumptions-speck32_64]
tests/unit/test_conditioning_worker.py::test_worker_loads_identical_formula_and_changes_only_conditioning_interface[temporary_assumptions-simon32_64]
tests/unit/test_conditioning_worker.py::test_temporary_assumptions_are_removed_and_permanent_units_persist
tests/unit/test_conditioning_worker.py::test_reference_known_check_uses_cipher_and_key_schedule[speck32_64]
tests/unit/test_conditioning_worker.py::test_reference_known_check_uses_cipher_and_key_schedule[simon32_64]
tests/unit/test_conditioning_worker.py::test_invalid_model_is_correctness_alarm_and_truth_is_not_read
tests/unit/test_conditioning_worker.py::test_known_check_failure_stops_before_truth_load
tests/unit/test_conditioning_worker.py::test_valid_sat_model_without_fresh_match_is_retained_as_nonunique_outcome
tests/unit/test_conditioning_worker.py::test_truth_hash_is_checked_only_after_known_model_check
tests/unit/test_conditioning_worker.py::test_true_key_unsat_is_a_distinct_correctness_alarm
tests/unit/test_conditioning_worker.py::test_environment_rejection_keeps_initial_telemetry_and_power_sample
tests/unit/test_conditioning_worker.py::test_thread_environment_mismatch_stops_before_solver_certification
tests/unit/test_conditioning_worker.py::test_invalid_fixed_cell_is_rejected_before_environment_or_solver[rounds-True]
tests/unit/test_conditioning_worker.py::test_invalid_fixed_cell_is_rejected_before_environment_or_solver[key_index-8]
tests/unit/test_conditioning_worker.py::test_invalid_fixed_cell_is_rejected_before_environment_or_solver[observations-2]
tests/unit/test_conditioning_worker.py::test_invalid_fixed_cell_is_rejected_before_environment_or_solver[trailing_keys-0]
tests/unit/test_conditioning_worker.py::test_invalid_fixed_cell_is_rejected_before_environment_or_solver[arm-unknown]
tests/unit/test_conditioning_worker.py::test_main_reads_checked_payload_and_atomically_persists_result
tests/unit/test_queue_order_worker.py::test_traversal_reuses_model_after_unsat_but_starts_a_fresh_model_per_arm
tests/unit/test_queue_order_worker.py::test_unresolved_status_rebuilds_before_next_candidate[solver_timeout]
tests/unit/test_queue_order_worker.py::test_unresolved_status_rebuilds_before_next_candidate[unsat_or_timeout]
tests/unit/test_queue_order_worker.py::test_verification_failure_rebuilds_and_is_not_reported_as_recovery
tests/unit/test_queue_order_worker.py::test_traverse_rejects_wrong_observation_counts_before_model_construction[127-2]
tests/unit/test_queue_order_worker.py::test_traverse_rejects_wrong_observation_counts_before_model_construction[128-1]
tests/unit/test_queue_order_worker.py::test_new_model_uses_only_the_leading_four_known_observations[speck32_64]
tests/unit/test_queue_order_worker.py::test_new_model_uses_only_the_leading_four_known_observations[simon32_64]
tests/unit/test_queue_order_worker.py::test_traverse_builds_chronological_round_key_assumptions[speck32_64]
tests/unit/test_queue_order_worker.py::test_traverse_builds_chronological_round_key_assumptions[simon32_64]
tests/unit/test_queue_order_worker.py::test_four_round_cipher_sat_trailing_assumptions_and_reference_verification[speck32_64]
tests/unit/test_queue_order_worker.py::test_four_round_cipher_sat_trailing_assumptions_and_reference_verification[simon32_64]
tests/unit/test_solver_isolation.py::test_every_compared_method_receives_distinct_object_identity
tests/unit/test_solver_isolation.py::test_reordering_method_execution_does_not_alter_results
tests/unit/test_solver_isolation.py::test_candidate_assumptions_remain_incremental_within_one_method
tests/unit/test_solver_isolation.py::test_solver_state_cannot_be_passed_across_methods
tests/unit/test_solver_isolation.py::test_resume_logic_recreates_method_local_solver_state
tests/unit/test_solver_isolation.py::test_seed_role_validation_rejects_quarantined_seed_97001
tests/unit/test_solver_isolation.py::test_seed_role_validation_rejects_retired_seed_99001
tests/unit/test_solver_isolation.py::test_cli_default_contains_no_protected_research_seed
tests/unit/test_solver_isolation.py::test_historical_seed_98001_rejected_for_model_selection
tests/unit/test_solver_isolation.py::test_seed_role_validation_accepts_owner_seeds_for_their_roles
tests/unit/test_solver_isolation.py::test_seed_role_validation_rejects_cross_role_contamination

79 tests collected in 9.79s

Exit code: 0
Elapsed seconds: 12.375

```

## ranking and neural

- `tests/unit/test_backend_ranking_depth_figure.py`
- `tests/unit/test_backend_ranking_depth_n32_figure.py`
- `tests/unit/test_beam_adjudication.py`
- `tests/unit/test_confidence.py`
- `tests/unit/test_fallback_redesign.py`
- `tests/unit/test_g2_pipeline.py`
- `tests/unit/test_hierarchical_multi_round.py`
- `tests/unit/test_hierarchical_ranking.py`
- `tests/unit/test_multi_round_recovery.py`
- `tests/unit/test_neural_dataset.py`
- `tests/unit/test_neural_model.py`
- `tests/unit/test_neural_train.py`
- `tests/unit/test_numpy_inference.py`
- `tests/unit/test_pair_group_data.py`
- `tests/unit/test_pair_group_model.py`
- `tests/unit/test_pair_group_ranking.py`
- `tests/unit/test_phase10_vtc.py`
- `tests/unit/test_phase11b_prime_pair_group.py`
- `tests/unit/test_phase6c_calibration.py`
- `tests/unit/test_phase6c_fallback.py`
- `tests/unit/test_queue_order_ranking.py`
- `tests/unit/test_ranking.py`
- `tests/unit/test_ranking_depth_ranking.py`
- `tests/unit/test_screened_group_ranking.py`
- `tests/unit/test_seven_round_comparators.py`
- `tests/unit/test_simon_b1_data.py`
- `tests/unit/test_simon_b1_training_runner.py`
- `tests/unit/test_simon_classical_ranking.py`
- `tests/unit/test_simon_exact_ddt_equivalence.py`
- `tests/unit/test_simon_faithful_b1.py`
- `tests/unit/test_simon_predicted_ddt.py`
- `tests/unit/test_simon_published_vv_ranking.py`
- `tests/unit/test_simon_ranking_features.py`

```text
COMMAND: <work>/hamza-venv/Scripts/python.exe -m pytest --collect-only -q tests/unit/test_backend_ranking_depth_figure.py tests/unit/test_backend_ranking_depth_n32_figure.py tests/unit/test_beam_adjudication.py tests/unit/test_confidence.py tests/unit/test_fallback_redesign.py tests/unit/test_g2_pipeline.py tests/unit/test_hierarchical_multi_round.py tests/unit/test_hierarchical_ranking.py tests/unit/test_multi_round_recovery.py tests/unit/test_neural_dataset.py tests/unit/test_neural_model.py tests/unit/test_neural_train.py tests/unit/test_numpy_inference.py tests/unit/test_pair_group_data.py tests/unit/test_pair_group_model.py tests/unit/test_pair_group_ranking.py tests/unit/test_phase10_vtc.py tests/unit/test_phase11b_prime_pair_group.py tests/unit/test_phase6c_calibration.py tests/unit/test_phase6c_fallback.py tests/unit/test_queue_order_ranking.py tests/unit/test_ranking.py tests/unit/test_ranking_depth_ranking.py tests/unit/test_screened_group_ranking.py tests/unit/test_seven_round_comparators.py tests/unit/test_simon_b1_data.py tests/unit/test_simon_b1_training_runner.py tests/unit/test_simon_classical_ranking.py tests/unit/test_simon_exact_ddt_equivalence.py tests/unit/test_simon_faithful_b1.py tests/unit/test_simon_predicted_ddt.py tests/unit/test_simon_published_vv_ranking.py tests/unit/test_simon_ranking_features.py
tests/unit/test_backend_ranking_depth_figure.py::test_plot_totals_agree_with_independent_raw_audit
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[duplicate]
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[missing]
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[string_flag]
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[wrong_outcome]
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[late]
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[nan]
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[premature]
tests/unit/test_backend_ranking_depth_figure.py::test_corrupted_sidecar_rejected
tests/unit/test_backend_ranking_depth_n32_figure.py::test_canonical_source_hash_and_sidecar
tests/unit/test_backend_ranking_depth_n32_figure.py::test_n32_plot_totals_and_par2_agree_with_audit
tests/unit/test_backend_ranking_depth_n32_figure.py::test_containment_invariants
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[duplicate_cell]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[missing_cell]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[string_flag]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[wrong_outcome_recovered_but_false]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[wrong_outcome_unrecovered_but_true]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[unexpected_unrecovered_outcome]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[late_recovery_over_300s]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[premature_deadline_under_300s]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[nan_wall_time]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[negative_wall_time]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[unmeasured_containment]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[containment_key_mismatch]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[wrong_cell_count]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_corrupted_sidecar_or_hash_rejected
tests/unit/test_backend_ranking_depth_n32_figure.py::test_generate_end_to_end_in_tmp_path
tests/unit/test_beam_adjudication.py::test_seed_governance_prohibitions
tests/unit/test_beam_adjudication.py::test_draw_csprng_seed_returns_valid_seed
tests/unit/test_beam_adjudication.py::test_configuration_parameter_invariants
tests/unit/test_beam_adjudication.py::test_counterbalancing_schedule
tests/unit/test_beam_adjudication.py::test_selection_rule_tier1_higher_recovery_wins
tests/unit/test_beam_adjudication.py::test_selection_rule_tier2_lower_par2_wins_on_tie
tests/unit/test_beam_adjudication.py::test_selection_rule_tier4_reference_a_retained_on_complete_tie
tests/unit/test_beam_adjudication.py::test_exact_mcnemar_on_discordant_pairs
tests/unit/test_beam_adjudication.py::test_mismatched_instance_sets_rejected
tests/unit/test_confidence.py::test_features_use_ordered_scores_and_have_expected_values
tests/unit/test_confidence.py::test_serialized_features_and_target_membership_are_deterministic
tests/unit/test_confidence.py::test_pair_subset_stability_is_exposed_when_present
tests/unit/test_confidence.py::test_tied_scores_keep_the_supplied_deterministic_order
tests/unit/test_confidence.py::test_malformed_rankings_are_rejected[candidate_values0-candidate_scores0-candidate_order0]
tests/unit/test_confidence.py::test_malformed_rankings_are_rejected[candidate_values1-candidate_scores1-candidate_order1]
tests/unit/test_confidence.py::test_malformed_rankings_are_rejected[candidate_values2-candidate_scores2-candidate_order2]
tests/unit/test_confidence.py::test_malformed_rankings_are_rejected[candidate_values3-candidate_scores3-candidate_order3]
tests/unit/test_confidence.py::test_malformed_rankings_are_rejected[candidate_values4-candidate_scores4-candidate_order4]
tests/unit/test_confidence.py::test_invalid_top_k_is_rejected[0]
tests/unit/test_confidence.py::test_invalid_top_k_is_rejected[1]
tests/unit/test_confidence.py::test_invalid_top_k_is_rejected[5]
tests/unit/test_confidence.py::test_invalid_top_k_is_rejected[True]
tests/unit/test_confidence.py::test_invalid_top_k_is_rejected[2.0]
tests/unit/test_confidence.py::test_risk_coverage_is_validation_only_and_deterministic
tests/unit/test_confidence.py::test_extended_risk_coverage_requires_stability_data
tests/unit/test_confidence.py::test_risk_coverage_rejects_unknown_or_empty_inputs
tests/unit/test_fallback_redesign.py::test_invariant_1_g1_feature_extraction_is_deterministic
tests/unit/test_fallback_redesign.py::test_invariant_2_g1_handles_degenerate_distributions_gracefully
tests/unit/test_fallback_redesign.py::test_invariant_3_normalized_logit_margin_matches_cattelan_silva
tests/unit/test_fallback_redesign.py::test_invariant_4_r1_hard_negative_miner_labels_correctly
tests/unit/test_fallback_redesign.py::test_invariant_5_r1_shortlist_builder_preserves_true_subkey
tests/unit/test_fallback_redesign.py::test_invariant_6_rescue_ceiling_bounded_by_top_m_recall
tests/unit/test_fallback_redesign.py::test_invariant_7_cascade_routes_to_b1_when_reliability_high
tests/unit/test_fallback_redesign.py::test_invariant_8_cascade_routes_to_r1_when_reliability_low
tests/unit/test_fallback_redesign.py::test_invariant_9_lazy_b0_never_computed_unless_primary_fails
tests/unit/test_fallback_redesign.py::test_invariant_10_solver_isolation_maintained_in_cascade
tests/unit/test_fallback_redesign.py::test_invariant_11_state_machine_tracks_cumulative_time
tests/unit/test_fallback_redesign.py::test_invariant_12_seed_validation_rejects_quarantined_seeds
tests/unit/test_fallback_redesign.py::test_invariant_13_locked_validation_role_cannot_enter_training_or_calibration
tests/unit/test_fallback_redesign.py::test_invariant_14_g1_optimized_reference_equivalence
tests/unit/test_fallback_redesign.py::test_invariant_15_r1_shortlist_budget_selection
tests/unit/test_fallback_redesign.py::test_invariant_16_terminal_fallback_deduplication_and_cumulative_budget
tests/unit/test_fallback_redesign.py::test_invariant_17_g1_failure_risk_routing_polarity
tests/unit/test_g2_pipeline.py::test_partition_masks_properties
tests/unit/test_g2_pipeline.py::test_distributional_features_shift_invariance
tests/unit/test_g2_pipeline.py::test_stability_plus_features
tests/unit/test_g2_pipeline.py::test_g2_feature_vector_extraction
tests/unit/test_g2_pipeline.py::test_g2_models_train_and_predict
tests/unit/test_g2_pipeline.py::test_distributional_exact_arithmetic
tests/unit/test_g2_pipeline.py::test_stability_plus_exact_arithmetic
tests/unit/test_hierarchical_multi_round.py::test_hierarchical_stage_rejects_invalid_arguments
tests/unit/test_hierarchical_multi_round.py::test_attack_rejects_invalid_arguments
tests/unit/test_hierarchical_multi_round.py::test_single_stage_attack_recovers_five_round_instance_key
tests/unit/test_hierarchical_multi_round.py::test_two_stage_attack_refutes_a_wrong_pair_before_verifying
tests/unit/test_hierarchical_multi_round.py::test_identical_runs_produce_identical_search_and_order
tests/unit/test_hierarchical_multi_round.py::test_beam_width_bounds_the_number_of_tested_tuples
tests/unit/test_hierarchical_multi_round.py::test_pruned_first_stage_candidate_reports_ranking_failure
tests/unit/test_hierarchical_multi_round.py::test_pruning_inside_the_final_stage_reports_ranking_failure
tests/unit/test_hierarchical_multi_round.py::test_search_does_not_read_declared_ground_truth
tests/unit/test_hierarchical_multi_round.py::test_frozen_b1_checkpoint_satisfies_the_stage_interface
tests/unit/test_hierarchical_multi_round.py::test_screened_hierarchical_stage_validation_and_execution
tests/unit/test_hierarchical_multi_round.py::test_stage_cache_and_excluded_paths_cascade
tests/unit/test_hierarchical_ranking.py::test_h1_preserves_b1_window_set_and_tail
tests/unit/test_hierarchical_ranking.py::test_h1_b0_ties_use_ascending_candidate_value
tests/unit/test_hierarchical_ranking.py::test_h1_invokes_b0_only_for_the_sixteen_candidate_window
tests/unit/test_hierarchical_ranking.py::test_h1_uses_existing_b0_sum_logits_sorting
tests/unit/test_hierarchical_ranking.py::test_h1_scalar_and_batched_window_scores_match
tests/unit/test_hierarchical_ranking.py::test_h1_rerun_is_deterministic
tests/unit/test_hierarchical_ranking.py::test_h1_serialization_round_trip
tests/unit/test_hierarchical_ranking.py::test_h1_deserialization_rejects_order_inconsistent_with_b0_scores
tests/unit/test_hierarchical_ranking.py::test_h1_deserialization_rejects_corrupt_protocol_metadata[tie_rule]
tests/unit/test_hierarchical_ranking.py::test_h1_deserialization_rejects_corrupt_protocol_metadata[candidate_domain]
tests/unit/test_hierarchical_ranking.py::test_h1_timing_fields_are_finite_and_non_negative
tests/unit/test_hierarchical_ranking.py::test_h1_ordering_api_does_not_accept_or_serialize_true_key
tests/unit/test_hierarchical_ranking.py::test_h1_rejects_incomplete_or_duplicate_b1_order
tests/unit/test_hierarchical_ranking.py::test_h1_rejects_unfrozen_budget_and_invalid_batch_size[kwargs0-window_size]
tests/unit/test_hierarchical_ranking.py::test_h1_rejects_unfrozen_budget_and_invalid_batch_size[kwargs1-b0_candidate_batch_size]
tests/unit/test_hierarchical_ranking.py::test_h1_rejects_wrong_instance_pair_count
tests/unit/test_hierarchical_ranking.py::test_h1_rejects_wrong_ranking_type
tests/unit/test_hierarchical_ranking.py::test_h1_rejects_non_model_b0_input
tests/unit/test_multi_round_recovery.py::test_assumptions_for_round_key_pins_every_bit
tests/unit/test_multi_round_recovery.py::test_assumptions_reject_invalid_inputs
tests/unit/test_multi_round_recovery.py::test_assumptions_for_round_keys_are_order_independent
tests/unit/test_multi_round_recovery.py::test_trailing_round_key_indices
tests/unit/test_multi_round_recovery.py::test_multi_round_recovery_verifies_true_candidate
tests/unit/test_multi_round_recovery.py::test_multi_round_recovery_skips_wrong_candidate
tests/unit/test_multi_round_recovery.py::test_multi_round_recovery_reports_exhausted_candidates
tests/unit/test_multi_round_recovery.py::test_partial_decrypt_instance_matches_shorter_encryption
tests/unit/test_multi_round_recovery.py::test_partial_decrypt_instance_rejects_invalid_inputs
tests/unit/test_multi_round_recovery.py::test_multi_round_recovery_requires_uniform_candidate_width
tests/unit/test_multi_round_recovery.py::test_seven_round_recovery_needs_two_assumed_round_keys
tests/unit/test_neural_dataset.py::test_features_to_tensor_reshapes_64_bits_to_4_words
tests/unit/test_neural_dataset.py::test_dataset_from_fixture_json
tests/unit/test_neural_dataset.py::test_features_to_tensor_rejects_nonbinary_values
tests/unit/test_neural_dataset.py::test_classifier_splits_load_and_verify_fixture
tests/unit/test_neural_dataset.py::test_dataset_from_generated_in_memory
tests/unit/test_neural_model.py::test_gohr_resnet_forward_shapes
tests/unit/test_neural_model.py::test_gohr_resnet_probabilities_in_range
tests/unit/test_neural_model.py::test_predict_proba_supports_one_sample_without_changing_training_mode
tests/unit/test_neural_model.py::test_single_sample_training_batch_is_supported
tests/unit/test_neural_model.py::test_invalid_resnet_configuration_is_rejected
tests/unit/test_neural_model.py::test_batched_vs_unbatched_inference_equivalence
tests/unit/test_neural_model.py::test_gradient_flows_through_all_parameters
tests/unit/test_neural_train.py::test_roc_auc_metric_perfect_and_random
tests/unit/test_neural_train.py::test_metrics_reject_nonbinary_labels_and_invalid_probabilities
tests/unit/test_neural_train.py::test_train_distinguisher_runs_and_improves_loss
tests/unit/test_neural_train.py::test_seeded_model_training_is_reproducible
tests/unit/test_neural_train.py::test_save_and_load_checkpoint
tests/unit/test_numpy_inference.py::test_numpy_pair_group_outputs_match_pytorch[b1_mean]
tests/unit/test_numpy_inference.py::test_numpy_pair_group_outputs_match_pytorch[b2_moments]
tests/unit/test_numpy_inference.py::test_numpy_pair_group_outputs_match_pytorch[b3_weighted]
tests/unit/test_numpy_inference.py::test_numpy_b0_outputs_match_pytorch
tests/unit/test_numpy_inference.py::test_numpy_adapter_returns_torch_logits
tests/unit/test_numpy_inference.py::test_numpy_backend_rejects_wrong_dtype
tests/unit/test_numpy_inference.py::test_numpy_backend_rejects_wrong_group_shape
tests/unit/test_numpy_inference.py::test_vectorized_final_round_features_match_scalar_reference
tests/unit/test_pair_group_data.py::test_pair_group_generation_is_deterministic_and_typed
tests/unit/test_pair_group_data.py::test_different_pair_group_seed_changes_content
tests/unit/test_pair_group_data.py::test_group_keys_and_positive_negative_construction_are_correct
tests/unit/test_pair_group_data.py::test_group_split_keys_are_disjoint
tests/unit/test_pair_group_data.py::test_pair_group_export_load_and_manifest_verification
tests/unit/test_pair_group_data.py::test_phase6a_attack_export_is_validation_only
tests/unit/test_pair_group_model.py::test_pair_group_forward_shapes_and_permutation_invariance[b1_mean]
tests/unit/test_pair_group_model.py::test_pair_group_forward_shapes_and_permutation_invariance[b2_moments]
tests/unit/test_pair_group_model.py::test_pair_group_forward_shapes_and_permutation_invariance[b3_weighted]
tests/unit/test_pair_group_model.py::test_pair_group_initialization_is_deterministic_and_transfers_b0_encoder
tests/unit/test_pair_group_model.py::test_pair_group_rejects_wrong_dtype_and_shape
tests/unit/test_pair_group_model.py::test_pair_group_gradients_are_finite
tests/unit/test_pair_group_model.py::test_pair_group_checkpoint_round_trip
tests/unit/test_pair_group_model.py::test_pair_group_training_records_finite_validation_bce
tests/unit/test_pair_group_ranking.py::test_pair_group_candidate_batching_and_mode_restore
tests/unit/test_pair_group_ranking.py::test_pair_group_ties_use_ascending_candidate_value
tests/unit/test_pair_group_ranking.py::test_numpy_pair_group_ranking_matches_reference_on_small_domain
tests/unit/test_phase10_vtc.py::test_solver_isolation_invariant
tests/unit/test_phase10_vtc.py::test_vtc_early_termination_on_b1_success
tests/unit/test_phase10_vtc.py::test_vtc_escalation_on_b1_exhaustion
tests/unit/test_phase10_vtc.py::test_vtc_candidate_deduplication
tests/unit/test_phase10_vtc.py::test_vtc_budget_capping
tests/unit/test_phase10_vtc.py::test_vtc_serialization_roundtrip
tests/unit/test_phase11b_prime_pair_group.py::test_six_round_generation_is_deterministic
tests/unit/test_phase11b_prime_pair_group.py::test_six_round_metadata_records_six_rounds_and_owned_dataset_id
tests/unit/test_phase11b_prime_pair_group.py::test_six_round_features_use_six_round_encryption
tests/unit/test_phase11b_prime_pair_group.py::test_six_round_config_rejects_five_rounds
tests/unit/test_phase11b_prime_pair_group.py::test_frozen_five_round_config_still_rejects_six_rounds
tests/unit/test_phase11b_prime_pair_group.py::test_frozen_five_round_identifiers_are_unchanged
tests/unit/test_phase11b_prime_pair_group.py::test_six_round_export_manifest_and_shard_round_trip
tests/unit/test_phase11b_prime_pair_group.py::test_five_round_path_rejects_six_round_artefacts
tests/unit/test_phase11b_prime_pair_group.py::test_five_round_export_cannot_emit_six_round_configurations
tests/unit/test_phase11b_prime_pair_group.py::test_stage1_seed_role_accepts_its_own_seeds
tests/unit/test_phase11b_prime_pair_group.py::test_stage1_seed_role_rejects_history_and_other_protected_seeds
tests/unit/test_phase11b_prime_pair_group.py::test_protected_stage1_seeds_are_rejected_by_other_roles
tests/unit/test_phase6c_calibration.py::test_phase6c_feature_uses_ordered_top8_and_all_score_std
tests/unit/test_phase6c_calibration.py::test_phase6c_feature_has_zero_score_std_guard
tests/unit/test_phase6c_calibration.py::test_platt_fit_is_deterministic_bounded_and_serializable
tests/unit/test_phase6c_calibration.py::test_platt_fit_rejects_nonconvergence_and_single_class
tests/unit/test_phase6c_calibration.py::test_threshold_selection_is_unique_deterministic_and_fails_closed
tests/unit/test_phase6c_calibration.py::test_ece_boundaries_and_auc_tie_credit_are_explicit
tests/unit/test_phase6c_calibration.py::test_calibration_report_contains_prevalence_brier_and_risk_coverage
tests/unit/test_phase6c_calibration.py::test_bootstrap_mean_interval_is_seed_deterministic
tests/unit/test_phase6c_fallback.py::test_fallback_alternates_and_fills_exactly_32_unique_candidates
tests/unit/test_phase6c_fallback.py::test_fallback_first_occurrence_and_duplicate_fill_are_deterministic
tests/unit/test_phase6c_fallback.py::test_route_boundary_uses_trusted_route_at_threshold
tests/unit/test_phase6c_fallback.py::test_fallback_rejects_non_complete_inputs[b10-b00]
tests/unit/test_phase6c_fallback.py::test_fallback_rejects_non_complete_inputs[b11-b01]
tests/unit/test_phase6c_fallback.py::test_fallback_rejects_non_complete_inputs[b12-b02]
tests/unit/test_queue_order_ranking.py::test_speck_composition_matches_original_one_based_stage_ties
tests/unit/test_queue_order_ranking.py::test_simon_composition_matches_original_zero_based_numpy_orders
tests/unit/test_queue_order_ranking.py::test_composers_reject_malformed_prefixes[stage10-stage20]
tests/unit/test_queue_order_ranking.py::test_composers_reject_malformed_prefixes[stage11-stage21]
tests/unit/test_queue_order_ranking.py::test_composers_reject_malformed_prefixes[stage12-stage22]
tests/unit/test_queue_order_ranking.py::test_composers_reject_malformed_prefixes[stage13-stage23]
tests/unit/test_queue_order_ranking.py::test_speck_rank_queue_uses_frozen_stage_calls_and_public_only_instance
tests/unit/test_queue_order_ranking.py::test_speck_ranking_fails_closed_without_mps
tests/unit/test_queue_order_ranking.py::test_simon_classical_uses_fixed_public_ranking_settings_and_numpy_orders
tests/unit/test_queue_order_ranking.py::test_simon_bao_path_uses_existing_pinned_subprocess_builder
tests/unit/test_queue_order_ranking.py::test_rank_queue_rejects_truth_fields_and_policy_drift_before_dispatch
tests/unit/test_ranking.py::test_true_candidate_maps_to_the_expected_pre_final_state
tests/unit/test_ranking.py::test_correct_candidate_ranks_first_and_model_mode_is_restored
tests/unit/test_ranking.py::test_candidate_batching_produces_identical_scores_and_order
tests/unit/test_ranking.py::test_score_ties_use_ascending_candidate_value
tests/unit/test_ranking.py::test_invalid_candidate_sets_are_rejected[candidate_values0]
tests/unit/test_ranking.py::test_invalid_candidate_sets_are_rejected[candidate_values1]
tests/unit/test_ranking.py::test_invalid_candidate_sets_are_rejected[candidate_values2]
tests/unit/test_ranking.py::test_invalid_candidate_sets_are_rejected[candidate_values3]
tests/unit/test_ranking.py::test_invalid_candidate_sets_are_rejected[candidate_values4]
tests/unit/test_ranking.py::test_empty_instance_is_rejected
tests/unit/test_ranking.py::test_model_must_return_one_logit_per_row
tests/unit/test_ranking.py::test_default_candidate_set_is_the_complete_16_bit_domain
tests/unit/test_ranking.py::test_evaluation_reports_missing_true_candidate_without_guessing
tests/unit/test_ranking.py::test_pair_subset_stability_reuses_per_pair_logits
tests/unit/test_ranking.py::test_invalid_pair_subset_stability_is_rejected[pair_subsets0]
tests/unit/test_ranking.py::test_invalid_pair_subset_stability_is_rejected[pair_subsets1]
tests/unit/test_ranking.py::test_invalid_pair_subset_stability_is_rejected[pair_subsets2]
tests/unit/test_ranking.py::test_invalid_pair_subset_stability_is_rejected[pair_subsets3]
tests/unit/test_ranking.py::test_invalid_pair_subset_stability_is_rejected[pair_subsets4]
tests/unit/test_ranking.py::test_invalid_pair_subset_stability_is_rejected[pair_subsets5]
tests/unit/test_ranking_depth_ranking.py::test_speck_stage1_spy_receives_only_ciphertext_and_fixed_mps_batch
tests/unit/test_ranking_depth_ranking.py::test_speck_sparse_r5_stage2_uses_all_fixed_parents_and_existing_queue_order
tests/unit/test_ranking_depth_ranking.py::test_one_and_two_key_simon_queues_share_persisted_stage1_prefix
tests/unit/test_ranking_depth_ranking.py::test_rejects_nonfrozen_trailing_key_counts[0]
tests/unit/test_ranking_depth_ranking.py::test_rejects_nonfrozen_trailing_key_counts[3]
tests/unit/test_ranking_depth_ranking.py::test_rejects_nonfrozen_trailing_key_counts[True]
tests/unit/test_ranking_depth_ranking.py::test_rejects_nonfrozen_trailing_key_counts[1.0]
tests/unit/test_ranking_depth_ranking.py::test_accepts_only_exact_public_ciphertext_pair_shape[bad_pairs0]
tests/unit/test_ranking_depth_ranking.py::test_accepts_only_exact_public_ciphertext_pair_shape[bad_pairs1]
tests/unit/test_ranking_depth_ranking.py::test_accepts_only_exact_public_ciphertext_pair_shape[bad_pairs2]
tests/unit/test_ranking_depth_ranking.py::test_accepts_only_exact_public_ciphertext_pair_shape[bad_pairs3]
tests/unit/test_ranking_depth_ranking.py::test_rejects_bool_and_duplicate_rank_words
tests/unit/test_ranking_depth_ranking.py::test_tf_child_profile_is_stage1_only_and_has_no_truth_inputs
tests/unit/test_ranking_depth_ranking.py::test_tf_import_boundary_bypasses_simon_init_and_blocks_sat
tests/unit/test_ranking_depth_ranking.py::test_source_receipts_hash_project_package_code_not_runtime_packages
tests/unit/test_screened_group_ranking.py::test_screened_ranking_invariants
tests/unit/test_screened_group_ranking.py::test_screened_ranking_validations
tests/unit/test_seven_round_comparators.py::test_stage_model_validation
tests/unit/test_seven_round_comparators.py::test_classical_stage_conditional_scoring
tests/unit/test_seven_round_comparators.py::test_truth_free_invariant
tests/unit/test_seven_round_comparators.py::test_global_timeout_during_ranking
tests/unit/test_seven_round_comparators.py::test_global_timeout_in_multi_round_recovery
tests/unit/test_seven_round_comparators.py::test_random_multi_round_candidates
tests/unit/test_seven_round_comparators.py::test_run_random_multi_round_recovery
tests/unit/test_seven_round_comparators.py::test_run_multi_round_sat_only_recovery
tests/unit/test_seven_round_comparators.py::test_solver_isolation_across_comparators
tests/unit/test_seven_round_comparators.py::test_resume_preserves_unselected_methods_behavioral_main
tests/unit/test_seven_round_comparators.py::test_resume_without_output_fails_loudly
tests/unit/test_seven_round_comparators.py::test_method_alias_normalization
tests/unit/test_seven_round_comparators.py::test_candidate_budget_defaults_to_m1_times_m2
tests/unit/test_seven_round_comparators.py::test_sat_only_respects_global_timeout
tests/unit/test_seven_round_comparators.py::test_recovery_continues_after_single_candidate_timeout
tests/unit/test_seven_round_comparators.py::test_solver_rebuilt_on_candidate_timeout
tests/unit/test_seven_round_comparators.py::test_distinct_outcome_categories
tests/unit/test_seven_round_comparators.py::test_resume_fail_closed_validation
tests/unit/test_seven_round_comparators.py::test_sat_only_observation_counts_and_formula_sizes
tests/unit/test_seven_round_comparators.py::test_model_construction_consumes_method_budget_and_clamps_solver_timeout
tests/unit/test_seven_round_comparators.py::test_method_budget_isolation_and_order_invariance
tests/unit/test_seven_round_comparators.py::test_solver_timeout_leaves_instance_unverified_with_cause
tests/unit/test_seven_round_comparators.py::test_runner_prospective_gate_fails_closed_on_proposed_manifest
tests/unit/test_seven_round_comparators.py::test_runner_flags_conflict_rejection
tests/unit/test_seven_round_comparators.py::test_runner_rebuilding_seconds_and_summary
tests/unit/test_seven_round_comparators.py::test_runner_non_interruptible_overshoot_reclassification
tests/unit/test_seven_round_comparators.py::test_runner_resume_validation_fail_closed
tests/unit/test_seven_round_comparators.py::test_runner_late_verified_recovery_reclassification
tests/unit/test_seven_round_comparators.py::test_zero_remaining_time_clean_exit
tests/unit/test_seven_round_comparators.py::test_verify_record_integrity_contradictions
tests/unit/test_seven_round_comparators.py::test_delayed_diagnostics_regression
tests/unit/test_seven_round_comparators.py::test_disjoint_timing_fields
tests/unit/test_seven_round_comparators.py::test_solver_exception_handling
tests/unit/test_seven_round_comparators.py::test_device_enforcement_prospective
tests/unit/test_seven_round_comparators.py::test_counterbalanced_method_order_cyclic
tests/unit/test_seven_round_comparators.py::test_resume_cohort_and_code_drift_validation
tests/unit/test_simon_b1_data.py::test_training_key_role_audit_rejects_duplicates_and_cross_role_overlap
tests/unit/test_simon_b1_data.py::test_fixed_9round_direct_pair_construction
tests/unit/test_simon_b1_data.py::test_group_master_key_sharing
tests/unit/test_simon_b1_data.py::test_strict_master_key_splitting
tests/unit/test_simon_b1_data.py::test_label_balance_and_permutation
tests/unit/test_simon_b1_data.py::test_deterministic_replay
tests/unit/test_simon_b1_data.py::test_feature_layout_and_bit_slicing
tests/unit/test_simon_b1_data.py::test_to_b0_data_flattening
tests/unit/test_simon_b1_data.py::test_pytorch_datasets_and_dataloaders
tests/unit/test_simon_b1_data.py::test_train_and_eval_step_b0
tests/unit/test_simon_b1_data.py::test_train_and_eval_step_b1
tests/unit/test_simon_b1_data.py::test_b0_weight_transfer_initialize_from_b0
tests/unit/test_simon_b1_data.py::test_checkpoint_save_and_load
tests/unit/test_simon_b1_data.py::test_synthetic_overfit_check_b1
tests/unit/test_simon_b1_data.py::test_synthetic_overfit_check_b0
tests/unit/test_simon_b1_data.py::test_proposed_training_manifest_specification
tests/unit/test_simon_b1_training_runner.py::test_proposed_manifest_loads_and_validates
tests/unit/test_simon_b1_training_runner.py::test_manifest_validation_rejects_missing_file_and_invalid_json
tests/unit/test_simon_b1_training_runner.py::test_manifest_validation_rejects_missing_required_keys
tests/unit/test_simon_b1_training_runner.py::test_manifest_validation_rejects_invalid_governing_decisions
tests/unit/test_simon_b1_training_runner.py::test_manifest_validation_rejects_invalid_architectures
tests/unit/test_simon_b1_training_runner.py::test_manifest_validation_rejects_forbidden_seeds
tests/unit/test_simon_b1_training_runner.py::test_manifest_validation_rejects_invalid_hyperparameters
tests/unit/test_simon_b1_training_runner.py::test_key_audit_rejects_duplicates_before_training
tests/unit/test_simon_b1_training_runner.py::test_key_audit_rejects_train_val_overlap_before_training
tests/unit/test_simon_b1_training_runner.py::test_preflight_cryptography_scalar_and_oracle_verification
tests/unit/test_simon_b1_training_runner.py::test_bounded_smoke_test_execution_and_checkpoint_reload
tests/unit/test_simon_b1_training_runner.py::test_deterministic_checkpoint_selection_on_tied_and_improving_losses
tests/unit/test_simon_b1_training_runner.py::test_deterministic_checkpoint_selection_in_b0_stage
tests/unit/test_simon_b1_training_runner.py::test_weak_model_retained_and_labeled_descriptive
tests/unit/test_simon_b1_training_runner.py::test_nan_loss_aborts_immediately
tests/unit/test_simon_b1_training_runner.py::test_zero_gradients_aborts_immediately
tests/unit/test_simon_b1_training_runner.py::test_non_finite_weights_aborts_immediately
tests/unit/test_simon_b1_training_runner.py::test_cli_requires_smoke_or_confirmation
tests/unit/test_simon_b1_training_runner.py::test_cli_smoke_test_runs_cleanly
tests/unit/test_simon_classical_ranking.py::test_ondemand_stage1_removes_temporary_output_on_success_and_failure
tests/unit/test_simon_classical_ranking.py::test_sparse_full_distribution_row_has_exact_zero_bins
tests/unit/test_simon_classical_ranking.py::test_row_gather_score_matches_independent_scalar_source_index[None]
tests/unit/test_simon_classical_ranking.py::test_row_gather_score_matches_independent_scalar_source_index[0]
tests/unit/test_simon_classical_ranking.py::test_row_gather_score_matches_independent_scalar_source_index[4660]
tests/unit/test_simon_classical_ranking.py::test_complete_screen_and_rescore_use_deterministic_ascending_ties[None-7]
tests/unit/test_simon_classical_ranking.py::test_complete_screen_and_rescore_use_deterministic_ascending_ties[7-6]
tests/unit/test_simon_classical_ranking.py::test_nan_row_and_wrong_stage_fail_explicitly
tests/unit/test_simon_classical_ranking.py::test_tie_count_and_collision_count_semantics
tests/unit/test_simon_classical_ranking.py::test_rank_classical_domain_rescores_all_64_pairs
tests/unit/test_simon_exact_ddt_equivalence.py::test_toy_width_exact_transition_equivalence
tests/unit/test_simon_exact_ddt_equivalence.py::test_queried_row_reconstruction_toy
tests/unit/test_simon_exact_ddt_equivalence.py::test_full_width_16bit_affine_basis_rank
tests/unit/test_simon_faithful_b1.py::test_simon_gohr_resnet_shapes_and_param_count
tests/unit/test_simon_faithful_b1.py::test_shared_simon_pair_encoder
tests/unit/test_simon_faithful_b1.py::test_simon_pair_group_ranker_permutation_invariance
tests/unit/test_simon_faithful_b1.py::test_simon_pair_group_ranker_flexible_pair_count[1]
tests/unit/test_simon_faithful_b1.py::test_simon_pair_group_ranker_flexible_pair_count[16]
tests/unit/test_simon_faithful_b1.py::test_simon_pair_group_ranker_flexible_pair_count[32]
tests/unit/test_simon_faithful_b1.py::test_simon_pair_group_ranker_flexible_pair_count[64]
tests/unit/test_simon_faithful_b1.py::test_simon_pair_group_ranker_flexible_pair_count[128]
tests/unit/test_simon_faithful_b1.py::test_b0_to_b1_weight_transfer
tests/unit/test_simon_faithful_b1.py::test_batch_norm_single_sample_fallback
tests/unit/test_simon_faithful_b1.py::test_faithful_feature_extraction
tests/unit/test_simon_faithful_b1.py::test_true_r10_key_features_equal_independent_nine_round_oracle
tests/unit/test_simon_faithful_b1.py::test_b1_group_mean_is_invariant_to_fourfold_pair_duplication
tests/unit/test_simon_faithful_b1.py::test_faithful_ranking_pair_count_enforcement
tests/unit/test_simon_faithful_b1.py::test_faithful_deterministic_tie_breaking
tests/unit/test_simon_faithful_b1.py::test_faithful_hybrid_8parent_queue
tests/unit/test_simon_faithful_b1.py::test_audit_contrast_faithful_vs_toy
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[0-3]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[0-4]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[0-6]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[1-3]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[1-4]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[1-6]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[3-3]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[3-4]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[3-6]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[7-3]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[7-4]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[7-6]
tests/unit/test_simon_predicted_ddt.py::test_full_width_affine_support_matches_exhaustive_scalar_histogram[0]
tests/unit/test_simon_predicted_ddt.py::test_full_width_affine_support_matches_exhaustive_scalar_histogram[1]
tests/unit/test_simon_predicted_ddt.py::test_full_width_affine_support_matches_exhaustive_scalar_histogram[64]
tests/unit/test_simon_predicted_ddt.py::test_full_width_affine_support_matches_exhaustive_scalar_histogram[128]
tests/unit/test_simon_predicted_ddt.py::test_full_width_affine_support_matches_exhaustive_scalar_histogram[4660]
tests/unit/test_simon_predicted_ddt.py::test_full_width_affine_support_matches_exhaustive_scalar_histogram[65535]
tests/unit/test_simon_predicted_ddt.py::test_affine_kernel_matches_pinned_source_probability_for_all_outputs[0]
tests/unit/test_simon_predicted_ddt.py::test_affine_kernel_matches_pinned_source_probability_for_all_outputs[1]
tests/unit/test_simon_predicted_ddt.py::test_affine_kernel_matches_pinned_source_probability_for_all_outputs[64]
tests/unit/test_simon_predicted_ddt.py::test_affine_kernel_matches_pinned_source_probability_for_all_outputs[4660]
tests/unit/test_simon_predicted_ddt.py::test_affine_kernel_matches_pinned_source_probability_for_all_outputs[65535]
tests/unit/test_simon_predicted_ddt.py::test_reference_rejects_full_width_generation_and_malformed_inputs
tests/unit/test_simon_published_vv_ranking.py::test_published_log_odds_polarity_matches_source_formula
tests/unit/test_simon_published_vv_ranking.py::test_published_log_odds_rejects_malformed_outputs[malformed0]
tests/unit/test_simon_published_vv_ranking.py::test_published_log_odds_rejects_malformed_outputs[malformed1]
tests/unit/test_simon_published_vv_ranking.py::test_published_log_odds_rejects_malformed_outputs[malformed2]
tests/unit/test_simon_published_vv_ranking.py::test_published_log_odds_rejects_malformed_outputs[malformed3]
tests/unit/test_simon_published_vv_ranking.py::test_published_log_odds_rejects_malformed_outputs[malformed4]
tests/unit/test_simon_published_vv_ranking.py::test_complete_domain_uses_candidate_word_ties[None]
tests/unit/test_simon_published_vv_ranking.py::test_complete_domain_uses_candidate_word_ties[7]
tests/unit/test_simon_published_vv_ranking.py::test_neural_ranker_rejects_invalid_pair_budget
tests/unit/test_simon_published_vv_ranking.py::test_neural_candidate_pair_order_matches_independent_scalar_oracle[None]
tests/unit/test_simon_published_vv_ranking.py::test_neural_candidate_pair_order_matches_independent_scalar_oracle[22136]
tests/unit/test_simon_published_vv_ranking.py::test_vv9_and_vv8_feature_and_score_aggregation_pinned_reference
tests/unit/test_simon_published_vv_ranking.py::test_rank_neural_domain_rescores_all_64_pairs
tests/unit/test_simon_ranking_features.py::test_source_word_and_msb_bit_order
tests/unit/test_simon_ranking_features.py::test_vectorized_candidate_states_match_scalar_and_independent_oracle[None]
tests/unit/test_simon_ranking_features.py::test_vectorized_candidate_states_match_scalar_and_independent_oracle[0]
tests/unit/test_simon_ranking_features.py::test_vectorized_candidate_states_match_scalar_and_independent_oracle[65535]
tests/unit/test_simon_ranking_features.py::test_correct_keys_have_independent_seven_and_six_round_differences
tests/unit/test_simon_ranking_features.py::test_direct_difference_expression_and_candidate_collision
tests/unit/test_simon_ranking_features.py::test_malformed_candidate_feature_inputs_are_rejected[pairs0-candidates0-None-ValueError]
tests/unit/test_simon_ranking_features.py::test_malformed_candidate_feature_inputs_are_rejected[pairs1-candidates1-None-ValueError]
tests/unit/test_simon_ranking_features.py::test_malformed_candidate_feature_inputs_are_rejected[pairs2-candidates2-None-ValueError]
tests/unit/test_simon_ranking_features.py::test_malformed_candidate_feature_inputs_are_rejected[pairs3-candidates3-None-TypeError]
tests/unit/test_simon_ranking_features.py::test_malformed_candidate_feature_inputs_are_rejected[pairs4-candidates4--1-ValueError]
tests/unit/test_simon_ranking_features.py::test_malformed_candidate_feature_inputs_are_rejected[pairs5-candidates5-None-ValueError]

390 tests collected in 20.32s

Exit code: 0
Elapsed seconds: 23.453

```

## statistics

- `tests/unit/test_compute_mcnemar_power.py`
- `tests/unit/test_depth_analysis.py`
- `tests/unit/test_evaluation_metrics.py`
- `tests/unit/test_paired_stats.py`
- `tests/unit/test_seven_round_analysis.py`
- `tests/unit/test_simon_confirmation_analysis.py`

```text
COMMAND: <work>/hamza-venv/Scripts/python.exe -m pytest --collect-only -q tests/unit/test_compute_mcnemar_power.py tests/unit/test_depth_analysis.py tests/unit/test_evaluation_metrics.py tests/unit/test_paired_stats.py tests/unit/test_seven_round_analysis.py tests/unit/test_simon_confirmation_analysis.py
tests/unit/test_compute_mcnemar_power.py::test_exact_binomial_twoside_pvalue_symmetry
tests/unit/test_compute_mcnemar_power.py::test_compute_exact_mcnemar_power_null_hypothesis
tests/unit/test_compute_mcnemar_power.py::test_compute_exact_mcnemar_power_monotonicity
tests/unit/test_compute_mcnemar_power.py::test_compute_exact_mcnemar_power_invalid_inputs
tests/unit/test_compute_mcnemar_power.py::test_generate_sensitivity_grid
tests/unit/test_compute_mcnemar_power.py::test_parse_args
tests/unit/test_depth_analysis.py::test_timeouts_are_unavailable_and_keep_failure_penalty
tests/unit/test_depth_analysis.py::test_analysis_rejects_grid_or_metric_inconsistency[missing]
tests/unit/test_depth_analysis.py::test_analysis_rejects_grid_or_metric_inconsistency[identity]
tests/unit/test_depth_analysis.py::test_analysis_rejects_grid_or_metric_inconsistency[flag]
tests/unit/test_depth_analysis.py::test_analysis_rejects_grid_or_metric_inconsistency[counts]
tests/unit/test_depth_analysis.py::test_analysis_rejects_grid_or_metric_inconsistency[late]
tests/unit/test_depth_analysis.py::test_hash_corruption_cannot_be_analyzed
tests/unit/test_evaluation_metrics.py::test_calculate_global_par2_basic
tests/unit/test_evaluation_metrics.py::test_calculate_global_par2_solved_exceeds_cutoff
tests/unit/test_evaluation_metrics.py::test_calculate_global_par2_solved_exact_cutoff
tests/unit/test_evaluation_metrics.py::test_calculate_global_par2_empty_and_errors
tests/unit/test_evaluation_metrics.py::test_wilson_score_interval_exact
tests/unit/test_evaluation_metrics.py::test_wilson_score_interval_validation
tests/unit/test_evaluation_metrics.py::test_exact_mcnemar_test_basic
tests/unit/test_evaluation_metrics.py::test_calculate_cactus_curve_monotonic
tests/unit/test_paired_stats.py::TestContingency2x2::test_compute_2x2_contingency_basic
tests/unit/test_paired_stats.py::TestContingency2x2::test_compute_2x2_contingency_mismatch
tests/unit/test_paired_stats.py::TestExactMcNemar::test_exact_binomial_twoside_pvalue
tests/unit/test_paired_stats.py::TestExactMcNemar::test_exact_mcnemar_literature_values
tests/unit/test_paired_stats.py::TestExactMcNemar::test_exact_mcnemar_errors
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_altman_table_6_2_literature_fixture
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_altman_table_6_2_reversed_symmetry
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_newcombe_table_3_literature_fixture
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_zero_discordance_ties
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_different_concordant_splits
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_negative_association
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_boundary_proportions_full_success
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_boundary_proportions_both_zero
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_boundary_proportions_both_one
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_complete_separation
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_small_sample_sizes_n1
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_reversal_symmetry_general
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_validation_errors
tests/unit/test_paired_stats.py::TestHolmBonferroniAdjustment::test_two_primary_contrasts_both_rejected
tests/unit/test_paired_stats.py::TestHolmBonferroniAdjustment::test_two_primary_contrasts_neither_rejected
tests/unit/test_paired_stats.py::TestHolmBonferroniAdjustment::test_two_primary_contrasts_one_rejected
tests/unit/test_paired_stats.py::TestHolmBonferroniAdjustment::test_order_invariance
tests/unit/test_paired_stats.py::TestHolmBonferroniAdjustment::test_boundary_cases
tests/unit/test_paired_stats.py::TestMcNemarPowerSensitivity::test_protocol_power_fixtures
tests/unit/test_paired_stats.py::TestMcNemarPowerSensitivity::test_power_edge_cases
tests/unit/test_seven_round_analysis.py::test_holm_bonferroni_correction
tests/unit/test_seven_round_analysis.py::test_compute_paired_contingency
tests/unit/test_seven_round_analysis.py::test_paired_risk_difference_ci
tests/unit/test_seven_round_analysis.py::test_analyze_records_end_to_end
tests/unit/test_seven_round_analysis.py::test_shuffled_instance_order_alignment
tests/unit/test_seven_round_analysis.py::test_duplicate_instance_id_rejection
tests/unit/test_seven_round_analysis.py::test_missing_method_rejection
tests/unit/test_seven_round_analysis.py::test_mismatched_instance_sets_rejection
tests/unit/test_seven_round_analysis.py::test_contradictory_record_integrity
tests/unit/test_seven_round_analysis.py::test_prospective_requires_n_128
tests/unit/test_seven_round_analysis.py::test_even_n_median_correctness
tests/unit/test_seven_round_analysis.py::test_newcombe_method_10_edge_cases
tests/unit/test_seven_round_analysis.py::test_analyzer_prospective_validation_gate
tests/unit/test_simon_confirmation_analysis.py::test_extract_record_success
tests/unit/test_simon_confirmation_analysis.py::test_compute_method_summary
tests/unit/test_simon_confirmation_analysis.py::test_analyze_all_comparators_complete
tests/unit/test_simon_confirmation_analysis.py::test_analyze_instance_mismatch_fails_closed
tests/unit/test_simon_confirmation_analysis.py::test_load_ledger_records_formats
tests/unit/test_simon_confirmation_analysis.py::test_generate_markdown_audit_report
tests/unit/test_simon_confirmation_analysis.py::test_missing_oracle_fails_closed
tests/unit/test_simon_confirmation_analysis.py::test_corrupt_sidecar_fails_closed
tests/unit/test_simon_confirmation_analysis.py::test_drift_rejected_in_main

68 tests collected in 11.28s

Exit code: 0
Elapsed seconds: 14.438

```

## brute-force completer

- `tests/unit/test_bruteforce_completion.py`

```text
COMMAND: <work>/hamza-venv/Scripts/python.exe -m pytest --collect-only -q tests/unit/test_bruteforce_completion.py
tests/unit/test_bruteforce_completion.py::test_fixtures_match_task_known_answer_values
tests/unit/test_bruteforce_completion.py::test_official_known_answers_through_c_encryption
tests/unit/test_bruteforce_completion.py::test_official_known_answers_recovered_by_completion
tests/unit/test_bruteforce_completion.py::test_schedule_inversion_round_trip[speck-rounds_range0]
tests/unit/test_bruteforce_completion.py::test_schedule_inversion_round_trip[simon-rounds_range1]
tests/unit/test_bruteforce_completion.py::test_arbitrary_windows_invert_consistently[speck-22]
tests/unit/test_bruteforce_completion.py::test_arbitrary_windows_invert_consistently[simon-32]
tests/unit/test_bruteforce_completion.py::test_full_m3_space_visits_distinct_master_keys[speck-8]
tests/unit/test_bruteforce_completion.py::test_full_m3_space_visits_distinct_master_keys[simon-11]
tests/unit/test_bruteforce_completion.py::test_expand_and_encrypt_agree_with_python_reference[speck-22]
tests/unit/test_bruteforce_completion.py::test_expand_and_encrypt_agree_with_python_reference[simon-32]
tests/unit/test_bruteforce_completion.py::test_speck_expand_agrees_with_gohr_convention
tests/unit/test_bruteforce_completion.py::test_full_space_m3_completion_finds_true_key[speck-8]
tests/unit/test_bruteforce_completion.py::test_full_space_m3_completion_finds_true_key[simon-11]
tests/unit/test_bruteforce_completion.py::test_narrow_ranges_for_partial_trailing_keys[speck-8-2]
tests/unit/test_bruteforce_completion.py::test_narrow_ranges_for_partial_trailing_keys[speck-8-1]
tests/unit/test_bruteforce_completion.py::test_narrow_ranges_for_partial_trailing_keys[simon-11-2]
tests/unit/test_bruteforce_completion.py::test_narrow_ranges_for_partial_trailing_keys[simon-11-1]
tests/unit/test_bruteforce_completion.py::test_wrong_trailing_key_is_rejected_over_full_space[speck-8]
tests/unit/test_bruteforce_completion.py::test_wrong_trailing_key_is_rejected_over_full_space[simon-11]
tests/unit/test_bruteforce_completion.py::test_every_block_is_verified_not_only_the_first
tests/unit/test_bruteforce_completion.py::test_invalid_arguments_are_rejected
tests/unit/test_bruteforce_completion.py::test_build_uses_frozen_flags_and_produces_library
tests/unit/test_bruteforce_completion.py::test_quick_benchmark_cli_writes_complete_record

24 tests collected in 9.38s

Exit code: 0
Elapsed seconds: 12.359

```

## run guards

- `tests/unit/test_backend_environment.py`
- `tests/unit/test_depth_map_runner.py`
- `tests/unit/test_phase7_protocol.py`
- `tests/unit/test_protocol_manifest.py`
- `tests/unit/test_queue_order_environment.py`
- `tests/unit/test_simon_candidate_history_observation.py`
- `tests/unit/test_simon_confirmation_guards.py`
- `tests/unit/test_simon_confirmation_launch.py`
- `tests/unit/test_simon_operating_points_map.py`
- `tests/unit/test_simon_operating_window.py`
- `tests/unit/test_simon_stage2_control.py`
- `tests/unit/test_simon_stage2_receipts.py`
- `tests/unit/test_simon_stage2_watchdog.py`
- `tests/unit/test_simon_stage2_worker.py`
- `tests/unit/test_worker_telemetry.py`

```text
COMMAND: <work>/hamza-venv/Scripts/python.exe -m pytest --collect-only -q tests/unit/test_backend_environment.py tests/unit/test_depth_map_runner.py tests/unit/test_phase7_protocol.py tests/unit/test_protocol_manifest.py tests/unit/test_queue_order_environment.py tests/unit/test_simon_candidate_history_observation.py tests/unit/test_simon_confirmation_guards.py tests/unit/test_simon_confirmation_launch.py tests/unit/test_simon_operating_points_map.py tests/unit/test_simon_operating_window.py tests/unit/test_simon_stage2_control.py tests/unit/test_simon_stage2_receipts.py tests/unit/test_simon_stage2_watchdog.py tests/unit/test_simon_stage2_worker.py tests/unit/test_worker_telemetry.py
tests/unit/test_backend_environment.py::test_power_observation_requires_successful_ac_and_sleep[Now drawing from 'AC Power'-PreventSystemSleep 1-0-True]
tests/unit/test_backend_environment.py::test_power_observation_requires_successful_ac_and_sleep[Now drawing from 'Battery Power'-PreventSystemSleep 1-0-False]
tests/unit/test_backend_environment.py::test_power_observation_requires_successful_ac_and_sleep[Now drawing from 'AC Power'-PreventSystemSleep 0-0-False]
tests/unit/test_backend_environment.py::test_power_observation_requires_successful_ac_and_sleep[Now drawing from 'AC Power'-PreventSystemSleep 1-1-False]
tests/unit/test_backend_environment.py::test_unavailable_power_query_rejects
tests/unit/test_backend_environment.py::test_process_policy_does_not_substitute_for_thread_qos[0-0-0-True]
tests/unit/test_backend_environment.py::test_process_policy_does_not_substitute_for_thread_qos[0-1-0-False]
tests/unit/test_backend_environment.py::test_process_policy_does_not_substitute_for_thread_qos[1-0-0-False]
tests/unit/test_backend_environment.py::test_process_policy_does_not_substitute_for_thread_qos[0-0-3-False]
tests/unit/test_backend_environment.py::test_unsupported_platform_is_explicit
tests/unit/test_depth_map_runner.py::test_manifest_validation_passes
tests/unit/test_depth_map_runner.py::test_manifest_validation_rejects_tampering
tests/unit/test_depth_map_runner.py::test_schedule_properties
tests/unit/test_depth_map_runner.py::test_parent_watchdog_terminates_hanging_worker
tests/unit/test_depth_map_runner.py::test_unsat_triggers_correctness_alarm
tests/unit/test_depth_map_runner.py::test_atomic_journal_and_interrupted_cell
tests/unit/test_depth_map_runner.py::test_single_cell_real_solve
tests/unit/test_depth_map_runner.py::test_simon_single_cell_real_solve
tests/unit/test_depth_map_runner.py::test_late_recovery_rejected
tests/unit/test_phase7_protocol.py::test_phase7_config_is_held_out_test_and_seed_disjoint
tests/unit/test_phase7_protocol.py::test_phase7_methods_include_all_five_configurations
tests/unit/test_phase7_protocol.py::test_phase7_counterbalanced_order_is_cyclic_permutation
tests/unit/test_phase7_protocol.py::test_phase7_atomic_write_and_summary_aggregation
tests/unit/test_protocol_manifest.py::test_authoritative_manifest_passes_validation
tests/unit/test_protocol_manifest.py::test_seed_governance_rejection
tests/unit/test_protocol_manifest.py::test_hash_tamper_rejection
tests/unit/test_protocol_manifest.py::test_parameter_tamper_rejection
tests/unit/test_protocol_manifest.py::test_comparator_methods_completeness
tests/unit/test_protocol_manifest.py::test_duplicate_method_ids_rejection
tests/unit/test_protocol_manifest.py::test_method_parameter_bounds_rejection
tests/unit/test_protocol_manifest.py::test_missing_required_section_rejection
tests/unit/test_protocol_manifest.py::test_failure_taxonomy_completeness_and_distinction
tests/unit/test_protocol_manifest.py::test_statistical_plan_and_multiplicity_rejection
tests/unit/test_protocol_manifest.py::test_formula_metrics_and_environment_rejection
tests/unit/test_protocol_manifest.py::test_overwrite_protection
tests/unit/test_protocol_manifest.py::test_prospective_execution_refused_on_proposed_status
tests/unit/test_protocol_manifest.py::test_prospective_execution_refused_on_null_seed
tests/unit/test_protocol_manifest.py::test_prospective_execution_accepted_on_frozen_authorized
tests/unit/test_protocol_manifest.py::test_d030_rejected_for_prospective_authorization
tests/unit/test_protocol_manifest.py::test_retired_seed_rejection_in_prospective
tests/unit/test_protocol_manifest.py::test_failure_taxonomy_orthogonal_fields_validation
tests/unit/test_protocol_manifest.py::test_classify_attack_outcome_mapping
tests/unit/test_protocol_manifest.py::test_prospective_manifest_rejects_seed_406737
tests/unit/test_protocol_manifest.py::test_prospective_dataset_validation_rejects_seed_406737
tests/unit/test_protocol_manifest.py::test_development_dataset_inspection_allowed_without_prospective_status
tests/unit/test_protocol_manifest.py::test_seed_policy_governance_for_406737
tests/unit/test_protocol_manifest.py::test_critical_code_mutation_invalidates_manifest
tests/unit/test_queue_order_environment.py::test_author_allows_battery_or_ac_with_observed_sleep_protection[False]
tests/unit/test_queue_order_environment.py::test_author_allows_battery_or_ac_with_observed_sleep_protection[True]
tests/unit/test_queue_order_environment.py::test_battery_allowance_keeps_telemetry_and_sleep_gates[supported-False]
tests/unit/test_queue_order_environment.py::test_battery_allowance_keeps_telemetry_and_sleep_gates[sleep_assertion_observed-False]
tests/unit/test_queue_order_environment.py::test_battery_allowance_keeps_telemetry_and_sleep_gates[assertions_returncode-1]
tests/unit/test_queue_order_environment.py::test_battery_allowance_keeps_telemetry_and_sleep_gates[power_returncode-1]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[Now drawing from 'Unknown Power'\n-False]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[-False]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[None-False]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[Now drawing from 'Battery Power'\nNow drawing from 'AC Power'\n-True]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[Now drawing from 'AC Power'\nNow drawing from 'Unknown Power'\n-True]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[Now drawing from 'Battery Power'\n-True]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[Now drawing from 'AC Power'\n-False]
tests/unit/test_queue_order_environment.py::test_interpreter_alias_accepts_same_venv_but_rejects_other_venv_or_base
tests/unit/test_simon_candidate_history_observation.py::test_complete_grid_derives_only_matched_target_with_unsat_history
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[duplicate-Duplicate method/key]
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[missing-Incomplete method/key]
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[flag-Non-Boolean]
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[false_flag-Outcome/flag conflict]
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[wrong_target-Target tuple differs]
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[rebuild-Intervening solver rebuild]
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[unknown_prefix-non-UNSAT]
tests/unit/test_simon_candidate_history_observation.py::test_false_master_key_and_failed_fresh_verification_are_rejected
tests/unit/test_simon_candidate_history_observation.py::test_frozen_hash_sidecar_and_duplicate_fields
tests/unit/test_simon_confirmation_guards.py::test_unapproved_manifest_rejected
tests/unit/test_simon_confirmation_guards.py::test_unauthorized_execution_rejected
tests/unit/test_simon_confirmation_guards.py::test_null_seed_rejected
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[260926]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[406737]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[98001]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[880001]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[880002]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[999999]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[669981]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[101001]
tests/unit/test_simon_confirmation_guards.py::test_invalid_selected_instances_rejected[None]
tests/unit/test_simon_confirmation_guards.py::test_invalid_selected_instances_rejected[16]
tests/unit/test_simon_confirmation_guards.py::test_invalid_selected_instances_rejected[48]
tests/unit/test_simon_confirmation_guards.py::test_invalid_selected_instances_rejected[128]
tests/unit/test_simon_confirmation_guards.py::test_invalid_selected_instances_rejected[32]
tests/unit/test_simon_confirmation_guards.py::test_missing_or_modified_methods_rejected
tests/unit/test_simon_confirmation_guards.py::test_modified_queue_control_seed_rejected
tests/unit/test_simon_confirmation_guards.py::test_spy_verifies_zero_worker_calls_on_preflight_rejection
tests/unit/test_simon_confirmation_guards.py::test_provenance_verification_fails_closed_on_hash_drift
tests/unit/test_simon_confirmation_guards.py::test_disjoint_family_certificate_rejects_duplicate_keys
tests/unit/test_simon_confirmation_guards.py::test_disjoint_family_certificate_detects_development_overlap
tests/unit/test_simon_confirmation_guards.py::test_disjoint_family_certificate_rejects_non_confirmation_role
tests/unit/test_simon_confirmation_guards.py::test_disjoint_family_certificate_build_and_verify_passes_on_fresh_keys
tests/unit/test_simon_confirmation_guards.py::test_resume_guards_reject_unflagged_rerun_on_existing_output
tests/unit/test_simon_confirmation_guards.py::test_resume_guards_reject_missing_checksum_sidecar
tests/unit/test_simon_confirmation_guards.py::test_resume_guards_reject_corrupted_checksum_sidecar
tests/unit/test_simon_confirmation_guards.py::test_resume_guards_reject_tampered_execution_identity
tests/unit/test_simon_confirmation_guards.py::test_resume_guards_reject_duplicate_cell
tests/unit/test_simon_confirmation_guards.py::test_resume_guards_reject_unverified_claimed_recovery
tests/unit/test_simon_confirmation_guards.py::test_resume_correctly_filters_completed_cells
tests/unit/test_simon_confirmation_guards.py::test_forged_zero_certificate_rejected_by_verify_disjoint_family_certificate
tests/unit/test_simon_confirmation_guards.py::test_generate_confirmation_dataset_fails_closed_without_authorization
tests/unit/test_simon_confirmation_guards.py::test_generate_confirmation_dataset_succeeds_when_authorized
tests/unit/test_simon_confirmation_guards.py::test_derive_confirmation_queue_seed_properties
tests/unit/test_simon_confirmation_launch.py::test_nested_profile_drift_is_rejected[resource_policy-solver_threads-2]
tests/unit/test_simon_confirmation_launch.py::test_nested_profile_drift_is_rejected[random_control_stream-prng_algorithm-another generator]
tests/unit/test_simon_confirmation_launch.py::test_nested_profile_drift_is_rejected[statistical_analysis_plan-analysis_freeze_status-changed]
tests/unit/test_simon_confirmation_launch.py::test_nested_profile_drift_is_rejected[candidate_sat_specification-formula_variables-1857]
tests/unit/test_simon_confirmation_launch.py::test_opened_eight_pair_demo_is_in_overlap_union
tests/unit/test_simon_confirmation_launch.py::test_generation_rejects_unapproved_before_any_data_draw
tests/unit/test_simon_confirmation_launch.py::test_generation_refuses_existing_artifact
tests/unit/test_simon_confirmation_launch.py::test_execution_pins_reject_source_change_before_runtime_probe
tests/unit/test_simon_confirmation_launch.py::test_analysis_cannot_accept_missing_confirmation_identity
tests/unit/test_simon_confirmation_launch.py::test_success_flag_cannot_replace_missing_fresh_verification
tests/unit/test_simon_confirmation_launch.py::test_missing_component_time_is_unknown
tests/unit/test_simon_confirmation_launch.py::test_training_replay_matches_original_generator[None]
tests/unit/test_simon_confirmation_launch.py::test_training_replay_matches_original_generator[64]
tests/unit/test_simon_confirmation_launch.py::test_real_frozen_sample_size_cannot_drift
tests/unit/test_simon_confirmation_launch.py::test_same_package_version_changed_native_bytes_changes_identity
tests/unit/test_simon_confirmation_launch.py::test_contradictory_authorization_is_rejected[seed]
tests/unit/test_simon_confirmation_launch.py::test_contradictory_authorization_is_rejected[gate]
tests/unit/test_simon_confirmation_launch.py::test_contradictory_authorization_is_rejected[blocker]
tests/unit/test_simon_confirmation_launch.py::test_confirmation_analysis_rejects_equal_partial_grid
tests/unit/test_simon_operating_points_map.py::test_cohort_identity_slicing_and_divergence
tests/unit/test_simon_operating_points_map.py::test_deterministic_replay
tests/unit/test_simon_operating_points_map.py::test_independent_solver_construction
tests/unit/test_simon_operating_points_map.py::test_worker_solve_budget_exhausted
tests/unit/test_simon_operating_points_map.py::test_worker_solve_timeout_classification
tests/unit/test_simon_operating_points_map.py::test_fresh_pair_verification_rejects_wrong_key
tests/unit/test_simon_operating_points_map.py::test_run_map_workers_guard
tests/unit/test_simon_operating_points_map.py::test_parent_deadline_kills_overrunning_worker
tests/unit/test_simon_operating_points_map.py::test_worker_finishes_just_within_deadline
tests/unit/test_simon_operating_points_map.py::test_deadline_classifications_separate_outcomes
tests/unit/test_simon_operating_points_map.py::test_run_map_r10_serial_profile_cells
tests/unit/test_simon_operating_points_map.py::test_resume_rejects_changed_profile_or_worker_count
tests/unit/test_simon_operating_window.py::test_observations_are_ciphertexts_not_differential_pairs
tests/unit/test_simon_operating_window.py::test_demo_keys_are_distinct_and_pairing_is_stable
tests/unit/test_simon_operating_window.py::test_wrong_tuple_probe_is_unique_truth_excluded_and_reproducible
tests/unit/test_simon_operating_window.py::test_grid_screen_applies_paired_counts_and_censored_median
tests/unit/test_simon_operating_window.py::test_checkpoint_refuses_overwrite_and_bad_resume
tests/unit/test_simon_operating_window.py::test_one_true_key_control_matches_reference_and_fresh_verification
tests/unit/test_simon_operating_window.py::test_solver_timeout_is_unresolved_not_unsat
tests/unit/test_simon_operating_window.py::test_detached_checksum_refuses_modified_artifact
tests/unit/test_simon_stage2_control.py::test_fixed_job_grid_and_ordered_ciphertext_only_projection
tests/unit/test_simon_stage2_control.py::test_public_payload_and_words_reject_extra_truth_and_non_integer_values
tests/unit/test_simon_stage2_control.py::test_payload_contract_and_rank_sum_queue_order
tests/unit/test_simon_stage2_control.py::test_full_binding_accepts_d053_and_rejects_old_pilot[D-051]
tests/unit/test_simon_stage2_control.py::test_full_binding_accepts_d053_and_rejects_old_pilot[D-052]
tests/unit/test_simon_stage2_control.py::test_full_analyzer_keeps_8_key_denominators_and_uses_phase_costs
tests/unit/test_simon_stage2_control.py::test_analyzer_separates_structural_parent_miss_from_partial_stage2_unavailability
tests/unit/test_simon_stage2_control.py::test_tombstones_keep_unknown_dispatch_and_proven_no_worker_distinct
tests/unit/test_simon_stage2_control.py::test_worker_cleanup_overshoot_is_preserved_without_violating_phase_caps
tests/unit/test_simon_stage2_control.py::test_interrupted_success_receipt_does_not_invent_queue_persistence
tests/unit/test_simon_stage2_control.py::test_only_exact_sealed_orphan_tombstone_is_analyzable
tests/unit/test_simon_stage2_control.py::test_pilot_returns_non_scientific_without_touching_truth_and_closure_is_strict
tests/unit/test_simon_stage2_control.py::test_interpreter_file_pin_preserves_lexical_symlink
tests/unit/test_simon_stage2_receipts.py::test_completed_pilot_accepts_bound_rss_policy_and_ac_receipts
tests/unit/test_simon_stage2_receipts.py::test_battery_is_valid_under_explicit_author_policy
tests/unit/test_simon_stage2_receipts.py::test_pilot_requires_observed_worker_member
tests/unit/test_simon_stage2_receipts.py::test_rss_stop_accepts_only_a_real_over_limit_sample
tests/unit/test_simon_stage2_receipts.py::test_telemetry_must_match_observed_runtime_policy[pid-3999-Telemetry PID]
tests/unit/test_simon_stage2_receipts.py::test_telemetry_must_match_observed_runtime_policy[parent_pid-2999-parent PID]
tests/unit/test_simon_stage2_receipts.py::test_telemetry_must_match_observed_runtime_policy[git_commit-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb-source commit]
tests/unit/test_simon_stage2_receipts.py::test_telemetry_must_match_observed_runtime_policy[tmpdir_safe-False-TMPDIR]
tests/unit/test_simon_stage2_receipts.py::test_telemetry_must_match_observed_runtime_policy[thread_environment-value4-thread environment]
tests/unit/test_simon_stage2_receipts.py::test_telemetry_must_match_observed_runtime_policy[qos_class-9-QoS]
tests/unit/test_simon_stage2_receipts.py::test_process_policy_errno_is_not_hidden_by_accepted_flag
tests/unit/test_simon_stage2_receipts.py::test_sleep_assertion_owner_is_bound_to_guardian
tests/unit/test_simon_stage2_receipts.py::test_corrupt_sample_sidecar_fails_closed
tests/unit/test_simon_stage2_receipts.py::test_partial_power_only_sample_is_allowed_without_worker_observation
tests/unit/test_simon_stage2_receipts.py::test_missing_samples_on_partial_row_remain_unobserved_not_zero_rss
tests/unit/test_simon_stage2_watchdog.py::test_accepts_loaded_then_all_eight_calls_and_advances_deadline
tests/unit/test_simon_stage2_watchdog.py::test_waits_briefly_for_event_sidecar_without_rejecting_complete_event
tests/unit/test_simon_stage2_watchdog.py::test_deadline_stops_hangs_and_retains_only_completed_calls[hang_load-load_timeout-0]
tests/unit/test_simon_stage2_watchdog.py::test_deadline_stops_hangs_and_retains_only_completed_calls[hang_call-rank_timeout-1]
tests/unit/test_simon_stage2_watchdog.py::test_deadline_stops_hangs_and_retains_only_completed_calls[hang_finalize-rank_timeout-2]
tests/unit/test_simon_stage2_watchdog.py::test_corrupt_or_nonsequential_journal_fails_closed[future_event]
tests/unit/test_simon_stage2_watchdog.py::test_corrupt_or_nonsequential_journal_fails_closed[bad_result_hash]
tests/unit/test_simon_stage2_watchdog.py::test_corrupt_or_nonsequential_journal_fails_closed[wrong_result_identity]
tests/unit/test_simon_stage2_watchdog.py::test_corrupt_or_nonsequential_journal_fails_closed[wrong_event_identity]
tests/unit/test_simon_stage2_watchdog.py::test_corrupt_or_nonsequential_journal_fails_closed[backdated_second_call]
tests/unit/test_simon_stage2_watchdog.py::test_campaign_deadline_caps_loading_deadline
tests/unit/test_simon_stage2_watchdog.py::test_configuration_drift_is_rejected_before_worker_dispatch
tests/unit/test_simon_stage2_watchdog.py::test_coordinator_loss_before_dispatch_writes_stop_receipt
tests/unit/test_simon_stage2_watchdog.py::test_coordinator_loss_after_dispatch_stops_worker_group
tests/unit/test_simon_stage2_watchdog.py::test_live_groups_matches_only_exact_stage2_argv
tests/unit/test_simon_stage2_worker.py::test_call_schedule_preserves_all_eight_parent_ordinals
tests/unit/test_simon_stage2_worker.py::test_rank_arguments_keep_the_frozen_screen_rescore_and_depths
tests/unit/test_simon_stage2_worker.py::test_checkpoint_loader_selects_vv9_vv8_and_ddt7_depths
tests/unit/test_simon_stage2_worker.py::test_rank_result_rejects_duplicate_and_wrong_width_orders
tests/unit/test_simon_stage2_worker.py::test_real_ranker_uint16_array_crosses_strict_json_boundary[vv9-None]
tests/unit/test_simon_stage2_worker.py::test_real_ranker_uint16_array_crosses_strict_json_boundary[vv8-123]
tests/unit/test_simon_stage2_worker.py::test_real_ranker_uint16_array_crosses_strict_json_boundary[ddt7-123]
tests/unit/test_simon_stage2_worker.py::test_float_rank_array_is_not_silently_truncated_to_words
tests/unit/test_simon_stage2_worker.py::test_worker_loads_one_checkpoint_and_passes_ciphertext_only_to_eight_calls
tests/unit/test_simon_stage2_worker.py::test_namespace_bootstrap_skips_both_package_initializers
tests/unit/test_simon_stage2_worker.py::test_import_blocker_rejects_solver_and_verifier_modules
tests/unit/test_simon_stage2_worker.py::test_importing_worker_module_does_not_import_project_or_solver_modules
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[   PreventSystemSleep              1\n-True]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[\tPreventUserIdleSystemSleep\t1\t\n-True]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[PreventSystemSleep 0\nPreventUserIdleSystemSleep 1\n-True]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[PreventSystemSleep 1-True]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[PreventSystemSleep 0\nPreventUserIdleSystemSleep 0\n-False]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[PreventSystemSleep 10\n-False]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[OtherPreventSystemSleep 1\n-False]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[pid 42: PreventSystemSleep 1\n-False]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[PreventSystemSleep 1 owner detail\n-False]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[\n-False]
tests/unit/test_worker_telemetry.py::test_sleep_assertions_unsupported_off_darwin
tests/unit/test_worker_telemetry.py::test_live_worker_telemetry_observation
tests/unit/test_worker_telemetry.py::test_adversarial_qos9_zero_priority_rejected
tests/unit/test_worker_telemetry.py::test_unspecified_or_failed_qos_rejected
tests/unit/test_worker_telemetry.py::test_failed_priority_or_io_queries_rejected
tests/unit/test_worker_telemetry.py::test_darwin_bg_state_rejected
tests/unit/test_worker_telemetry.py::test_io_throttle_rejected
tests/unit/test_worker_telemetry.py::test_degraded_process_priority_rejected
tests/unit/test_worker_telemetry.py::test_unsafe_tmpdir_rejected
tests/unit/test_worker_telemetry.py::test_symlink_escape_tmpdir_rejected
tests/unit/test_worker_telemetry.py::test_valid_worker_certification_passes
tests/unit/test_worker_telemetry.py::test_save_worker_receipt_with_sidecar
tests/unit/test_worker_telemetry.py::test_stale_or_mismatched_receipt_identity

223 tests collected in 13.88s

Exit code: 0
Elapsed seconds: 16.859

```

## other

- `tests/unit/test_attack_instances.py`
- `tests/unit/test_backend_artifact_semantics.py`
- `tests/unit/test_backend_artifacts.py`
- `tests/unit/test_backend_completed_evidence.py`
- `tests/unit/test_backend_evidence.py`
- `tests/unit/test_conditioning_final_audit.py`
- `tests/unit/test_data_generator.py`
- `tests/unit/test_dataset_export.py`
- `tests/unit/test_dataset_v1_freeze.py`
- `tests/unit/test_demo_depth_dataset.py`
- `tests/unit/test_encryption_calibration.py`
- `tests/unit/test_final_publication_figures.py`
- `tests/unit/test_phase6b_proposal_figures.py`
- `tests/unit/test_phase7_evaluation_figures.py`
- `tests/unit/test_publication_evidence_package.py`
- `tests/unit/test_reference_encryption_report.py`
- `tests/unit/test_seven_round_prospective_data.py`
- `tests/unit/test_simon_attack_instance.py`
- `tests/unit/test_simon_higher_round.py`
- `tests/unit/test_simon_sat_300s_control.py`

```text
COMMAND: <work>/hamza-venv/Scripts/python.exe -m pytest --collect-only -q tests/unit/test_attack_instances.py tests/unit/test_backend_artifact_semantics.py tests/unit/test_backend_artifacts.py tests/unit/test_backend_completed_evidence.py tests/unit/test_backend_evidence.py tests/unit/test_conditioning_final_audit.py tests/unit/test_data_generator.py tests/unit/test_dataset_export.py tests/unit/test_dataset_v1_freeze.py tests/unit/test_demo_depth_dataset.py tests/unit/test_encryption_calibration.py tests/unit/test_final_publication_figures.py tests/unit/test_phase6b_proposal_figures.py tests/unit/test_phase7_evaluation_figures.py tests/unit/test_publication_evidence_package.py tests/unit/test_reference_encryption_report.py tests/unit/test_seven_round_prospective_data.py tests/unit/test_simon_attack_instance.py tests/unit/test_simon_higher_round.py tests/unit/test_simon_sat_300s_control.py
tests/unit/test_attack_instances.py::test_attack_generation_is_identical_for_a_fixed_seed
tests/unit/test_attack_instances.py::test_different_attack_seeds_produce_different_instances
tests/unit/test_attack_instances.py::test_attack_pairs_share_one_key_and_the_selected_difference
tests/unit/test_attack_instances.py::test_attack_split_keys_do_not_overlap
tests/unit/test_attack_instances.py::test_attack_export_checksums_and_independent_load
tests/unit/test_attack_instances.py::test_attack_loader_rejects_tampered_ciphertext
tests/unit/test_attack_instances.py::test_invalid_attack_configuration_is_rejected[0-2-7-train]
tests/unit/test_attack_instances.py::test_invalid_attack_configuration_is_rejected[2-0-7-train]
tests/unit/test_attack_instances.py::test_invalid_attack_configuration_is_rejected[2-2-0-train]
tests/unit/test_attack_instances.py::test_invalid_attack_configuration_is_rejected[2-2-23-train]
tests/unit/test_attack_instances.py::test_invalid_attack_configuration_is_rejected[2-2-7-holdout]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-observation]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[truth-leaked-into-public]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-trailing-key-unit]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-master-truth]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-verification-pair]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-variable-map]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-bit-order]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-native-formula]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-pure-cnf-formula]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_forged_catalog_metadata[counts]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_forged_catalog_metadata[known-observations]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_forged_catalog_metadata[assumption-units]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_accepts_canonical_minimal_bundle
tests/unit/test_backend_artifact_semantics.py::test_catalog_observations_are_checked_against_formula_not_file_hash_only
tests/unit/test_backend_artifacts.py::test_solver_public_boundary_and_fresh_verification[speck32_64-7]
tests/unit/test_backend_artifacts.py::test_solver_public_boundary_and_fresh_verification[simon32_64-10]
tests/unit/test_backend_completed_evidence.py::test_signed_xor_parity_and_unit_clause[x1 2 0]
tests/unit/test_backend_completed_evidence.py::test_signed_xor_parity_and_unit_clause[x 1 2 0]
tests/unit/test_backend_completed_evidence.py::test_negative_xor_literal_changes_parity
tests/unit/test_backend_completed_evidence.py::test_unsatisfied_disjunction_or_empty_constraint_is_rejected[1 -2 0]
tests/unit/test_backend_completed_evidence.py::test_unsatisfied_disjunction_or_empty_constraint_is_rejected[0]
tests/unit/test_backend_completed_evidence.py::test_unsatisfied_disjunction_or_empty_constraint_is_rejected[x0]
tests/unit/test_backend_completed_evidence.py::test_incomplete_nonboolean_and_out_of_range_assignments_are_rejected[assignment0]
tests/unit/test_backend_completed_evidence.py::test_incomplete_nonboolean_and_out_of_range_assignments_are_rejected[assignment1]
tests/unit/test_backend_completed_evidence.py::test_incomplete_nonboolean_and_out_of_range_assignments_are_rejected[assignment2]
tests/unit/test_backend_completed_evidence.py::test_malformed_header_range_or_record_is_rejected[p cnf 2 2\n1 0\n]
tests/unit/test_backend_completed_evidence.py::test_malformed_header_range_or_record_is_rejected[p cnf 2 1\n3 0\n]
tests/unit/test_backend_completed_evidence.py::test_malformed_header_range_or_record_is_rejected[p cnf 2 1\n1 0 2 0\n]
tests/unit/test_backend_completed_evidence.py::test_decode_is_lsb_first_and_uses_variable_map_order
tests/unit/test_backend_completed_evidence.py::test_forced_extension_handles_negative_xor_and_auxiliary_clause
tests/unit/test_backend_completed_evidence.py::test_contradictory_reference_witness_is_rejected[cnf0-xors0]
tests/unit/test_backend_completed_evidence.py::test_contradictory_reference_witness_is_rejected[cnf1-xors1]
tests/unit/test_backend_completed_evidence.py::test_unforced_reference_extension_is_unavailable_not_filled_arbitrarily
tests/unit/test_backend_evidence.py::test_audit_reparses_complete_native_sat_model
tests/unit/test_backend_evidence.py::test_audit_accepts_native_unsat_as_recorded_alarm
tests/unit/test_backend_evidence.py::test_wrapper_audit_does_not_claim_a_captured_native_model
tests/unit/test_backend_evidence.py::test_late_worker_recovery_is_preserved_but_not_credited
tests/unit/test_backend_evidence.py::test_interruption_reports_unavailable_time_without_normal_receipts
tests/unit/test_backend_evidence.py::test_missing_environment_result_requires_parent_stop_proof[coordinator_stop-True]
tests/unit/test_backend_evidence.py::test_missing_environment_result_requires_parent_stop_proof[coordinator_stop-False]
tests/unit/test_backend_evidence.py::test_missing_environment_result_requires_parent_stop_proof[worker_exit-True]
tests/unit/test_backend_evidence.py::test_coordinator_record_mutation_is_rejected
tests/unit/test_backend_evidence.py::test_nonzero_worker_exit_receipt_is_rejected_even_when_resealed
tests/unit/test_backend_evidence.py::test_unsealed_or_tampered_raw_solver_output_is_rejected
tests/unit/test_backend_evidence.py::test_guardian_must_itself_certify_group_absence[False]
tests/unit/test_backend_evidence.py::test_guardian_must_itself_certify_group_absence[None]
tests/unit/test_backend_evidence.py::test_guardian_must_itself_certify_group_absence[true]
tests/unit/test_backend_evidence.py::test_resealed_solver_output_must_still_match_worker_status
tests/unit/test_backend_evidence.py::test_raw_worker_key_must_match_coordinator_result
tests/unit/test_backend_evidence.py::test_worker_directory_cannot_be_redirected_or_symlinked
tests/unit/test_backend_evidence.py::test_raw_artifact_paths_must_be_basenames
tests/unit/test_backend_evidence.py::test_interrupted_record_must_preserve_original_group_and_invocation
tests/unit/test_conditioning_final_audit.py::test_original_formula_check_rejects_parity_and_conditioning_errors
tests/unit/test_conditioning_final_audit.py::test_duplicate_json_fields_are_rejected
tests/unit/test_conditioning_final_audit.py::test_checksum_and_sidecar_filename_are_both_required
tests/unit/test_conditioning_final_audit.py::test_summary_keeps_failures_and_recomputes_paired_denominators
tests/unit/test_data_generator.py::test_word_encoding_is_most_significant_bit_first
tests/unit/test_data_generator.py::test_generation_is_identical_for_a_fixed_configuration_and_seed
tests/unit/test_data_generator.py::test_different_seeds_produce_different_samples
tests/unit/test_data_generator.py::test_generated_shapes_labels_and_debug_pairs_are_consistent
tests/unit/test_data_generator.py::test_positive_debug_pairs_use_the_selected_difference
tests/unit/test_data_generator.py::test_split_configs_require_distinct_seeds_and_shared_protocol
tests/unit/test_data_generator.py::test_invalid_dataset_configuration_is_rejected[0-7-1001-train]
tests/unit/test_data_generator.py::test_invalid_dataset_configuration_is_rejected[3-7-1001-train]
tests/unit/test_data_generator.py::test_invalid_dataset_configuration_is_rejected[8-0-1001-train]
tests/unit/test_data_generator.py::test_invalid_dataset_configuration_is_rejected[8-23-1001-train]
tests/unit/test_data_generator.py::test_invalid_dataset_configuration_is_rejected[8-7-1001-holdout]
tests/unit/test_dataset_export.py::test_classifier_export_is_identical_for_a_fixed_seed
tests/unit/test_dataset_export.py::test_different_seeds_change_exported_classifier_data
tests/unit/test_dataset_export.py::test_exported_classifier_metadata_and_balance_are_complete
tests/unit/test_dataset_export.py::test_classifier_benchmark_verifies_all_shards_against_manifest
tests/unit/test_dataset_export.py::test_classifier_loader_rejects_nonbinary_content
tests/unit/test_dataset_export.py::test_hex_word_rejects_noncanonical_text[ 123]
tests/unit/test_dataset_export.py::test_hex_word_rejects_noncanonical_text[+123]
tests/unit/test_dataset_export.py::test_hex_word_rejects_noncanonical_text[0x12]
tests/unit/test_dataset_export.py::test_hex_word_rejects_noncanonical_text[12-3]
tests/unit/test_dataset_export.py::test_exported_classifier_loads_without_the_generator
tests/unit/test_dataset_export.py::test_classifier_checksums_detect_a_tampered_shard
tests/unit/test_dataset_export.py::test_loaded_classifier_shard_matches_in_memory_generation
tests/unit/test_dataset_export.py::test_split_validation_still_requires_one_role_each
tests/unit/test_dataset_v1_freeze.py::test_classifier_freeze_matches_code_and_fixture
tests/unit/test_dataset_v1_freeze.py::test_attack_freeze_uses_plaintext_difference
tests/unit/test_dataset_v1_freeze.py::test_classifier_and_attack_remain_separate_frozen_datasets
tests/unit/test_demo_depth_dataset.py::test_dataset_sidecar_checksum_valid
tests/unit/test_demo_depth_dataset.py::test_cipher_known_answer_tests
tests/unit/test_demo_depth_dataset.py::test_dataset_structure_and_invariants
tests/unit/test_encryption_calibration.py::test_native_kats_all_prefixes_and_public_fixture
tests/unit/test_encryption_calibration.py::test_effective_cpu_policy_is_checked_without_timing
tests/unit/test_encryption_calibration.py::test_observer_deadline_is_passed_to_subprocess
tests/unit/test_encryption_calibration.py::test_pinned_source_is_the_actual_imported_module
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[0 7 1 2\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[0 7 1 2 3 4 5 6 7\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[2 7 1 2 3 4 5 6\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[0 23 1 2 3 4 5 6\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[1 33 1 2 3 4 5 6\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[0 7 -1 2 3 4 5 6\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[0 7 184467440737095516161 2 3 4 5 6\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_invalid_arguments_before_work[args0]
tests/unit/test_encryption_calibration.py::test_native_rejects_invalid_arguments_before_work[args1]
tests/unit/test_encryption_calibration.py::test_native_rejects_invalid_arguments_before_work[args2]
tests/unit/test_encryption_calibration.py::test_native_rejects_invalid_arguments_before_work[args3]
tests/unit/test_encryption_calibration.py::test_native_rejects_invalid_arguments_before_work[args4]
tests/unit/test_encryption_calibration.py::test_native_rejects_invalid_arguments_before_work[args5]
tests/unit/test_encryption_calibration.py::test_full_fixture_has_distinct_keys_and_blocks_and_known_digest
tests/unit/test_encryption_calibration.py::test_measured_conversion_uses_all_repetitions_and_one_profile_rate
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[elapsed_seconds-nan]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[elapsed_seconds--1]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[elapsed_seconds-1.99]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[batches-True]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[blocks-1]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[checksum-wrong]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[qos_class-9]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[nice-5]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[threads-8]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[peak_rss_bytes-0]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[peak_rss_bytes-100000000]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[identity_sha256-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[ordinal-3]
tests/unit/test_encryption_calibration.py::test_missing_duplicate_and_wrong_order_fail_closed
tests/unit/test_encryption_calibration.py::test_frozen_profile_drift_rejected[threads-True]
tests/unit/test_encryption_calibration.py::test_frozen_profile_drift_rejected[repetitions-5]
tests/unit/test_encryption_calibration.py::test_frozen_profile_drift_rejected[minimum_measurement_seconds-1]
tests/unit/test_encryption_calibration.py::test_frozen_profile_drift_rejected[compiler_flags-value3]
tests/unit/test_encryption_calibration.py::test_frozen_profile_drift_rejected[protected_data_access-True]
tests/unit/test_final_publication_figures.py::test_source_hash_mismatch_fails_closed
tests/unit/test_final_publication_figures.py::test_attack_duplicate_or_missing_pair_rejected[speck32_64]
tests/unit/test_final_publication_figures.py::test_attack_duplicate_or_missing_pair_rejected[simon32_64]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[verified_on_fresh_pairs-false]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[verified_on_fresh_pairs-False]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[complete_online_wall_seconds-0]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[complete_online_wall_seconds-nan]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[complete_online_wall_seconds-301]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[instance_id-another_key]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[outcome-fabricated]
tests/unit/test_final_publication_figures.py::test_speck_boolean_and_late_flags_rejected[counts_as_verified_recovery-false]
tests/unit/test_final_publication_figures.py::test_speck_boolean_and_late_flags_rejected[success-False]
tests/unit/test_final_publication_figures.py::test_speck_boolean_and_late_flags_rejected[late_verified_recovery-True]
tests/unit/test_final_publication_figures.py::test_paired_intervals_match_literature_fixtures[counts0-expected0]
tests/unit/test_final_publication_figures.py::test_paired_intervals_match_literature_fixtures[counts1-expected1]
tests/unit/test_final_publication_figures.py::test_failure_matrix_rejects_wrong_total_and_duplicate
tests/unit/test_final_publication_figures.py::test_actual_table_totals_provenance_and_replay
tests/unit/test_final_publication_figures.py::test_recorded_no_entry_is_zero_but_depth_timeout_is_unavailable
tests/unit/test_final_publication_figures.py::test_corrected_paired_ci_and_frozen_holm_families
tests/unit/test_final_publication_figures.py::test_backend_interruption_excluded_only_from_affected_pairs
tests/unit/test_final_publication_figures.py::test_diagnostic_grids_reject_duplicates_and_bad_flags[depth-depth_raw]
tests/unit/test_final_publication_figures.py::test_diagnostic_grids_reject_duplicates_and_bad_flags[backend-backend_raw]
tests/unit/test_final_publication_figures.py::test_main_figures_only_two_panels_and_paper_names
tests/unit/test_phase6b_proposal_figures.py::test_phase6b_summary_is_validation_only_and_complete
tests/unit/test_phase6b_proposal_figures.py::test_phase6b_result_figure_uses_selected_b1_metrics
tests/unit/test_phase6b_proposal_figures.py::test_phase6b_methodology_marks_planned_components
tests/unit/test_phase6b_proposal_figures.py::test_all_eight_proposal_figures_generate_valid_svg
tests/unit/test_phase6b_proposal_figures.py::test_proposal_figures_generation_in_temp_dir
tests/unit/test_phase6b_proposal_figures.py::test_phase6b_figure_manifest_is_reproducible
tests/unit/test_phase7_evaluation_figures.py::test_phase7_summary_is_complete_and_has_all_five_methods
tests/unit/test_phase7_evaluation_figures.py::test_phase7_figures_render_valid_svg_strings
tests/unit/test_phase7_evaluation_figures.py::test_phase7_latex_and_markdown_tables_are_valid
tests/unit/test_publication_evidence_package.py::test_relocation_retains_original_bytes_and_truth_separation
tests/unit/test_publication_evidence_package.py::test_nonportable_paths_fail_closed[.]
tests/unit/test_publication_evidence_package.py::test_nonportable_paths_fail_closed[../bad]
tests/unit/test_publication_evidence_package.py::test_nonportable_paths_fail_closed[/absolute]
tests/unit/test_publication_evidence_package.py::test_nonportable_paths_fail_closed[a/../b]
tests/unit/test_publication_evidence_package.py::test_nonportable_paths_fail_closed[a\\b]
tests/unit/test_publication_evidence_package.py::test_nonportable_paths_fail_closed[a//b]
tests/unit/test_publication_evidence_package.py::test_output_overwrite_and_source_tampering_are_rejected
tests/unit/test_publication_evidence_package.py::test_package_corruption_and_unexpected_files_are_rejected[missing]
tests/unit/test_publication_evidence_package.py::test_package_corruption_and_unexpected_files_are_rejected[extra]
tests/unit/test_publication_evidence_package.py::test_package_corruption_and_unexpected_files_are_rejected[altered]
tests/unit/test_publication_evidence_package.py::test_package_corruption_and_unexpected_files_are_rejected[symlink]
tests/unit/test_publication_evidence_package.py::test_package_corruption_and_unexpected_files_are_rejected[root_license]
tests/unit/test_publication_evidence_package.py::test_package_corruption_and_unexpected_files_are_rejected[source_license]
tests/unit/test_publication_evidence_package.py::test_license_provenance_detects_copies_changed_with_rehashed_file_manifest
tests/unit/test_publication_evidence_package.py::test_schema1_bundle_without_license_remains_verifiable
tests/unit/test_publication_evidence_package.py::test_build_requires_unchanged_committed_root_license_before_output[missing]
tests/unit/test_publication_evidence_package.py::test_build_requires_unchanged_committed_root_license_before_output[modified]
tests/unit/test_publication_evidence_package.py::test_build_requires_unchanged_committed_root_license_before_output[staged]
tests/unit/test_publication_evidence_package.py::test_build_requires_unchanged_committed_root_license_before_output[untracked]
tests/unit/test_publication_evidence_package.py::test_build_rechecks_license_before_creating_output
tests/unit/test_publication_evidence_package.py::test_portable_metadata_cannot_be_changed_even_if_local_hashes_updated
tests/unit/test_publication_evidence_package.py::test_duplicate_json_keys_rejected
tests/unit/test_publication_evidence_package.py::test_original_detached_sidecar_corruption_rejects_packaging
tests/unit/test_publication_evidence_package.py::test_formula_symlink_outside_pinned_directory_rejected
tests/unit/test_publication_evidence_package.py::test_scalar_namespace_loads_leaf_without_eager_training_exports
tests/unit/test_reference_encryption_report.py::test_inclusive_boundaries_and_failure_denominator
tests/unit/test_reference_encryption_report.py::test_invalid_cohort_record_rejected[id-b]
tests/unit/test_reference_encryption_report.py::test_invalid_cohort_record_rejected[success-false]
tests/unit/test_reference_encryption_report.py::test_invalid_cohort_record_rejected[wall-nan]
tests/unit/test_reference_encryption_report.py::test_invalid_cohort_record_rejected[wall-True]
tests/unit/test_reference_encryption_report.py::test_invalid_cohort_record_rejected[wall-0.0]
tests/unit/test_reference_encryption_report.py::test_invalid_cohort_record_rejected[wall-300.1]
tests/unit/test_reference_encryption_report.py::test_missing_record_rejected
tests/unit/test_seven_round_prospective_data.py::test_draw_csprng_seed_rejection_and_policy
tests/unit/test_seven_round_prospective_data.py::test_generation_config_hash_invariance
tests/unit/test_seven_round_prospective_data.py::test_prohibited_seeds_contain_all_critical_seeds
tests/unit/test_seven_round_prospective_data.py::test_small_cohort_generation_and_deterministic_replay
tests/unit/test_simon_attack_instance.py::test_demo_instance_generation_is_deterministic
tests/unit/test_simon_attack_instance.py::test_input_difference_structure_and_encryption
tests/unit/test_simon_attack_instance.py::test_fresh_verification_pairs_are_independently_valid
tests/unit/test_simon_attack_instance.py::test_generator_rejects_attack_and_verification_plaintext_collisions
tests/unit/test_simon_attack_instance.py::test_instance_rejects_overlap_after_deserialization_or_manual_construction
tests/unit/test_simon_attack_instance.py::test_zero_effective_input_difference_is_rejected[input_difference0]
tests/unit/test_simon_attack_instance.py::test_zero_effective_input_difference_is_rejected[input_difference1]
tests/unit/test_simon_attack_instance.py::test_impossible_unique_plaintext_request_is_rejected
tests/unit/test_simon_attack_instance.py::test_opened_demo_instances_replay_unchanged[8-8a762dee8ab2f4d0e8e96c10f9f3fb3e00a681fa419bfd18104570cadb1f16e8]
tests/unit/test_simon_attack_instance.py::test_opened_demo_instances_replay_unchanged[9-b294ee5fed9e0b6ae79437ef976da67b5e061b5f3318858948fd9bfcea98bca4]
tests/unit/test_simon_attack_instance.py::test_opened_demo_instances_replay_unchanged[10-e4e9b03e21a67c28d7764ab8e759f6c7bb2d8377f32c89e164946e45b3357718]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[98001]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[97001]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[99001]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[100001]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[101001]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[300001]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[406737]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[669981]
tests/unit/test_simon_attack_instance.py::test_role_governance_rejects_unauthorized_roles
tests/unit/test_simon_attack_instance.py::test_demo_role_requires_cli_demo_seed
tests/unit/test_simon_attack_instance.py::test_development_role_cannot_relabel_opened_demo_seed
tests/unit/test_simon_attack_instance.py::test_zero_or_negative_fresh_verification_count_is_rejected[0]
tests/unit/test_simon_attack_instance.py::test_zero_or_negative_fresh_verification_count_is_rejected[-1]
tests/unit/test_simon_attack_instance.py::test_noninteger_verification_count_is_rejected[True]
tests/unit/test_simon_attack_instance.py::test_noninteger_verification_count_is_rejected[1.5]
tests/unit/test_simon_attack_instance.py::test_noninteger_verification_count_is_rejected[2]
tests/unit/test_simon_attack_instance.py::test_ground_truth_isolation_properties
tests/unit/test_simon_attack_instance.py::test_serialization_round_trip
tests/unit/test_simon_higher_round.py::test_simons11_schedule_word_order_and_chronological_trailing_window
tests/unit/test_simon_higher_round.py::test_peel_order_matches_independent_reduced_round_reference
tests/unit/test_simon_higher_round.py::test_queue_order_ties_are_deterministic_and_duplicate_paths_keep_first
tests/unit/test_simon_higher_round.py::test_native_simon11_completion_checks_128_known_blocks_and_two_fresh_pairs
tests/unit/test_simon_higher_round.py::test_run_cell_success_uses_stubbed_rankers_and_fresh_verification
tests/unit/test_simon_higher_round.py::test_run_cell_rejects_completion_that_fails_fresh_verification
tests/unit/test_simon_higher_round.py::test_run_cell_stops_dispatch_after_ranking_crosses_deadline
tests/unit/test_simon_higher_round.py::test_run_cell_does_not_accept_verification_completed_after_deadline
tests/unit/test_simon_higher_round.py::test_run_cell_rejects_malformed_instance_before_ranking[rounds]
tests/unit/test_simon_higher_round.py::test_run_cell_rejects_malformed_instance_before_ranking[pairs]
tests/unit/test_simon_higher_round.py::test_run_cell_rejects_malformed_instance_before_ranking[verification_pairs]
tests/unit/test_simon_higher_round.py::test_run_cell_rejects_malformed_instance_before_ranking[overlap]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_accepts_only_demo_without_explicit_role
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-98001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-97001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-99001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-101001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-201001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-300001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-406737]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-669981]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-6503664708853956672]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-701004]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-98001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-97001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-99001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-101001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-201001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-300001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-406737]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-669981]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-6503664708853956672]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-701004]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-98001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-97001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-99001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-101001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-201001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-300001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-406737]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-669981]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-6503664708853956672]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-701004]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-98001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-97001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-99001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-101001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-201001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-300001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-406737]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-669981]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-6503664708853956672]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-701004]
tests/unit/test_simon_higher_round.py::test_parse_args_requires_run_parameters_and_limits_tensorflow_threads
tests/unit/test_simon_higher_round.py::test_ledger_creation_is_exclusive_and_append_requires_existing_file
tests/unit/test_simon_higher_round.py::test_classical_and_hybrid_stages_batch_parents_once[S11-C]
tests/unit/test_simon_higher_round.py::test_classical_and_hybrid_stages_batch_parents_once[S11-H]
tests/unit/test_simon_higher_round.py::test_row_score_ties_use_word_order_and_nonfinite_scores_fail
tests/unit/test_simon_higher_round.py::test_bad_table_header_is_rejected_without_dense_allocation
tests/unit/test_simon_higher_round.py::test_batch_rows_reject_invalid_probabilities[-1.0]
tests/unit/test_simon_higher_round.py::test_batch_rows_reject_invalid_probabilities[nan]
tests/unit/test_simon_higher_round.py::test_batch_rows_reject_invalid_probabilities[1.01]
tests/unit/test_simon_higher_round.py::test_header_pins_sources_and_only_assets_used_by_selected_arm
tests/unit/test_simon_higher_round.py::test_rejected_cli_never_draws_data_or_initializes_workers[existing_ledger]
tests/unit/test_simon_higher_round.py::test_rejected_cli_never_draws_data_or_initializes_workers[protected_seed]
tests/unit/test_simon_higher_round.py::test_rejected_cli_never_draws_data_or_initializes_workers[missing_threads]
tests/unit/test_simon_higher_round.py::test_rejected_cli_never_draws_data_or_initializes_workers[role_specific_seed]
tests/unit/test_simon_higher_round.py::test_rejected_cli_never_draws_data_or_initializes_workers[unfrozen_role]
tests/unit/test_simon_higher_round.py::test_loader_time_is_charged_and_no_rank_starts_at_exact_deadline
tests/unit/test_simon_higher_round.py::test_vv10_pin_registration_is_scoped_to_one_load
tests/unit/test_simon_higher_round.py::test_serial_schedule_rotates_by_absolute_key_index_with_stubbed_cells
tests/unit/test_simon_higher_round.py::test_wrong_input_difference_is_rejected_before_model_load
tests/unit/test_simon_higher_round.py::test_posthoc_truth_time_does_not_turn_ontime_recovery_into_deadline
tests/unit/test_simon_sat_300s_control.py::test_worker_receives_only_attack_and_verification_observations
tests/unit/test_simon_sat_300s_control.py::test_worker_classifies_limit_as_unresolved
tests/unit/test_simon_sat_300s_control.py::test_worker_rejects_truth_bearing_or_wrong_sized_payload
tests/unit/test_simon_sat_300s_control.py::test_parent_watchdog_is_unconditional_failure
tests/unit/test_simon_sat_300s_control.py::test_resume_refuses_source_drift_and_duplicate_records

314 tests collected in 17.67s

Exit code: 0
Elapsed seconds: 20.703

```


```text
COMMAND: <work>/hamza-venv/Scripts/python.exe -m pytest --collect-only -q tests
tests/integration/test_dataset_v1_fixture.py::test_classifier_fixture_matches_the_generator
tests/integration/test_dataset_v1_fixture.py::test_attack_fixture_matches_the_generator
tests/integration/test_dataset_v1_fixture.py::test_classifier_and_attack_fixtures_are_separate_schemas
tests/integration/test_dataset_v1_fixture.py::test_classifier_fixture_can_be_loaded_as_plain_json
tests/integration/test_phase5_recovery.py::test_candidate_recovery_verifies_the_decoded_master_key
tests/integration/test_phase5_recovery.py::test_sat_only_recovery_requires_independent_verification
tests/integration/test_phase5_recovery.py::test_random_candidate_order_is_seeded_without_hidden_state
tests/integration/test_phase5_recovery.py::test_candidate_recovery_rejects_invalid_orders[candidate_order0]
tests/integration/test_phase5_recovery.py::test_candidate_recovery_rejects_invalid_orders[candidate_order1]
tests/integration/test_phase5_recovery.py::test_candidate_recovery_rejects_invalid_orders[candidate_order2]
tests/integration/test_phase5_recovery.py::test_candidate_recovery_rejects_invalid_orders[candidate_order3]
tests/integration/test_sat_model.py::test_fixed_key_model_states_match_the_python_reference
tests/integration/test_sat_model.py::test_final_round_subkey_assumptions_follow_the_key_schedule
tests/integration/test_sat_model.py::test_final_round_assumptions_are_temporary
tests/integration/test_sat_model.py::test_final_round_subkey_assumption_rejects_invalid_values[True]
tests/integration/test_sat_model.py::test_final_round_subkey_assumption_rejects_invalid_values[False]
tests/integration/test_sat_model.py::test_final_round_subkey_assumption_rejects_invalid_values[-1]
tests/integration/test_sat_model.py::test_final_round_subkey_assumption_rejects_invalid_values[65536]
tests/integration/test_sat_model.py::test_final_round_subkey_assumption_rejects_invalid_values[1.0]
tests/integration/test_sat_model.py::test_unknown_key_recovery_produces_a_verifiable_candidate
tests/integration/test_sat_model.py::test_modified_ciphertext_is_unsatisfiable_under_the_true_key
tests/integration/test_sat_model.py::test_blocking_clause_exposes_multiple_solutions_when_key_is_underdetermined
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[0-3-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[0-4-3-5]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[0-6-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[0-6-63-7]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[1-3-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[1-4-3-5]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[1-6-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[1-6-63-7]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[3-3-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[3-4-3-5]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[3-6-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[3-6-63-7]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[6-3-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[6-4-3-5]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[6-6-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[6-6-63-7]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[7-3-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[7-4-3-5]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[7-6-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[7-6-63-7]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_full_width_rejects_unbounded_dense_round
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_record_cap_fails_without_silent_truncation
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_rejects_zero_difference_and_existing_output
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_rejects_malformed_word_width[16junk]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_rejects_malformed_word_width[-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_rejects_malformed_word_width[4294967312]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_full_width_three_rounds_matches_source_style_transition
tests/integration/test_simon_predicted_ddt_cpp.py::test_dense_seven_round_table_matches_independent_complete_oracle[4-3-5]
tests/integration/test_simon_predicted_ddt_cpp.py::test_dense_seven_round_table_matches_independent_complete_oracle[6-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_dense_seven_round_table_matches_independent_complete_oracle[6-63-7]
tests/integration/test_simon_predicted_ddt_cpp.py::test_dense_generator_rejects_corrupted_sparse_input
tests/integration/test_simon_published_vv.py::test_published_vv_loader_rejects_unapproved_round
tests/integration/test_simon_published_vv.py::test_safe_checkpoint_load_matches_all_original_weights_and_clip_formula[7]
tests/integration/test_simon_published_vv.py::test_safe_checkpoint_load_matches_all_original_weights_and_clip_formula[8]
tests/integration/test_simon_published_vv.py::test_safe_checkpoint_load_matches_all_original_weights_and_clip_formula[9]
tests/integration/test_simon_published_vv.py::test_published_vv_loader_rejects_tampered_file
tests/integration/test_tiny_dataset_fixture.py::test_tiny_fixture_matches_the_generator
tests/test_classical_ranking.py::test_round1_analytical_derivation_and_scalar_equivalence
tests/test_classical_ranking.py::test_scalar_vs_vectorized_generator_equivalence
tests/test_classical_ranking.py::test_round2_analytical_dyadic_probabilities
tests/test_classical_ranking.py::test_msb_symmetry_and_tie_intervals
tests/test_classical_ranking.py::test_candidate_subset_ranking
tests/test_classical_ranking.py::test_deterministic_replay
tests/test_classical_ranking.py::test_distribution_table_validation_and_serialization
tests/test_classical_ranking.py::test_invalid_table_construction
tests/test_queue_cost_model.py::test_km_without_censoring_matches_hand_computation
tests/test_queue_cost_model.py::test_km_right_censoring_shrinks_the_risk_set
tests/test_queue_cost_model.py::test_km_censored_tie_with_event_stays_at_risk
tests/test_queue_cost_model.py::test_km_all_censored_samples_infinity
tests/test_queue_cost_model.py::test_km_quantile_is_infinite_when_median_is_not_reached
tests/test_queue_cost_model.py::test_km_sample_maps_uniforms_through_the_cdf
tests/test_queue_cost_model.py::test_p_recover_is_zero_for_absent_candidate
tests/test_queue_cost_model.py::test_p_recover_is_one_when_clearly_within_budget
tests/test_queue_cost_model.py::test_p_recover_is_zero_when_preparation_alone_exceeds_budget
tests/test_queue_cost_model.py::test_true_candidate_slower_than_cap_is_lost
tests/test_queue_cost_model.py::test_wrong_candidate_costs_are_truncated_at_tau
tests/test_queue_cost_model.py::test_leave_one_key_out_excludes_that_key
tests/test_queue_cost_model.py::test_leave_one_key_out_falls_back_to_zero_overhead_when_none_left
tests/test_queue_cost_model.py::test_evidence_predictions_and_summary_match_audit_counts
tests/test_verify_p0_p2_artifacts.py::test_resolve_json_pointer
tests/test_verify_p0_p2_artifacts.py::test_valid_manifest_and_claim
tests/test_verify_p0_p2_artifacts.py::test_missing_claim_id
tests/test_verify_p0_p2_artifacts.py::test_duplicate_claim_id
tests/test_verify_p0_p2_artifacts.py::test_stale_schema_rejection
tests/test_verify_p0_p2_artifacts.py::test_hash_mismatch_rejection
tests/test_verify_p0_p2_artifacts.py::test_value_mismatch_rejection
tests/test_verify_p0_p2_artifacts.py::test_prohibited_wording_rejection
tests/unit/test_attack_instances.py::test_attack_generation_is_identical_for_a_fixed_seed
tests/unit/test_attack_instances.py::test_different_attack_seeds_produce_different_instances
tests/unit/test_attack_instances.py::test_attack_pairs_share_one_key_and_the_selected_difference
tests/unit/test_attack_instances.py::test_attack_split_keys_do_not_overlap
tests/unit/test_attack_instances.py::test_attack_export_checksums_and_independent_load
tests/unit/test_attack_instances.py::test_attack_loader_rejects_tampered_ciphertext
tests/unit/test_attack_instances.py::test_invalid_attack_configuration_is_rejected[0-2-7-train]
tests/unit/test_attack_instances.py::test_invalid_attack_configuration_is_rejected[2-0-7-train]
tests/unit/test_attack_instances.py::test_invalid_attack_configuration_is_rejected[2-2-0-train]
tests/unit/test_attack_instances.py::test_invalid_attack_configuration_is_rejected[2-2-23-train]
tests/unit/test_attack_instances.py::test_invalid_attack_configuration_is_rejected[2-2-7-holdout]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-observation]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[truth-leaked-into-public]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-trailing-key-unit]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-master-truth]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-verification-pair]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-variable-map]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-bit-order]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-native-formula]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-pure-cnf-formula]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_forged_catalog_metadata[counts]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_forged_catalog_metadata[known-observations]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_forged_catalog_metadata[assumption-units]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_accepts_canonical_minimal_bundle
tests/unit/test_backend_artifact_semantics.py::test_catalog_observations_are_checked_against_formula_not_file_hash_only
tests/unit/test_backend_artifacts.py::test_solver_public_boundary_and_fresh_verification[speck32_64-7]
tests/unit/test_backend_artifacts.py::test_solver_public_boundary_and_fresh_verification[simon32_64-10]
tests/unit/test_backend_completed_evidence.py::test_signed_xor_parity_and_unit_clause[x1 2 0]
tests/unit/test_backend_completed_evidence.py::test_signed_xor_parity_and_unit_clause[x 1 2 0]
tests/unit/test_backend_completed_evidence.py::test_negative_xor_literal_changes_parity
tests/unit/test_backend_completed_evidence.py::test_unsatisfied_disjunction_or_empty_constraint_is_rejected[1 -2 0]
tests/unit/test_backend_completed_evidence.py::test_unsatisfied_disjunction_or_empty_constraint_is_rejected[0]
tests/unit/test_backend_completed_evidence.py::test_unsatisfied_disjunction_or_empty_constraint_is_rejected[x0]
tests/unit/test_backend_completed_evidence.py::test_incomplete_nonboolean_and_out_of_range_assignments_are_rejected[assignment0]
tests/unit/test_backend_completed_evidence.py::test_incomplete_nonboolean_and_out_of_range_assignments_are_rejected[assignment1]
tests/unit/test_backend_completed_evidence.py::test_incomplete_nonboolean_and_out_of_range_assignments_are_rejected[assignment2]
tests/unit/test_backend_completed_evidence.py::test_malformed_header_range_or_record_is_rejected[p cnf 2 2\n1 0\n]
tests/unit/test_backend_completed_evidence.py::test_malformed_header_range_or_record_is_rejected[p cnf 2 1\n3 0\n]
tests/unit/test_backend_completed_evidence.py::test_malformed_header_range_or_record_is_rejected[p cnf 2 1\n1 0 2 0\n]
tests/unit/test_backend_completed_evidence.py::test_decode_is_lsb_first_and_uses_variable_map_order
tests/unit/test_backend_completed_evidence.py::test_forced_extension_handles_negative_xor_and_auxiliary_clause
tests/unit/test_backend_completed_evidence.py::test_contradictory_reference_witness_is_rejected[cnf0-xors0]
tests/unit/test_backend_completed_evidence.py::test_contradictory_reference_witness_is_rejected[cnf1-xors1]
tests/unit/test_backend_completed_evidence.py::test_unforced_reference_extension_is_unavailable_not_filled_arbitrarily
tests/unit/test_backend_environment.py::test_power_observation_requires_successful_ac_and_sleep[Now drawing from 'AC Power'-PreventSystemSleep 1-0-True]
tests/unit/test_backend_environment.py::test_power_observation_requires_successful_ac_and_sleep[Now drawing from 'Battery Power'-PreventSystemSleep 1-0-False]
tests/unit/test_backend_environment.py::test_power_observation_requires_successful_ac_and_sleep[Now drawing from 'AC Power'-PreventSystemSleep 0-0-False]
tests/unit/test_backend_environment.py::test_power_observation_requires_successful_ac_and_sleep[Now drawing from 'AC Power'-PreventSystemSleep 1-1-False]
tests/unit/test_backend_environment.py::test_unavailable_power_query_rejects
tests/unit/test_backend_environment.py::test_process_policy_does_not_substitute_for_thread_qos[0-0-0-True]
tests/unit/test_backend_environment.py::test_process_policy_does_not_substitute_for_thread_qos[0-1-0-False]
tests/unit/test_backend_environment.py::test_process_policy_does_not_substitute_for_thread_qos[1-0-0-False]
tests/unit/test_backend_environment.py::test_process_policy_does_not_substitute_for_thread_qos[0-0-3-False]
tests/unit/test_backend_environment.py::test_unsupported_platform_is_explicit
tests/unit/test_backend_evidence.py::test_audit_reparses_complete_native_sat_model
tests/unit/test_backend_evidence.py::test_audit_accepts_native_unsat_as_recorded_alarm
tests/unit/test_backend_evidence.py::test_wrapper_audit_does_not_claim_a_captured_native_model
tests/unit/test_backend_evidence.py::test_late_worker_recovery_is_preserved_but_not_credited
tests/unit/test_backend_evidence.py::test_interruption_reports_unavailable_time_without_normal_receipts
tests/unit/test_backend_evidence.py::test_missing_environment_result_requires_parent_stop_proof[coordinator_stop-True]
tests/unit/test_backend_evidence.py::test_missing_environment_result_requires_parent_stop_proof[coordinator_stop-False]
tests/unit/test_backend_evidence.py::test_missing_environment_result_requires_parent_stop_proof[worker_exit-True]
tests/unit/test_backend_evidence.py::test_coordinator_record_mutation_is_rejected
tests/unit/test_backend_evidence.py::test_nonzero_worker_exit_receipt_is_rejected_even_when_resealed
tests/unit/test_backend_evidence.py::test_unsealed_or_tampered_raw_solver_output_is_rejected
tests/unit/test_backend_evidence.py::test_guardian_must_itself_certify_group_absence[False]
tests/unit/test_backend_evidence.py::test_guardian_must_itself_certify_group_absence[None]
tests/unit/test_backend_evidence.py::test_guardian_must_itself_certify_group_absence[true]
tests/unit/test_backend_evidence.py::test_resealed_solver_output_must_still_match_worker_status
tests/unit/test_backend_evidence.py::test_raw_worker_key_must_match_coordinator_result
tests/unit/test_backend_evidence.py::test_worker_directory_cannot_be_redirected_or_symlinked
tests/unit/test_backend_evidence.py::test_raw_artifact_paths_must_be_basenames
tests/unit/test_backend_evidence.py::test_interrupted_record_must_preserve_original_group_and_invocation
tests/unit/test_backend_formula.py::test_formula_has_independent_reference_witness_and_chronological_assumptions[speck32_64-4-key0]
tests/unit/test_backend_formula.py::test_formula_has_independent_reference_witness_and_chronological_assumptions[simon32_64-5-key1]
tests/unit/test_backend_formula.py::test_unguided_build_never_reads_truth_or_verification_fields[speck32_64-4-key0]
tests/unit/test_backend_formula.py::test_unguided_build_never_reads_truth_or_verification_fields[simon32_64-5-key1]
tests/unit/test_backend_formula.py::test_observation_prefix_does_not_read_unselected_pairs
tests/unit/test_backend_formula.py::test_reference_key_conflicts_with_corrupted_known_ciphertext
tests/unit/test_backend_formula.py::test_speck_native_output_matches_existing_encoder_bytes
tests/unit/test_backend_formula.py::test_speck_conditioned_output_matches_existing_pinned_key_units
tests/unit/test_backend_formula.py::test_simon_native_output_matches_existing_recorder_bytes
tests/unit/test_backend_formula.py::test_simon_conditioned_output_matches_existing_assumption_serializer
tests/unit/test_backend_formula.py::test_simon_builder_never_constructs_solver
tests/unit/test_backend_formula.py::test_native_only_build_skips_pure_cnf_conversion
tests/unit/test_backend_formula.py::test_renderer_preserves_signed_rhs_and_empty_xor_semantics
tests/unit/test_backend_formula.py::test_long_xor_conversion_allocates_fresh_auxiliaries_repeatably
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[0-0-fields0]
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[3-0-fields1]
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[6-0-fields2]
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[2--1-fields3]
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[2-5-fields4]
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[2-1-fields5]
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[2-1-fields6]
tests/unit/test_backend_formula.py::test_invalid_rounds_and_observation_words_fail_closed
tests/unit/test_backend_formula.py::test_repeated_builds_are_deterministic_and_do_not_mutate_inputs
tests/unit/test_backend_ranking_depth_figure.py::test_plot_totals_agree_with_independent_raw_audit
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[duplicate]
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[missing]
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[string_flag]
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[wrong_outcome]
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[late]
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[nan]
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[premature]
tests/unit/test_backend_ranking_depth_figure.py::test_corrupted_sidecar_rejected
tests/unit/test_backend_ranking_depth_n32_figure.py::test_canonical_source_hash_and_sidecar
tests/unit/test_backend_ranking_depth_n32_figure.py::test_n32_plot_totals_and_par2_agree_with_audit
tests/unit/test_backend_ranking_depth_n32_figure.py::test_containment_invariants
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[duplicate_cell]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[missing_cell]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[string_flag]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[wrong_outcome_recovered_but_false]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[wrong_outcome_unrecovered_but_true]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[unexpected_unrecovered_outcome]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[late_recovery_over_300s]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[premature_deadline_under_300s]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[nan_wall_time]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[negative_wall_time]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[unmeasured_containment]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[containment_key_mismatch]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[wrong_cell_count]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_corrupted_sidecar_or_hash_rejected
tests/unit/test_backend_ranking_depth_n32_figure.py::test_generate_end_to_end_in_tmp_path
tests/unit/test_backend_solver_io.py::test_native_gauss_profiles_differ_only_by_matrix_limit
tests/unit/test_backend_solver_io.py::test_native_default_keeps_cms_defaults_and_pins_one_thread
tests/unit/test_backend_solver_io.py::test_cnf_profile_changes_only_input_representation_from_x
tests/unit/test_backend_solver_io.py::test_cadical_uses_quiet_mode_and_preserves_path_as_one_argument
tests/unit/test_backend_solver_io.py::test_solver_arguments_reject_unknown_profiles_and_missing_paths[unknown-/bin/cms-/bin/cadical-formula.cnf]
tests/unit/test_backend_solver_io.py::test_solver_arguments_reject_unknown_profiles_and_missing_paths[cms_native_default--/bin/cadical-formula.cnf]
tests/unit/test_backend_solver_io.py::test_solver_arguments_reject_unknown_profiles_and_missing_paths[cadical_cnf-/bin/cms--formula.cnf]
tests/unit/test_backend_solver_io.py::test_solver_arguments_reject_unknown_profiles_and_missing_paths[cms_native_default-/bin/cms-/bin/cadical-]
tests/unit/test_backend_solver_io.py::test_sat_model_can_span_lines_and_returns_raw_gauss_diagnostics
tests/unit/test_backend_solver_io.py::test_cadical_signed_dimacs_model_and_comments
tests/unit/test_backend_solver_io.py::test_empty_unsat_result_has_empty_assignment[cms]
tests/unit/test_backend_solver_io.py::test_empty_unsat_result_has_empty_assignment[cadical]
tests/unit/test_backend_solver_io.py::test_unknown_outcomes[cms-15-s INDETERMINATE\n]
tests/unit/test_backend_solver_io.py::test_unknown_outcomes[cadical-0-s UNKNOWN\n]
tests/unit/test_backend_solver_io.py::test_unknown_needs_real_exit_semantics_and_explicit_output[cms-0-s UNKNOWN\n]
tests/unit/test_backend_solver_io.py::test_unknown_needs_real_exit_semantics_and_explicit_output[cadical-0-]
tests/unit/test_backend_solver_io.py::test_exit_code_status_disagreement_is_error
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-v 1 0\n-required0-1-no explicit solver status]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\ns SATISFIABLE\nv 1 0\n-required1-1-repeated]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 1 0 2\n-required2-2-after terminal zero]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 1 0\nv -1 0\n-required3-1-after terminal zero]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 1 -1 0\n-required4-1-conflicting]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 1 1_0 0\n-required5-10-non-integer]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 4 0\n-required6-3-outside]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 1 -2\n-required7-2-truncated]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 1 0\n-required8-2-omits required]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[20-s UNSATISFIABLE\nv 0\n-required9-0-UNSAT result unexpectedly]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[0-s UNKNOWN\nv 1 0\n-required10-1-UNKNOWN result unexpectedly]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE extra\nv 1 0\n-required11-1-malformed status]
tests/unit/test_backend_solver_io.py::test_invalid_requested_variable_sets_are_rejected[required0-2]
tests/unit/test_backend_solver_io.py::test_invalid_requested_variable_sets_are_rejected[required1-2]
tests/unit/test_backend_solver_io.py::test_invalid_requested_variable_sets_are_rejected[required2-2]
tests/unit/test_backend_solver_io.py::test_invalid_requested_variable_sets_are_rejected[required3-2]
tests/unit/test_backend_solver_io.py::test_invalid_requested_variable_sets_are_rejected[required4--1]
tests/unit/test_backend_solver_io.py::test_unknown_engine_and_unexpected_exit_code_are_errors
tests/unit/test_beam_adjudication.py::test_seed_governance_prohibitions
tests/unit/test_beam_adjudication.py::test_draw_csprng_seed_returns_valid_seed
tests/unit/test_beam_adjudication.py::test_configuration_parameter_invariants
tests/unit/test_beam_adjudication.py::test_counterbalancing_schedule
tests/unit/test_beam_adjudication.py::test_selection_rule_tier1_higher_recovery_wins
tests/unit/test_beam_adjudication.py::test_selection_rule_tier2_lower_par2_wins_on_tie
tests/unit/test_beam_adjudication.py::test_selection_rule_tier4_reference_a_retained_on_complete_tie
tests/unit/test_beam_adjudication.py::test_exact_mcnemar_on_discordant_pairs
tests/unit/test_beam_adjudication.py::test_mismatched_instance_sets_rejected
tests/unit/test_benchmark_formula_export.py::test_benchmark_formula_manifest_integrity
tests/unit/test_benchmark_formula_export.py::test_speck_subcritical_formula_solvable_by_cadical
tests/unit/test_bruteforce_completion.py::test_fixtures_match_task_known_answer_values
tests/unit/test_bruteforce_completion.py::test_official_known_answers_through_c_encryption
tests/unit/test_bruteforce_completion.py::test_official_known_answers_recovered_by_completion
tests/unit/test_bruteforce_completion.py::test_schedule_inversion_round_trip[speck-rounds_range0]
tests/unit/test_bruteforce_completion.py::test_schedule_inversion_round_trip[simon-rounds_range1]
tests/unit/test_bruteforce_completion.py::test_arbitrary_windows_invert_consistently[speck-22]
tests/unit/test_bruteforce_completion.py::test_arbitrary_windows_invert_consistently[simon-32]
tests/unit/test_bruteforce_completion.py::test_full_m3_space_visits_distinct_master_keys[speck-8]
tests/unit/test_bruteforce_completion.py::test_full_m3_space_visits_distinct_master_keys[simon-11]
tests/unit/test_bruteforce_completion.py::test_expand_and_encrypt_agree_with_python_reference[speck-22]
tests/unit/test_bruteforce_completion.py::test_expand_and_encrypt_agree_with_python_reference[simon-32]
tests/unit/test_bruteforce_completion.py::test_speck_expand_agrees_with_gohr_convention
tests/unit/test_bruteforce_completion.py::test_full_space_m3_completion_finds_true_key[speck-8]
tests/unit/test_bruteforce_completion.py::test_full_space_m3_completion_finds_true_key[simon-11]
tests/unit/test_bruteforce_completion.py::test_narrow_ranges_for_partial_trailing_keys[speck-8-2]
tests/unit/test_bruteforce_completion.py::test_narrow_ranges_for_partial_trailing_keys[speck-8-1]
tests/unit/test_bruteforce_completion.py::test_narrow_ranges_for_partial_trailing_keys[simon-11-2]
tests/unit/test_bruteforce_completion.py::test_narrow_ranges_for_partial_trailing_keys[simon-11-1]
tests/unit/test_bruteforce_completion.py::test_wrong_trailing_key_is_rejected_over_full_space[speck-8]
tests/unit/test_bruteforce_completion.py::test_wrong_trailing_key_is_rejected_over_full_space[simon-11]
tests/unit/test_bruteforce_completion.py::test_every_block_is_verified_not_only_the_first
tests/unit/test_bruteforce_completion.py::test_invalid_arguments_are_rejected
tests/unit/test_bruteforce_completion.py::test_build_uses_frozen_flags_and_produces_library
tests/unit/test_bruteforce_completion.py::test_quick_benchmark_cli_writes_complete_record
tests/unit/test_cipher.py::test_one_round_encryption_matches_regression_state
tests/unit/test_cipher.py::test_one_round_decryption_is_the_inverse
tests/unit/test_cipher.py::test_full_known_answer_vector
tests/unit/test_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[1-expected0]
tests/unit/test_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[2-expected1]
tests/unit/test_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[7-expected2]
tests/unit/test_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[11-expected3]
tests/unit/test_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[22-expected4]
tests/unit/test_cipher.py::test_deterministic_random_round_trips
tests/unit/test_cipher.py::test_round_and_block_inputs_are_explicitly_masked
tests/unit/test_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block-0]
tests/unit/test_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block--1]
tests/unit/test_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block-23]
tests/unit/test_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block-0]
tests/unit/test_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block--1]
tests/unit/test_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block-23]
tests/unit/test_cipher.py::test_non_integer_round_counts_are_rejected[encrypt_block-1.0]
tests/unit/test_cipher.py::test_non_integer_round_counts_are_rejected[encrypt_block-True]
tests/unit/test_cipher.py::test_non_integer_round_counts_are_rejected[decrypt_block-1.0]
tests/unit/test_cipher.py::test_non_integer_round_counts_are_rejected[decrypt_block-True]
tests/unit/test_cipher.py::test_malformed_round_state_is_rejected
tests/unit/test_compute_mcnemar_power.py::test_exact_binomial_twoside_pvalue_symmetry
tests/unit/test_compute_mcnemar_power.py::test_compute_exact_mcnemar_power_null_hypothesis
tests/unit/test_compute_mcnemar_power.py::test_compute_exact_mcnemar_power_monotonicity
tests/unit/test_compute_mcnemar_power.py::test_compute_exact_mcnemar_power_invalid_inputs
tests/unit/test_compute_mcnemar_power.py::test_generate_sensitivity_grid
tests/unit/test_compute_mcnemar_power.py::test_parse_args
tests/unit/test_conditioning_final_audit.py::test_original_formula_check_rejects_parity_and_conditioning_errors
tests/unit/test_conditioning_final_audit.py::test_duplicate_json_fields_are_rejected
tests/unit/test_conditioning_final_audit.py::test_checksum_and_sidecar_filename_are_both_required
tests/unit/test_conditioning_final_audit.py::test_summary_keeps_failures_and_recomputes_paired_denominators
tests/unit/test_conditioning_worker.py::test_production_solver_factory_uses_the_fixed_wrapper_options
tests/unit/test_conditioning_worker.py::test_worker_loads_identical_formula_and_changes_only_conditioning_interface[permanent_units-speck32_64]
tests/unit/test_conditioning_worker.py::test_worker_loads_identical_formula_and_changes_only_conditioning_interface[permanent_units-simon32_64]
tests/unit/test_conditioning_worker.py::test_worker_loads_identical_formula_and_changes_only_conditioning_interface[temporary_assumptions-speck32_64]
tests/unit/test_conditioning_worker.py::test_worker_loads_identical_formula_and_changes_only_conditioning_interface[temporary_assumptions-simon32_64]
tests/unit/test_conditioning_worker.py::test_temporary_assumptions_are_removed_and_permanent_units_persist
tests/unit/test_conditioning_worker.py::test_reference_known_check_uses_cipher_and_key_schedule[speck32_64]
tests/unit/test_conditioning_worker.py::test_reference_known_check_uses_cipher_and_key_schedule[simon32_64]
tests/unit/test_conditioning_worker.py::test_invalid_model_is_correctness_alarm_and_truth_is_not_read
tests/unit/test_conditioning_worker.py::test_known_check_failure_stops_before_truth_load
tests/unit/test_conditioning_worker.py::test_valid_sat_model_without_fresh_match_is_retained_as_nonunique_outcome
tests/unit/test_conditioning_worker.py::test_truth_hash_is_checked_only_after_known_model_check
tests/unit/test_conditioning_worker.py::test_true_key_unsat_is_a_distinct_correctness_alarm
tests/unit/test_conditioning_worker.py::test_environment_rejection_keeps_initial_telemetry_and_power_sample
tests/unit/test_conditioning_worker.py::test_thread_environment_mismatch_stops_before_solver_certification
tests/unit/test_conditioning_worker.py::test_invalid_fixed_cell_is_rejected_before_environment_or_solver[rounds-True]
tests/unit/test_conditioning_worker.py::test_invalid_fixed_cell_is_rejected_before_environment_or_solver[key_index-8]
tests/unit/test_conditioning_worker.py::test_invalid_fixed_cell_is_rejected_before_environment_or_solver[observations-2]
tests/unit/test_conditioning_worker.py::test_invalid_fixed_cell_is_rejected_before_environment_or_solver[trailing_keys-0]
tests/unit/test_conditioning_worker.py::test_invalid_fixed_cell_is_rejected_before_environment_or_solver[arm-unknown]
tests/unit/test_conditioning_worker.py::test_main_reads_checked_payload_and_atomically_persists_result
tests/unit/test_confidence.py::test_features_use_ordered_scores_and_have_expected_values
tests/unit/test_confidence.py::test_serialized_features_and_target_membership_are_deterministic
tests/unit/test_confidence.py::test_pair_subset_stability_is_exposed_when_present
tests/unit/test_confidence.py::test_tied_scores_keep_the_supplied_deterministic_order
tests/unit/test_confidence.py::test_malformed_rankings_are_rejected[candidate_values0-candidate_scores0-candidate_order0]
tests/unit/test_confidence.py::test_malformed_rankings_are_rejected[candidate_values1-candidate_scores1-candidate_order1]
tests/unit/test_confidence.py::test_malformed_rankings_are_rejected[candidate_values2-candidate_scores2-candidate_order2]
tests/unit/test_confidence.py::test_malformed_rankings_are_rejected[candidate_values3-candidate_scores3-candidate_order3]
tests/unit/test_confidence.py::test_malformed_rankings_are_rejected[candidate_values4-candidate_scores4-candidate_order4]
tests/unit/test_confidence.py::test_invalid_top_k_is_rejected[0]
tests/unit/test_confidence.py::test_invalid_top_k_is_rejected[1]
tests/unit/test_confidence.py::test_invalid_top_k_is_rejected[5]
tests/unit/test_confidence.py::test_invalid_top_k_is_rejected[True]
tests/unit/test_confidence.py::test_invalid_top_k_is_rejected[2.0]
tests/unit/test_confidence.py::test_risk_coverage_is_validation_only_and_deterministic
tests/unit/test_confidence.py::test_extended_risk_coverage_requires_stability_data
tests/unit/test_confidence.py::test_risk_coverage_rejects_unknown_or_empty_inputs
tests/unit/test_data_generator.py::test_word_encoding_is_most_significant_bit_first
tests/unit/test_data_generator.py::test_generation_is_identical_for_a_fixed_configuration_and_seed
tests/unit/test_data_generator.py::test_different_seeds_produce_different_samples
tests/unit/test_data_generator.py::test_generated_shapes_labels_and_debug_pairs_are_consistent
tests/unit/test_data_generator.py::test_positive_debug_pairs_use_the_selected_difference
tests/unit/test_data_generator.py::test_split_configs_require_distinct_seeds_and_shared_protocol
tests/unit/test_data_generator.py::test_invalid_dataset_configuration_is_rejected[0-7-1001-train]
tests/unit/test_data_generator.py::test_invalid_dataset_configuration_is_rejected[3-7-1001-train]
tests/unit/test_data_generator.py::test_invalid_dataset_configuration_is_rejected[8-0-1001-train]
tests/unit/test_data_generator.py::test_invalid_dataset_configuration_is_rejected[8-23-1001-train]
tests/unit/test_data_generator.py::test_invalid_dataset_configuration_is_rejected[8-7-1001-holdout]
tests/unit/test_dataset_export.py::test_classifier_export_is_identical_for_a_fixed_seed
tests/unit/test_dataset_export.py::test_different_seeds_change_exported_classifier_data
tests/unit/test_dataset_export.py::test_exported_classifier_metadata_and_balance_are_complete
tests/unit/test_dataset_export.py::test_classifier_benchmark_verifies_all_shards_against_manifest
tests/unit/test_dataset_export.py::test_classifier_loader_rejects_nonbinary_content
tests/unit/test_dataset_export.py::test_hex_word_rejects_noncanonical_text[ 123]
tests/unit/test_dataset_export.py::test_hex_word_rejects_noncanonical_text[+123]
tests/unit/test_dataset_export.py::test_hex_word_rejects_noncanonical_text[0x12]
tests/unit/test_dataset_export.py::test_hex_word_rejects_noncanonical_text[12-3]
tests/unit/test_dataset_export.py::test_exported_classifier_loads_without_the_generator
tests/unit/test_dataset_export.py::test_classifier_checksums_detect_a_tampered_shard
tests/unit/test_dataset_export.py::test_loaded_classifier_shard_matches_in_memory_generation
tests/unit/test_dataset_export.py::test_split_validation_still_requires_one_role_each
tests/unit/test_dataset_v1_freeze.py::test_classifier_freeze_matches_code_and_fixture
tests/unit/test_dataset_v1_freeze.py::test_attack_freeze_uses_plaintext_difference
tests/unit/test_dataset_v1_freeze.py::test_classifier_and_attack_remain_separate_frozen_datasets
tests/unit/test_demo_depth_dataset.py::test_dataset_sidecar_checksum_valid
tests/unit/test_demo_depth_dataset.py::test_cipher_known_answer_tests
tests/unit/test_demo_depth_dataset.py::test_dataset_structure_and_invariants
tests/unit/test_depth_analysis.py::test_timeouts_are_unavailable_and_keep_failure_penalty
tests/unit/test_depth_analysis.py::test_analysis_rejects_grid_or_metric_inconsistency[missing]
tests/unit/test_depth_analysis.py::test_analysis_rejects_grid_or_metric_inconsistency[identity]
tests/unit/test_depth_analysis.py::test_analysis_rejects_grid_or_metric_inconsistency[flag]
tests/unit/test_depth_analysis.py::test_analysis_rejects_grid_or_metric_inconsistency[counts]
tests/unit/test_depth_analysis.py::test_analysis_rejects_grid_or_metric_inconsistency[late]
tests/unit/test_depth_analysis.py::test_hash_corruption_cannot_be_analyzed
tests/unit/test_depth_map_runner.py::test_manifest_validation_passes
tests/unit/test_depth_map_runner.py::test_manifest_validation_rejects_tampering
tests/unit/test_depth_map_runner.py::test_schedule_properties
tests/unit/test_depth_map_runner.py::test_parent_watchdog_terminates_hanging_worker
tests/unit/test_depth_map_runner.py::test_unsat_triggers_correctness_alarm
tests/unit/test_depth_map_runner.py::test_atomic_journal_and_interrupted_cell
tests/unit/test_depth_map_runner.py::test_single_cell_real_solve
tests/unit/test_depth_map_runner.py::test_simon_single_cell_real_solve
tests/unit/test_depth_map_runner.py::test_late_recovery_rejected
tests/unit/test_dimacs.py::test_dimacs_export_and_solve_recovers_master_key
tests/unit/test_dimacs.py::test_dimacs_export_with_pinned_round_key
tests/unit/test_encryption_calibration.py::test_native_kats_all_prefixes_and_public_fixture
tests/unit/test_encryption_calibration.py::test_effective_cpu_policy_is_checked_without_timing
tests/unit/test_encryption_calibration.py::test_observer_deadline_is_passed_to_subprocess
tests/unit/test_encryption_calibration.py::test_pinned_source_is_the_actual_imported_module
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[0 7 1 2\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[0 7 1 2 3 4 5 6 7\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[2 7 1 2 3 4 5 6\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[0 23 1 2 3 4 5 6\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[1 33 1 2 3 4 5 6\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[0 7 -1 2 3 4 5 6\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[0 7 184467440737095516161 2 3 4 5 6\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_invalid_arguments_before_work[args0]
tests/unit/test_encryption_calibration.py::test_native_rejects_invalid_arguments_before_work[args1]
tests/unit/test_encryption_calibration.py::test_native_rejects_invalid_arguments_before_work[args2]
tests/unit/test_encryption_calibration.py::test_native_rejects_invalid_arguments_before_work[args3]
tests/unit/test_encryption_calibration.py::test_native_rejects_invalid_arguments_before_work[args4]
tests/unit/test_encryption_calibration.py::test_native_rejects_invalid_arguments_before_work[args5]
tests/unit/test_encryption_calibration.py::test_full_fixture_has_distinct_keys_and_blocks_and_known_digest
tests/unit/test_encryption_calibration.py::test_measured_conversion_uses_all_repetitions_and_one_profile_rate
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[elapsed_seconds-nan]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[elapsed_seconds--1]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[elapsed_seconds-1.99]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[batches-True]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[blocks-1]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[checksum-wrong]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[qos_class-9]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[nice-5]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[threads-8]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[peak_rss_bytes-0]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[peak_rss_bytes-100000000]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[identity_sha256-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[ordinal-3]
tests/unit/test_encryption_calibration.py::test_missing_duplicate_and_wrong_order_fail_closed
tests/unit/test_encryption_calibration.py::test_frozen_profile_drift_rejected[threads-True]
tests/unit/test_encryption_calibration.py::test_frozen_profile_drift_rejected[repetitions-5]
tests/unit/test_encryption_calibration.py::test_frozen_profile_drift_rejected[minimum_measurement_seconds-1]
tests/unit/test_encryption_calibration.py::test_frozen_profile_drift_rejected[compiler_flags-value3]
tests/unit/test_encryption_calibration.py::test_frozen_profile_drift_rejected[protected_data_access-True]
tests/unit/test_evaluation_metrics.py::test_calculate_global_par2_basic
tests/unit/test_evaluation_metrics.py::test_calculate_global_par2_solved_exceeds_cutoff
tests/unit/test_evaluation_metrics.py::test_calculate_global_par2_solved_exact_cutoff
tests/unit/test_evaluation_metrics.py::test_calculate_global_par2_empty_and_errors
tests/unit/test_evaluation_metrics.py::test_wilson_score_interval_exact
tests/unit/test_evaluation_metrics.py::test_wilson_score_interval_validation
tests/unit/test_evaluation_metrics.py::test_exact_mcnemar_test_basic
tests/unit/test_evaluation_metrics.py::test_calculate_cactus_curve_monotonic
tests/unit/test_fallback_redesign.py::test_invariant_1_g1_feature_extraction_is_deterministic
tests/unit/test_fallback_redesign.py::test_invariant_2_g1_handles_degenerate_distributions_gracefully
tests/unit/test_fallback_redesign.py::test_invariant_3_normalized_logit_margin_matches_cattelan_silva
tests/unit/test_fallback_redesign.py::test_invariant_4_r1_hard_negative_miner_labels_correctly
tests/unit/test_fallback_redesign.py::test_invariant_5_r1_shortlist_builder_preserves_true_subkey
tests/unit/test_fallback_redesign.py::test_invariant_6_rescue_ceiling_bounded_by_top_m_recall
tests/unit/test_fallback_redesign.py::test_invariant_7_cascade_routes_to_b1_when_reliability_high
tests/unit/test_fallback_redesign.py::test_invariant_8_cascade_routes_to_r1_when_reliability_low
tests/unit/test_fallback_redesign.py::test_invariant_9_lazy_b0_never_computed_unless_primary_fails
tests/unit/test_fallback_redesign.py::test_invariant_10_solver_isolation_maintained_in_cascade
tests/unit/test_fallback_redesign.py::test_invariant_11_state_machine_tracks_cumulative_time
tests/unit/test_fallback_redesign.py::test_invariant_12_seed_validation_rejects_quarantined_seeds
tests/unit/test_fallback_redesign.py::test_invariant_13_locked_validation_role_cannot_enter_training_or_calibration
tests/unit/test_fallback_redesign.py::test_invariant_14_g1_optimized_reference_equivalence
tests/unit/test_fallback_redesign.py::test_invariant_15_r1_shortlist_budget_selection
tests/unit/test_fallback_redesign.py::test_invariant_16_terminal_fallback_deduplication_and_cumulative_budget
tests/unit/test_fallback_redesign.py::test_invariant_17_g1_failure_risk_routing_polarity
tests/unit/test_final_publication_figures.py::test_source_hash_mismatch_fails_closed
tests/unit/test_final_publication_figures.py::test_attack_duplicate_or_missing_pair_rejected[speck32_64]
tests/unit/test_final_publication_figures.py::test_attack_duplicate_or_missing_pair_rejected[simon32_64]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[verified_on_fresh_pairs-false]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[verified_on_fresh_pairs-False]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[complete_online_wall_seconds-0]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[complete_online_wall_seconds-nan]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[complete_online_wall_seconds-301]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[instance_id-another_key]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[outcome-fabricated]
tests/unit/test_final_publication_figures.py::test_speck_boolean_and_late_flags_rejected[counts_as_verified_recovery-false]
tests/unit/test_final_publication_figures.py::test_speck_boolean_and_late_flags_rejected[success-False]
tests/unit/test_final_publication_figures.py::test_speck_boolean_and_late_flags_rejected[late_verified_recovery-True]
tests/unit/test_final_publication_figures.py::test_paired_intervals_match_literature_fixtures[counts0-expected0]
tests/unit/test_final_publication_figures.py::test_paired_intervals_match_literature_fixtures[counts1-expected1]
tests/unit/test_final_publication_figures.py::test_failure_matrix_rejects_wrong_total_and_duplicate
tests/unit/test_final_publication_figures.py::test_actual_table_totals_provenance_and_replay
tests/unit/test_final_publication_figures.py::test_recorded_no_entry_is_zero_but_depth_timeout_is_unavailable
tests/unit/test_final_publication_figures.py::test_corrected_paired_ci_and_frozen_holm_families
tests/unit/test_final_publication_figures.py::test_backend_interruption_excluded_only_from_affected_pairs
tests/unit/test_final_publication_figures.py::test_diagnostic_grids_reject_duplicates_and_bad_flags[depth-depth_raw]
tests/unit/test_final_publication_figures.py::test_diagnostic_grids_reject_duplicates_and_bad_flags[backend-backend_raw]
tests/unit/test_final_publication_figures.py::test_main_figures_only_two_panels_and_paper_names
tests/unit/test_g2_pipeline.py::test_partition_masks_properties
tests/unit/test_g2_pipeline.py::test_distributional_features_shift_invariance
tests/unit/test_g2_pipeline.py::test_stability_plus_features
tests/unit/test_g2_pipeline.py::test_g2_feature_vector_extraction
tests/unit/test_g2_pipeline.py::test_g2_models_train_and_predict
tests/unit/test_g2_pipeline.py::test_distributional_exact_arithmetic
tests/unit/test_g2_pipeline.py::test_stability_plus_exact_arithmetic
tests/unit/test_hierarchical_multi_round.py::test_hierarchical_stage_rejects_invalid_arguments
tests/unit/test_hierarchical_multi_round.py::test_attack_rejects_invalid_arguments
tests/unit/test_hierarchical_multi_round.py::test_single_stage_attack_recovers_five_round_instance_key
tests/unit/test_hierarchical_multi_round.py::test_two_stage_attack_refutes_a_wrong_pair_before_verifying
tests/unit/test_hierarchical_multi_round.py::test_identical_runs_produce_identical_search_and_order
tests/unit/test_hierarchical_multi_round.py::test_beam_width_bounds_the_number_of_tested_tuples
tests/unit/test_hierarchical_multi_round.py::test_pruned_first_stage_candidate_reports_ranking_failure
tests/unit/test_hierarchical_multi_round.py::test_pruning_inside_the_final_stage_reports_ranking_failure
tests/unit/test_hierarchical_multi_round.py::test_search_does_not_read_declared_ground_truth
tests/unit/test_hierarchical_multi_round.py::test_frozen_b1_checkpoint_satisfies_the_stage_interface
tests/unit/test_hierarchical_multi_round.py::test_screened_hierarchical_stage_validation_and_execution
tests/unit/test_hierarchical_multi_round.py::test_stage_cache_and_excluded_paths_cascade
tests/unit/test_hierarchical_ranking.py::test_h1_preserves_b1_window_set_and_tail
tests/unit/test_hierarchical_ranking.py::test_h1_b0_ties_use_ascending_candidate_value
tests/unit/test_hierarchical_ranking.py::test_h1_invokes_b0_only_for_the_sixteen_candidate_window
tests/unit/test_hierarchical_ranking.py::test_h1_uses_existing_b0_sum_logits_sorting
tests/unit/test_hierarchical_ranking.py::test_h1_scalar_and_batched_window_scores_match
tests/unit/test_hierarchical_ranking.py::test_h1_rerun_is_deterministic
tests/unit/test_hierarchical_ranking.py::test_h1_serialization_round_trip
tests/unit/test_hierarchical_ranking.py::test_h1_deserialization_rejects_order_inconsistent_with_b0_scores
tests/unit/test_hierarchical_ranking.py::test_h1_deserialization_rejects_corrupt_protocol_metadata[tie_rule]
tests/unit/test_hierarchical_ranking.py::test_h1_deserialization_rejects_corrupt_protocol_metadata[candidate_domain]
tests/unit/test_hierarchical_ranking.py::test_h1_timing_fields_are_finite_and_non_negative
tests/unit/test_hierarchical_ranking.py::test_h1_ordering_api_does_not_accept_or_serialize_true_key
tests/unit/test_hierarchical_ranking.py::test_h1_rejects_incomplete_or_duplicate_b1_order
tests/unit/test_hierarchical_ranking.py::test_h1_rejects_unfrozen_budget_and_invalid_batch_size[kwargs0-window_size]
tests/unit/test_hierarchical_ranking.py::test_h1_rejects_unfrozen_budget_and_invalid_batch_size[kwargs1-b0_candidate_batch_size]
tests/unit/test_hierarchical_ranking.py::test_h1_rejects_wrong_instance_pair_count
tests/unit/test_hierarchical_ranking.py::test_h1_rejects_wrong_ranking_type
tests/unit/test_hierarchical_ranking.py::test_h1_rejects_non_model_b0_input
tests/unit/test_key_schedule.py::test_full_key_schedule_matches_regression_values
tests/unit/test_key_schedule.py::test_reduced_key_schedule_is_a_prefix[1]
tests/unit/test_key_schedule.py::test_reduced_key_schedule_is_a_prefix[2]
tests/unit/test_key_schedule.py::test_reduced_key_schedule_is_a_prefix[7]
tests/unit/test_key_schedule.py::test_reduced_key_schedule_is_a_prefix[11]
tests/unit/test_key_schedule.py::test_reduced_key_schedule_is_a_prefix[22]
tests/unit/test_key_schedule.py::test_key_schedule_is_deterministic
tests/unit/test_key_schedule.py::test_key_schedule_outputs_16_bit_words
tests/unit/test_key_schedule.py::test_key_schedule_constants_match_verified_parameters
tests/unit/test_key_schedule.py::test_master_key_words_are_explicitly_masked
tests/unit/test_key_schedule.py::test_invalid_round_counts_are_rejected[0]
tests/unit/test_key_schedule.py::test_invalid_round_counts_are_rejected[-1]
tests/unit/test_key_schedule.py::test_invalid_round_counts_are_rejected[23]
tests/unit/test_key_schedule.py::test_non_integer_round_count_is_rejected
tests/unit/test_key_schedule.py::test_boolean_round_count_is_rejected
tests/unit/test_key_schedule.py::test_wrong_master_key_length_is_rejected[master_key0]
tests/unit/test_key_schedule.py::test_wrong_master_key_length_is_rejected[master_key1]
tests/unit/test_key_schedule.py::test_wrong_master_key_length_is_rejected[master_key2]
tests/unit/test_key_schedule.py::test_non_integer_master_key_word_is_rejected
tests/unit/test_multi_round_recovery.py::test_assumptions_for_round_key_pins_every_bit
tests/unit/test_multi_round_recovery.py::test_assumptions_reject_invalid_inputs
tests/unit/test_multi_round_recovery.py::test_assumptions_for_round_keys_are_order_independent
tests/unit/test_multi_round_recovery.py::test_trailing_round_key_indices
tests/unit/test_multi_round_recovery.py::test_multi_round_recovery_verifies_true_candidate
tests/unit/test_multi_round_recovery.py::test_multi_round_recovery_skips_wrong_candidate
tests/unit/test_multi_round_recovery.py::test_multi_round_recovery_reports_exhausted_candidates
tests/unit/test_multi_round_recovery.py::test_partial_decrypt_instance_matches_shorter_encryption
tests/unit/test_multi_round_recovery.py::test_partial_decrypt_instance_rejects_invalid_inputs
tests/unit/test_multi_round_recovery.py::test_multi_round_recovery_requires_uniform_candidate_width
tests/unit/test_multi_round_recovery.py::test_seven_round_recovery_needs_two_assumed_round_keys
tests/unit/test_neural_dataset.py::test_features_to_tensor_reshapes_64_bits_to_4_words
tests/unit/test_neural_dataset.py::test_dataset_from_fixture_json
tests/unit/test_neural_dataset.py::test_features_to_tensor_rejects_nonbinary_values
tests/unit/test_neural_dataset.py::test_classifier_splits_load_and_verify_fixture
tests/unit/test_neural_dataset.py::test_dataset_from_generated_in_memory
tests/unit/test_neural_model.py::test_gohr_resnet_forward_shapes
tests/unit/test_neural_model.py::test_gohr_resnet_probabilities_in_range
tests/unit/test_neural_model.py::test_predict_proba_supports_one_sample_without_changing_training_mode
tests/unit/test_neural_model.py::test_single_sample_training_batch_is_supported
tests/unit/test_neural_model.py::test_invalid_resnet_configuration_is_rejected
tests/unit/test_neural_model.py::test_batched_vs_unbatched_inference_equivalence
tests/unit/test_neural_model.py::test_gradient_flows_through_all_parameters
tests/unit/test_neural_train.py::test_roc_auc_metric_perfect_and_random
tests/unit/test_neural_train.py::test_metrics_reject_nonbinary_labels_and_invalid_probabilities
tests/unit/test_neural_train.py::test_train_distinguisher_runs_and_improves_loss
tests/unit/test_neural_train.py::test_seeded_model_training_is_reproducible
tests/unit/test_neural_train.py::test_save_and_load_checkpoint
tests/unit/test_numpy_inference.py::test_numpy_pair_group_outputs_match_pytorch[b1_mean]
tests/unit/test_numpy_inference.py::test_numpy_pair_group_outputs_match_pytorch[b2_moments]
tests/unit/test_numpy_inference.py::test_numpy_pair_group_outputs_match_pytorch[b3_weighted]
tests/unit/test_numpy_inference.py::test_numpy_b0_outputs_match_pytorch
tests/unit/test_numpy_inference.py::test_numpy_adapter_returns_torch_logits
tests/unit/test_numpy_inference.py::test_numpy_backend_rejects_wrong_dtype
tests/unit/test_numpy_inference.py::test_numpy_backend_rejects_wrong_group_shape
tests/unit/test_numpy_inference.py::test_vectorized_final_round_features_match_scalar_reference
tests/unit/test_pair_group_data.py::test_pair_group_generation_is_deterministic_and_typed
tests/unit/test_pair_group_data.py::test_different_pair_group_seed_changes_content
tests/unit/test_pair_group_data.py::test_group_keys_and_positive_negative_construction_are_correct
tests/unit/test_pair_group_data.py::test_group_split_keys_are_disjoint
tests/unit/test_pair_group_data.py::test_pair_group_export_load_and_manifest_verification
tests/unit/test_pair_group_data.py::test_phase6a_attack_export_is_validation_only
tests/unit/test_pair_group_model.py::test_pair_group_forward_shapes_and_permutation_invariance[b1_mean]
tests/unit/test_pair_group_model.py::test_pair_group_forward_shapes_and_permutation_invariance[b2_moments]
tests/unit/test_pair_group_model.py::test_pair_group_forward_shapes_and_permutation_invariance[b3_weighted]
tests/unit/test_pair_group_model.py::test_pair_group_initialization_is_deterministic_and_transfers_b0_encoder
tests/unit/test_pair_group_model.py::test_pair_group_rejects_wrong_dtype_and_shape
tests/unit/test_pair_group_model.py::test_pair_group_gradients_are_finite
tests/unit/test_pair_group_model.py::test_pair_group_checkpoint_round_trip
tests/unit/test_pair_group_model.py::test_pair_group_training_records_finite_validation_bce
tests/unit/test_pair_group_ranking.py::test_pair_group_candidate_batching_and_mode_restore
tests/unit/test_pair_group_ranking.py::test_pair_group_ties_use_ascending_candidate_value
tests/unit/test_pair_group_ranking.py::test_numpy_pair_group_ranking_matches_reference_on_small_domain
tests/unit/test_paired_stats.py::TestContingency2x2::test_compute_2x2_contingency_basic
tests/unit/test_paired_stats.py::TestContingency2x2::test_compute_2x2_contingency_mismatch
tests/unit/test_paired_stats.py::TestExactMcNemar::test_exact_binomial_twoside_pvalue
tests/unit/test_paired_stats.py::TestExactMcNemar::test_exact_mcnemar_literature_values
tests/unit/test_paired_stats.py::TestExactMcNemar::test_exact_mcnemar_errors
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_altman_table_6_2_literature_fixture
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_altman_table_6_2_reversed_symmetry
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_newcombe_table_3_literature_fixture
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_zero_discordance_ties
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_different_concordant_splits
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_negative_association
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_boundary_proportions_full_success
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_boundary_proportions_both_zero
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_boundary_proportions_both_one
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_complete_separation
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_small_sample_sizes_n1
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_reversal_symmetry_general
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_validation_errors
tests/unit/test_paired_stats.py::TestHolmBonferroniAdjustment::test_two_primary_contrasts_both_rejected
tests/unit/test_paired_stats.py::TestHolmBonferroniAdjustment::test_two_primary_contrasts_neither_rejected
tests/unit/test_paired_stats.py::TestHolmBonferroniAdjustment::test_two_primary_contrasts_one_rejected
tests/unit/test_paired_stats.py::TestHolmBonferroniAdjustment::test_order_invariance
tests/unit/test_paired_stats.py::TestHolmBonferroniAdjustment::test_boundary_cases
tests/unit/test_paired_stats.py::TestMcNemarPowerSensitivity::test_protocol_power_fixtures
tests/unit/test_paired_stats.py::TestMcNemarPowerSensitivity::test_power_edge_cases
tests/unit/test_partial.py::test_true_final_round_key_recovers_the_pre_final_state
tests/unit/test_partial.py::test_wrong_final_round_key_does_not_recover_the_pre_final_state
tests/unit/test_partial.py::test_partial_decryption_matches_inverse_round_for_seeded_states
tests/unit/test_phase10_vtc.py::test_solver_isolation_invariant
tests/unit/test_phase10_vtc.py::test_vtc_early_termination_on_b1_success
tests/unit/test_phase10_vtc.py::test_vtc_escalation_on_b1_exhaustion
tests/unit/test_phase10_vtc.py::test_vtc_candidate_deduplication
tests/unit/test_phase10_vtc.py::test_vtc_budget_capping
tests/unit/test_phase10_vtc.py::test_vtc_serialization_roundtrip
tests/unit/test_phase11b_prime_pair_group.py::test_six_round_generation_is_deterministic
tests/unit/test_phase11b_prime_pair_group.py::test_six_round_metadata_records_six_rounds_and_owned_dataset_id
tests/unit/test_phase11b_prime_pair_group.py::test_six_round_features_use_six_round_encryption
tests/unit/test_phase11b_prime_pair_group.py::test_six_round_config_rejects_five_rounds
tests/unit/test_phase11b_prime_pair_group.py::test_frozen_five_round_config_still_rejects_six_rounds
tests/unit/test_phase11b_prime_pair_group.py::test_frozen_five_round_identifiers_are_unchanged
tests/unit/test_phase11b_prime_pair_group.py::test_six_round_export_manifest_and_shard_round_trip
tests/unit/test_phase11b_prime_pair_group.py::test_five_round_path_rejects_six_round_artefacts
tests/unit/test_phase11b_prime_pair_group.py::test_five_round_export_cannot_emit_six_round_configurations
tests/unit/test_phase11b_prime_pair_group.py::test_stage1_seed_role_accepts_its_own_seeds
tests/unit/test_phase11b_prime_pair_group.py::test_stage1_seed_role_rejects_history_and_other_protected_seeds
tests/unit/test_phase11b_prime_pair_group.py::test_protected_stage1_seeds_are_rejected_by_other_roles
tests/unit/test_phase6b_proposal_figures.py::test_phase6b_summary_is_validation_only_and_complete
tests/unit/test_phase6b_proposal_figures.py::test_phase6b_result_figure_uses_selected_b1_metrics
tests/unit/test_phase6b_proposal_figures.py::test_phase6b_methodology_marks_planned_components
tests/unit/test_phase6b_proposal_figures.py::test_all_eight_proposal_figures_generate_valid_svg
tests/unit/test_phase6b_proposal_figures.py::test_proposal_figures_generation_in_temp_dir
tests/unit/test_phase6b_proposal_figures.py::test_phase6b_figure_manifest_is_reproducible
tests/unit/test_phase6c_calibration.py::test_phase6c_feature_uses_ordered_top8_and_all_score_std
tests/unit/test_phase6c_calibration.py::test_phase6c_feature_has_zero_score_std_guard
tests/unit/test_phase6c_calibration.py::test_platt_fit_is_deterministic_bounded_and_serializable
tests/unit/test_phase6c_calibration.py::test_platt_fit_rejects_nonconvergence_and_single_class
tests/unit/test_phase6c_calibration.py::test_threshold_selection_is_unique_deterministic_and_fails_closed
tests/unit/test_phase6c_calibration.py::test_ece_boundaries_and_auc_tie_credit_are_explicit
tests/unit/test_phase6c_calibration.py::test_calibration_report_contains_prevalence_brier_and_risk_coverage
tests/unit/test_phase6c_calibration.py::test_bootstrap_mean_interval_is_seed_deterministic
tests/unit/test_phase6c_fallback.py::test_fallback_alternates_and_fills_exactly_32_unique_candidates
tests/unit/test_phase6c_fallback.py::test_fallback_first_occurrence_and_duplicate_fill_are_deterministic
tests/unit/test_phase6c_fallback.py::test_route_boundary_uses_trusted_route_at_threshold
tests/unit/test_phase6c_fallback.py::test_fallback_rejects_non_complete_inputs[b10-b00]
tests/unit/test_phase6c_fallback.py::test_fallback_rejects_non_complete_inputs[b11-b01]
tests/unit/test_phase6c_fallback.py::test_fallback_rejects_non_complete_inputs[b12-b02]
tests/unit/test_phase7_evaluation_figures.py::test_phase7_summary_is_complete_and_has_all_five_methods
tests/unit/test_phase7_evaluation_figures.py::test_phase7_figures_render_valid_svg_strings
tests/unit/test_phase7_evaluation_figures.py::test_phase7_latex_and_markdown_tables_are_valid
tests/unit/test_phase7_protocol.py::test_phase7_config_is_held_out_test_and_seed_disjoint
tests/unit/test_phase7_protocol.py::test_phase7_methods_include_all_five_configurations
tests/unit/test_phase7_protocol.py::test_phase7_counterbalanced_order_is_cyclic_permutation
tests/unit/test_phase7_protocol.py::test_phase7_atomic_write_and_summary_aggregation
tests/unit/test_primitives.py::test_mask_word_wraps_to_16_bits[0-0]
tests/unit/test_primitives.py::test_mask_word_wraps_to_16_bits[1-1]
tests/unit/test_primitives.py::test_mask_word_wraps_to_16_bits[65535-65535]
tests/unit/test_primitives.py::test_mask_word_wraps_to_16_bits[65536-0]
tests/unit/test_primitives.py::test_mask_word_wraps_to_16_bits[-1-65535]
tests/unit/test_primitives.py::test_rotations_use_16_bit_words[4660-0-4660-4660]
tests/unit/test_primitives.py::test_rotations_use_16_bit_words[4660-4-9025-16675]
tests/unit/test_primitives.py::test_rotations_use_16_bit_words[4660-16-4660-4660]
tests/unit/test_primitives.py::test_rotations_use_16_bit_words[65535-7-65535-65535]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-0]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-1]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-2]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-7]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-15]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-16]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-31]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-0]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-1]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-2]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-7]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-15]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-16]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-31]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-0]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-1]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-2]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-7]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-15]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-16]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-31]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-0]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-1]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-2]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-7]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-15]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-16]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-31]
tests/unit/test_primitives.py::test_rotations_normalise_negative_counts
tests/unit/test_primitives.py::test_add_words_wraps
tests/unit/test_primitives.py::test_subtract_words_wraps
tests/unit/test_primitives.py::test_addition_and_subtraction_are_inverses[0-0]
tests/unit/test_primitives.py::test_addition_and_subtraction_are_inverses[1-2]
tests/unit/test_primitives.py::test_addition_and_subtraction_are_inverses[4660-65244]
tests/unit/test_primitives.py::test_addition_and_subtraction_are_inverses[65535-65535]
tests/unit/test_primitives.py::test_non_integer_word_inputs_are_rejected[mask_word]
tests/unit/test_primitives.py::test_non_integer_word_inputs_are_rejected[rotate_left]
tests/unit/test_primitives.py::test_non_integer_word_inputs_are_rejected[rotate_right]
tests/unit/test_primitives.py::test_boolean_inputs_are_rejected
tests/unit/test_protocol_manifest.py::test_authoritative_manifest_passes_validation
tests/unit/test_protocol_manifest.py::test_seed_governance_rejection
tests/unit/test_protocol_manifest.py::test_hash_tamper_rejection
tests/unit/test_protocol_manifest.py::test_parameter_tamper_rejection
tests/unit/test_protocol_manifest.py::test_comparator_methods_completeness
tests/unit/test_protocol_manifest.py::test_duplicate_method_ids_rejection
tests/unit/test_protocol_manifest.py::test_method_parameter_bounds_rejection
tests/unit/test_protocol_manifest.py::test_missing_required_section_rejection
tests/unit/test_protocol_manifest.py::test_failure_taxonomy_completeness_and_distinction
tests/unit/test_protocol_manifest.py::test_statistical_plan_and_multiplicity_rejection
tests/unit/test_protocol_manifest.py::test_formula_metrics_and_environment_rejection
tests/unit/test_protocol_manifest.py::test_overwrite_protection
tests/unit/test_protocol_manifest.py::test_prospective_execution_refused_on_proposed_status
tests/unit/test_protocol_manifest.py::test_prospective_execution_refused_on_null_seed
tests/unit/test_protocol_manifest.py::test_prospective_execution_accepted_on_frozen_authorized
tests/unit/test_protocol_manifest.py::test_d030_rejected_for_prospective_authorization
tests/unit/test_protocol_manifest.py::test_retired_seed_rejection_in_prospective
tests/unit/test_protocol_manifest.py::test_failure_taxonomy_orthogonal_fields_validation
tests/unit/test_protocol_manifest.py::test_classify_attack_outcome_mapping
tests/unit/test_protocol_manifest.py::test_prospective_manifest_rejects_seed_406737
tests/unit/test_protocol_manifest.py::test_prospective_dataset_validation_rejects_seed_406737
tests/unit/test_protocol_manifest.py::test_development_dataset_inspection_allowed_without_prospective_status
tests/unit/test_protocol_manifest.py::test_seed_policy_governance_for_406737
tests/unit/test_protocol_manifest.py::test_critical_code_mutation_invalidates_manifest
tests/unit/test_publication_evidence_package.py::test_relocation_retains_original_bytes_and_truth_separation
tests/unit/test_publication_evidence_package.py::test_nonportable_paths_fail_closed[.]
tests/unit/test_publication_evidence_package.py::test_nonportable_paths_fail_closed[../bad]
tests/unit/test_publication_evidence_package.py::test_nonportable_paths_fail_closed[/absolute]
tests/unit/test_publication_evidence_package.py::test_nonportable_paths_fail_closed[a/../b]
tests/unit/test_publication_evidence_package.py::test_nonportable_paths_fail_closed[a\\b]
tests/unit/test_publication_evidence_package.py::test_nonportable_paths_fail_closed[a//b]
tests/unit/test_publication_evidence_package.py::test_output_overwrite_and_source_tampering_are_rejected
tests/unit/test_publication_evidence_package.py::test_package_corruption_and_unexpected_files_are_rejected[missing]
tests/unit/test_publication_evidence_package.py::test_package_corruption_and_unexpected_files_are_rejected[extra]
tests/unit/test_publication_evidence_package.py::test_package_corruption_and_unexpected_files_are_rejected[altered]
tests/unit/test_publication_evidence_package.py::test_package_corruption_and_unexpected_files_are_rejected[symlink]
tests/unit/test_publication_evidence_package.py::test_package_corruption_and_unexpected_files_are_rejected[root_license]
tests/unit/test_publication_evidence_package.py::test_package_corruption_and_unexpected_files_are_rejected[source_license]
tests/unit/test_publication_evidence_package.py::test_license_provenance_detects_copies_changed_with_rehashed_file_manifest
tests/unit/test_publication_evidence_package.py::test_schema1_bundle_without_license_remains_verifiable
tests/unit/test_publication_evidence_package.py::test_build_requires_unchanged_committed_root_license_before_output[missing]
tests/unit/test_publication_evidence_package.py::test_build_requires_unchanged_committed_root_license_before_output[modified]
tests/unit/test_publication_evidence_package.py::test_build_requires_unchanged_committed_root_license_before_output[staged]
tests/unit/test_publication_evidence_package.py::test_build_requires_unchanged_committed_root_license_before_output[untracked]
tests/unit/test_publication_evidence_package.py::test_build_rechecks_license_before_creating_output
tests/unit/test_publication_evidence_package.py::test_portable_metadata_cannot_be_changed_even_if_local_hashes_updated
tests/unit/test_publication_evidence_package.py::test_duplicate_json_keys_rejected
tests/unit/test_publication_evidence_package.py::test_original_detached_sidecar_corruption_rejects_packaging
tests/unit/test_publication_evidence_package.py::test_formula_symlink_outside_pinned_directory_rejected
tests/unit/test_publication_evidence_package.py::test_scalar_namespace_loads_leaf_without_eager_training_exports
tests/unit/test_pure_cnf.py::test_speck_pure_cnf_export_solved_by_cadical_and_cms
tests/unit/test_pure_cnf.py::test_speck_pure_cnf_export_with_pinned_keys
tests/unit/test_pure_cnf_adversarial.py::test_empty_contradiction_survives_conversion[0]
tests/unit/test_pure_cnf_adversarial.py::test_empty_contradiction_survives_conversion[x 0]
tests/unit/test_pure_cnf_adversarial.py::test_empty_contradiction_survives_conversion[x0]
tests/unit/test_pure_cnf_adversarial.py::test_empty_formula_is_tautology
tests/unit/test_pure_cnf_adversarial.py::test_generic_empty_xor_preserves_both_rhs_values[False]
tests/unit/test_pure_cnf_adversarial.py::test_generic_empty_xor_preserves_both_rhs_values[True]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals0]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals1]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals2]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals3]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals4]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals5]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals6]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals7]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals8]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals0]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals1]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals2]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals3]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals4]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals5]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals6]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals7]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals8]
tests/unit/test_pure_cnf_adversarial.py::test_multiple_long_xors_use_disjoint_fresh_auxiliaries
tests/unit/test_pure_cnf_adversarial.py::test_declared_unused_variables_are_reserved
tests/unit/test_pure_cnf_adversarial.py::test_wrapped_and_multiple_records_preserve_semantics
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[1 0\n-1-header]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[-0-header]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\np cnf 1 1\n1 0\n-1-header]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 0\n1 0\n-1-record count]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 2\n1 0\n-1-record count]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\n1\n-1-unterminated]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\nx 1\n-1-unterminated]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\nx\n-1-unterminated]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 2\n1 x 1 0\n-1-unterminated]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\n2 0\n-1-range]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\nx -2 0\n-1-range]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 6 1\nx 1 2 3 4 5 6 0\n-5-match]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf -1 0\n-0-negative]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 -1\n-1-negative]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf one 0\n-1-noninteger]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1_0 0\n-10-noninteger]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\n\u0661 0\n-1-noninteger]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 10 1\n1_0 0\n-10-noninteger]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 0 trailing\n-1-expected header]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p other 1 0\n-1-expected header]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\nword 0\n-1-noninteger]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\nx 1 0 0\n-1-record count]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\n1 0 1\n-1-unterminated]
tests/unit/test_pure_cnf_adversarial.py::test_invalid_supplied_variable_count_rejected[True]
tests/unit/test_pure_cnf_adversarial.py::test_invalid_supplied_variable_count_rejected[False]
tests/unit/test_pure_cnf_adversarial.py::test_invalid_supplied_variable_count_rejected[-1]
tests/unit/test_pure_cnf_adversarial.py::test_invalid_supplied_variable_count_rejected[1.0]
tests/unit/test_pure_cnf_adversarial.py::test_invalid_supplied_variable_count_rejected[1]
tests/unit/test_pure_cnf_adversarial.py::test_invalid_supplied_variable_count_rejected[None]
tests/unit/test_queue_order_environment.py::test_author_allows_battery_or_ac_with_observed_sleep_protection[False]
tests/unit/test_queue_order_environment.py::test_author_allows_battery_or_ac_with_observed_sleep_protection[True]
tests/unit/test_queue_order_environment.py::test_battery_allowance_keeps_telemetry_and_sleep_gates[supported-False]
tests/unit/test_queue_order_environment.py::test_battery_allowance_keeps_telemetry_and_sleep_gates[sleep_assertion_observed-False]
tests/unit/test_queue_order_environment.py::test_battery_allowance_keeps_telemetry_and_sleep_gates[assertions_returncode-1]
tests/unit/test_queue_order_environment.py::test_battery_allowance_keeps_telemetry_and_sleep_gates[power_returncode-1]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[Now drawing from 'Unknown Power'\n-False]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[-False]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[None-False]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[Now drawing from 'Battery Power'\nNow drawing from 'AC Power'\n-True]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[Now drawing from 'AC Power'\nNow drawing from 'Unknown Power'\n-True]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[Now drawing from 'Battery Power'\n-True]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[Now drawing from 'AC Power'\n-False]
tests/unit/test_queue_order_environment.py::test_interpreter_alias_accepts_same_venv_but_rejects_other_venv_or_base
tests/unit/test_queue_order_ranking.py::test_speck_composition_matches_original_one_based_stage_ties
tests/unit/test_queue_order_ranking.py::test_simon_composition_matches_original_zero_based_numpy_orders
tests/unit/test_queue_order_ranking.py::test_composers_reject_malformed_prefixes[stage10-stage20]
tests/unit/test_queue_order_ranking.py::test_composers_reject_malformed_prefixes[stage11-stage21]
tests/unit/test_queue_order_ranking.py::test_composers_reject_malformed_prefixes[stage12-stage22]
tests/unit/test_queue_order_ranking.py::test_composers_reject_malformed_prefixes[stage13-stage23]
tests/unit/test_queue_order_ranking.py::test_speck_rank_queue_uses_frozen_stage_calls_and_public_only_instance
tests/unit/test_queue_order_ranking.py::test_speck_ranking_fails_closed_without_mps
tests/unit/test_queue_order_ranking.py::test_simon_classical_uses_fixed_public_ranking_settings_and_numpy_orders
tests/unit/test_queue_order_ranking.py::test_simon_bao_path_uses_existing_pinned_subprocess_builder
tests/unit/test_queue_order_ranking.py::test_rank_queue_rejects_truth_fields_and_policy_drift_before_dispatch
tests/unit/test_queue_order_worker.py::test_traversal_reuses_model_after_unsat_but_starts_a_fresh_model_per_arm
tests/unit/test_queue_order_worker.py::test_unresolved_status_rebuilds_before_next_candidate[solver_timeout]
tests/unit/test_queue_order_worker.py::test_unresolved_status_rebuilds_before_next_candidate[unsat_or_timeout]
tests/unit/test_queue_order_worker.py::test_verification_failure_rebuilds_and_is_not_reported_as_recovery
tests/unit/test_queue_order_worker.py::test_traverse_rejects_wrong_observation_counts_before_model_construction[127-2]
tests/unit/test_queue_order_worker.py::test_traverse_rejects_wrong_observation_counts_before_model_construction[128-1]
tests/unit/test_queue_order_worker.py::test_new_model_uses_only_the_leading_four_known_observations[speck32_64]
tests/unit/test_queue_order_worker.py::test_new_model_uses_only_the_leading_four_known_observations[simon32_64]
tests/unit/test_queue_order_worker.py::test_traverse_builds_chronological_round_key_assumptions[speck32_64]
tests/unit/test_queue_order_worker.py::test_traverse_builds_chronological_round_key_assumptions[simon32_64]
tests/unit/test_queue_order_worker.py::test_four_round_cipher_sat_trailing_assumptions_and_reference_verification[speck32_64]
tests/unit/test_queue_order_worker.py::test_four_round_cipher_sat_trailing_assumptions_and_reference_verification[simon32_64]
tests/unit/test_ranking.py::test_true_candidate_maps_to_the_expected_pre_final_state
tests/unit/test_ranking.py::test_correct_candidate_ranks_first_and_model_mode_is_restored
tests/unit/test_ranking.py::test_candidate_batching_produces_identical_scores_and_order
tests/unit/test_ranking.py::test_score_ties_use_ascending_candidate_value
tests/unit/test_ranking.py::test_invalid_candidate_sets_are_rejected[candidate_values0]
tests/unit/test_ranking.py::test_invalid_candidate_sets_are_rejected[candidate_values1]
tests/unit/test_ranking.py::test_invalid_candidate_sets_are_rejected[candidate_values2]
tests/unit/test_ranking.py::test_invalid_candidate_sets_are_rejected[candidate_values3]
tests/unit/test_ranking.py::test_invalid_candidate_sets_are_rejected[candidate_values4]
tests/unit/test_ranking.py::test_empty_instance_is_rejected
tests/unit/test_ranking.py::test_model_must_return_one_logit_per_row
tests/unit/test_ranking.py::test_default_candidate_set_is_the_complete_16_bit_domain
tests/unit/test_ranking.py::test_evaluation_reports_missing_true_candidate_without_guessing
tests/unit/test_ranking.py::test_pair_subset_stability_reuses_per_pair_logits
tests/unit/test_ranking.py::test_invalid_pair_subset_stability_is_rejected[pair_subsets0]
tests/unit/test_ranking.py::test_invalid_pair_subset_stability_is_rejected[pair_subsets1]
tests/unit/test_ranking.py::test_invalid_pair_subset_stability_is_rejected[pair_subsets2]
tests/unit/test_ranking.py::test_invalid_pair_subset_stability_is_rejected[pair_subsets3]
tests/unit/test_ranking.py::test_invalid_pair_subset_stability_is_rejected[pair_subsets4]
tests/unit/test_ranking.py::test_invalid_pair_subset_stability_is_rejected[pair_subsets5]
tests/unit/test_ranking_depth_ranking.py::test_speck_stage1_spy_receives_only_ciphertext_and_fixed_mps_batch
tests/unit/test_ranking_depth_ranking.py::test_speck_sparse_r5_stage2_uses_all_fixed_parents_and_existing_queue_order
tests/unit/test_ranking_depth_ranking.py::test_one_and_two_key_simon_queues_share_persisted_stage1_prefix
tests/unit/test_ranking_depth_ranking.py::test_rejects_nonfrozen_trailing_key_counts[0]
tests/unit/test_ranking_depth_ranking.py::test_rejects_nonfrozen_trailing_key_counts[3]
tests/unit/test_ranking_depth_ranking.py::test_rejects_nonfrozen_trailing_key_counts[True]
tests/unit/test_ranking_depth_ranking.py::test_rejects_nonfrozen_trailing_key_counts[1.0]
tests/unit/test_ranking_depth_ranking.py::test_accepts_only_exact_public_ciphertext_pair_shape[bad_pairs0]
tests/unit/test_ranking_depth_ranking.py::test_accepts_only_exact_public_ciphertext_pair_shape[bad_pairs1]
tests/unit/test_ranking_depth_ranking.py::test_accepts_only_exact_public_ciphertext_pair_shape[bad_pairs2]
tests/unit/test_ranking_depth_ranking.py::test_accepts_only_exact_public_ciphertext_pair_shape[bad_pairs3]
tests/unit/test_ranking_depth_ranking.py::test_rejects_bool_and_duplicate_rank_words
tests/unit/test_ranking_depth_ranking.py::test_tf_child_profile_is_stage1_only_and_has_no_truth_inputs
tests/unit/test_ranking_depth_ranking.py::test_tf_import_boundary_bypasses_simon_init_and_blocks_sat
tests/unit/test_ranking_depth_ranking.py::test_source_receipts_hash_project_package_code_not_runtime_packages
tests/unit/test_reference_encryption_report.py::test_inclusive_boundaries_and_failure_denominator
tests/unit/test_reference_encryption_report.py::test_invalid_cohort_record_rejected[id-b]
tests/unit/test_reference_encryption_report.py::test_invalid_cohort_record_rejected[success-false]
tests/unit/test_reference_encryption_report.py::test_invalid_cohort_record_rejected[wall-nan]
tests/unit/test_reference_encryption_report.py::test_invalid_cohort_record_rejected[wall-True]
tests/unit/test_reference_encryption_report.py::test_invalid_cohort_record_rejected[wall-0.0]
tests/unit/test_reference_encryption_report.py::test_invalid_cohort_record_rejected[wall-300.1]
tests/unit/test_reference_encryption_report.py::test_missing_record_rejected
tests/unit/test_sat_encoding.py::test_four_bit_modular_addition_truth_table
tests/unit/test_sat_encoding.py::test_native_xor_encoder_matches_truth_table
tests/unit/test_screened_group_ranking.py::test_screened_ranking_invariants
tests/unit/test_screened_group_ranking.py::test_screened_ranking_validations
tests/unit/test_seven_round_analysis.py::test_holm_bonferroni_correction
tests/unit/test_seven_round_analysis.py::test_compute_paired_contingency
tests/unit/test_seven_round_analysis.py::test_paired_risk_difference_ci
tests/unit/test_seven_round_analysis.py::test_analyze_records_end_to_end
tests/unit/test_seven_round_analysis.py::test_shuffled_instance_order_alignment
tests/unit/test_seven_round_analysis.py::test_duplicate_instance_id_rejection
tests/unit/test_seven_round_analysis.py::test_missing_method_rejection
tests/unit/test_seven_round_analysis.py::test_mismatched_instance_sets_rejection
tests/unit/test_seven_round_analysis.py::test_contradictory_record_integrity
tests/unit/test_seven_round_analysis.py::test_prospective_requires_n_128
tests/unit/test_seven_round_analysis.py::test_even_n_median_correctness
tests/unit/test_seven_round_analysis.py::test_newcombe_method_10_edge_cases
tests/unit/test_seven_round_analysis.py::test_analyzer_prospective_validation_gate
tests/unit/test_seven_round_comparators.py::test_stage_model_validation
tests/unit/test_seven_round_comparators.py::test_classical_stage_conditional_scoring
tests/unit/test_seven_round_comparators.py::test_truth_free_invariant
tests/unit/test_seven_round_comparators.py::test_global_timeout_during_ranking
tests/unit/test_seven_round_comparators.py::test_global_timeout_in_multi_round_recovery
tests/unit/test_seven_round_comparators.py::test_random_multi_round_candidates
tests/unit/test_seven_round_comparators.py::test_run_random_multi_round_recovery
tests/unit/test_seven_round_comparators.py::test_run_multi_round_sat_only_recovery
tests/unit/test_seven_round_comparators.py::test_solver_isolation_across_comparators
tests/unit/test_seven_round_comparators.py::test_resume_preserves_unselected_methods_behavioral_main
tests/unit/test_seven_round_comparators.py::test_resume_without_output_fails_loudly
tests/unit/test_seven_round_comparators.py::test_method_alias_normalization
tests/unit/test_seven_round_comparators.py::test_candidate_budget_defaults_to_m1_times_m2
tests/unit/test_seven_round_comparators.py::test_sat_only_respects_global_timeout
tests/unit/test_seven_round_comparators.py::test_recovery_continues_after_single_candidate_timeout
tests/unit/test_seven_round_comparators.py::test_solver_rebuilt_on_candidate_timeout
tests/unit/test_seven_round_comparators.py::test_distinct_outcome_categories
tests/unit/test_seven_round_comparators.py::test_resume_fail_closed_validation
tests/unit/test_seven_round_comparators.py::test_sat_only_observation_counts_and_formula_sizes
tests/unit/test_seven_round_comparators.py::test_model_construction_consumes_method_budget_and_clamps_solver_timeout
tests/unit/test_seven_round_comparators.py::test_method_budget_isolation_and_order_invariance
tests/unit/test_seven_round_comparators.py::test_solver_timeout_leaves_instance_unverified_with_cause
tests/unit/test_seven_round_comparators.py::test_runner_prospective_gate_fails_closed_on_proposed_manifest
tests/unit/test_seven_round_comparators.py::test_runner_flags_conflict_rejection
tests/unit/test_seven_round_comparators.py::test_runner_rebuilding_seconds_and_summary
tests/unit/test_seven_round_comparators.py::test_runner_non_interruptible_overshoot_reclassification
tests/unit/test_seven_round_comparators.py::test_runner_resume_validation_fail_closed
tests/unit/test_seven_round_comparators.py::test_runner_late_verified_recovery_reclassification
tests/unit/test_seven_round_comparators.py::test_zero_remaining_time_clean_exit
tests/unit/test_seven_round_comparators.py::test_verify_record_integrity_contradictions
tests/unit/test_seven_round_comparators.py::test_delayed_diagnostics_regression
tests/unit/test_seven_round_comparators.py::test_disjoint_timing_fields
tests/unit/test_seven_round_comparators.py::test_solver_exception_handling
tests/unit/test_seven_round_comparators.py::test_device_enforcement_prospective
tests/unit/test_seven_round_comparators.py::test_counterbalanced_method_order_cyclic
tests/unit/test_seven_round_comparators.py::test_resume_cohort_and_code_drift_validation
tests/unit/test_seven_round_prospective_data.py::test_draw_csprng_seed_rejection_and_policy
tests/unit/test_seven_round_prospective_data.py::test_generation_config_hash_invariance
tests/unit/test_seven_round_prospective_data.py::test_prohibited_seeds_contain_all_critical_seeds
tests/unit/test_seven_round_prospective_data.py::test_small_cohort_generation_and_deterministic_replay
tests/unit/test_simon_attack_instance.py::test_demo_instance_generation_is_deterministic
tests/unit/test_simon_attack_instance.py::test_input_difference_structure_and_encryption
tests/unit/test_simon_attack_instance.py::test_fresh_verification_pairs_are_independently_valid
tests/unit/test_simon_attack_instance.py::test_generator_rejects_attack_and_verification_plaintext_collisions
tests/unit/test_simon_attack_instance.py::test_instance_rejects_overlap_after_deserialization_or_manual_construction
tests/unit/test_simon_attack_instance.py::test_zero_effective_input_difference_is_rejected[input_difference0]
tests/unit/test_simon_attack_instance.py::test_zero_effective_input_difference_is_rejected[input_difference1]
tests/unit/test_simon_attack_instance.py::test_impossible_unique_plaintext_request_is_rejected
tests/unit/test_simon_attack_instance.py::test_opened_demo_instances_replay_unchanged[8-8a762dee8ab2f4d0e8e96c10f9f3fb3e00a681fa419bfd18104570cadb1f16e8]
tests/unit/test_simon_attack_instance.py::test_opened_demo_instances_replay_unchanged[9-b294ee5fed9e0b6ae79437ef976da67b5e061b5f3318858948fd9bfcea98bca4]
tests/unit/test_simon_attack_instance.py::test_opened_demo_instances_replay_unchanged[10-e4e9b03e21a67c28d7764ab8e759f6c7bb2d8377f32c89e164946e45b3357718]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[98001]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[97001]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[99001]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[100001]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[101001]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[300001]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[406737]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[669981]
tests/unit/test_simon_attack_instance.py::test_role_governance_rejects_unauthorized_roles
tests/unit/test_simon_attack_instance.py::test_demo_role_requires_cli_demo_seed
tests/unit/test_simon_attack_instance.py::test_development_role_cannot_relabel_opened_demo_seed
tests/unit/test_simon_attack_instance.py::test_zero_or_negative_fresh_verification_count_is_rejected[0]
tests/unit/test_simon_attack_instance.py::test_zero_or_negative_fresh_verification_count_is_rejected[-1]
tests/unit/test_simon_attack_instance.py::test_noninteger_verification_count_is_rejected[True]
tests/unit/test_simon_attack_instance.py::test_noninteger_verification_count_is_rejected[1.5]
tests/unit/test_simon_attack_instance.py::test_noninteger_verification_count_is_rejected[2]
tests/unit/test_simon_attack_instance.py::test_ground_truth_isolation_properties
tests/unit/test_simon_attack_instance.py::test_serialization_round_trip
tests/unit/test_simon_b1_data.py::test_training_key_role_audit_rejects_duplicates_and_cross_role_overlap
tests/unit/test_simon_b1_data.py::test_fixed_9round_direct_pair_construction
tests/unit/test_simon_b1_data.py::test_group_master_key_sharing
tests/unit/test_simon_b1_data.py::test_strict_master_key_splitting
tests/unit/test_simon_b1_data.py::test_label_balance_and_permutation
tests/unit/test_simon_b1_data.py::test_deterministic_replay
tests/unit/test_simon_b1_data.py::test_feature_layout_and_bit_slicing
tests/unit/test_simon_b1_data.py::test_to_b0_data_flattening
tests/unit/test_simon_b1_data.py::test_pytorch_datasets_and_dataloaders
tests/unit/test_simon_b1_data.py::test_train_and_eval_step_b0
tests/unit/test_simon_b1_data.py::test_train_and_eval_step_b1
tests/unit/test_simon_b1_data.py::test_b0_weight_transfer_initialize_from_b0
tests/unit/test_simon_b1_data.py::test_checkpoint_save_and_load
tests/unit/test_simon_b1_data.py::test_synthetic_overfit_check_b1
tests/unit/test_simon_b1_data.py::test_synthetic_overfit_check_b0
tests/unit/test_simon_b1_data.py::test_proposed_training_manifest_specification
tests/unit/test_simon_b1_training_runner.py::test_proposed_manifest_loads_and_validates
tests/unit/test_simon_b1_training_runner.py::test_manifest_validation_rejects_missing_file_and_invalid_json
tests/unit/test_simon_b1_training_runner.py::test_manifest_validation_rejects_missing_required_keys
tests/unit/test_simon_b1_training_runner.py::test_manifest_validation_rejects_invalid_governing_decisions
tests/unit/test_simon_b1_training_runner.py::test_manifest_validation_rejects_invalid_architectures
tests/unit/test_simon_b1_training_runner.py::test_manifest_validation_rejects_forbidden_seeds
tests/unit/test_simon_b1_training_runner.py::test_manifest_validation_rejects_invalid_hyperparameters
tests/unit/test_simon_b1_training_runner.py::test_key_audit_rejects_duplicates_before_training
tests/unit/test_simon_b1_training_runner.py::test_key_audit_rejects_train_val_overlap_before_training
tests/unit/test_simon_b1_training_runner.py::test_preflight_cryptography_scalar_and_oracle_verification
tests/unit/test_simon_b1_training_runner.py::test_bounded_smoke_test_execution_and_checkpoint_reload
tests/unit/test_simon_b1_training_runner.py::test_deterministic_checkpoint_selection_on_tied_and_improving_losses
tests/unit/test_simon_b1_training_runner.py::test_deterministic_checkpoint_selection_in_b0_stage
tests/unit/test_simon_b1_training_runner.py::test_weak_model_retained_and_labeled_descriptive
tests/unit/test_simon_b1_training_runner.py::test_nan_loss_aborts_immediately
tests/unit/test_simon_b1_training_runner.py::test_zero_gradients_aborts_immediately
tests/unit/test_simon_b1_training_runner.py::test_non_finite_weights_aborts_immediately
tests/unit/test_simon_b1_training_runner.py::test_cli_requires_smoke_or_confirmation
tests/unit/test_simon_b1_training_runner.py::test_cli_smoke_test_runs_cleanly
tests/unit/test_simon_candidate_history_observation.py::test_complete_grid_derives_only_matched_target_with_unsat_history
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[duplicate-Duplicate method/key]
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[missing-Incomplete method/key]
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[flag-Non-Boolean]
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[false_flag-Outcome/flag conflict]
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[wrong_target-Target tuple differs]
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[rebuild-Intervening solver rebuild]
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[unknown_prefix-non-UNSAT]
tests/unit/test_simon_candidate_history_observation.py::test_false_master_key_and_failed_fresh_verification_are_rejected
tests/unit/test_simon_candidate_history_observation.py::test_frozen_hash_sidecar_and_duplicate_fields
tests/unit/test_simon_cipher.py::test_one_round_encryption_matches_regression_state
tests/unit/test_simon_cipher.py::test_one_round_decryption_is_the_inverse
tests/unit/test_simon_cipher.py::test_full_known_answer_vector
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[1-expected0]
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[2-expected1]
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[4-expected2]
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[8-expected3]
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[16-expected4]
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[24-expected5]
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[32-expected6]
tests/unit/test_simon_cipher.py::test_key_schedule_matches_all_expected_round_keys
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[1]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[2]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[3]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[4]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[5]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[6]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[7]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[8]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[9]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[10]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[11]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[12]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[13]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[14]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[15]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[16]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[17]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[18]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[19]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[20]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[21]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[22]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[23]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[24]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[25]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[26]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[27]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[28]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[29]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[30]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[31]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[32]
tests/unit/test_simon_cipher.py::test_partial_decrypt_final_round_matches_prior_round
tests/unit/test_simon_cipher.py::test_simon_pair_difference_needs_candidate_sensitive_descent
tests/unit/test_simon_cipher.py::test_direct_pair_ranking_difference_has_r_minus_two_then_r_minus_three_depth
tests/unit/test_simon_cipher.py::test_deterministic_random_round_trips
tests/unit/test_simon_cipher.py::test_independent_reference_oracle_kat
tests/unit/test_simon_cipher.py::test_independent_reference_oracle_differential_random_comparison
tests/unit/test_simon_cipher.py::test_round_and_block_inputs_are_explicitly_masked
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block-0]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block--1]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block-33]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block-100]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block-0]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block--1]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block-33]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block-100]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[encrypt_block-1.0]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[encrypt_block-True]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[encrypt_block-32]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[encrypt_block-None]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[decrypt_block-1.0]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[decrypt_block-True]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[decrypt_block-32]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[decrypt_block-None]
tests/unit/test_simon_cipher.py::test_malformed_round_state_is_rejected
tests/unit/test_simon_cipher.py::test_key_schedule_validation
tests/unit/test_simon_cipher.py::test_primitives_properties
tests/unit/test_simon_classical_ranking.py::test_ondemand_stage1_removes_temporary_output_on_success_and_failure
tests/unit/test_simon_classical_ranking.py::test_sparse_full_distribution_row_has_exact_zero_bins
tests/unit/test_simon_classical_ranking.py::test_row_gather_score_matches_independent_scalar_source_index[None]
tests/unit/test_simon_classical_ranking.py::test_row_gather_score_matches_independent_scalar_source_index[0]
tests/unit/test_simon_classical_ranking.py::test_row_gather_score_matches_independent_scalar_source_index[4660]
tests/unit/test_simon_classical_ranking.py::test_complete_screen_and_rescore_use_deterministic_ascending_ties[None-7]
tests/unit/test_simon_classical_ranking.py::test_complete_screen_and_rescore_use_deterministic_ascending_ties[7-6]
tests/unit/test_simon_classical_ranking.py::test_nan_row_and_wrong_stage_fail_explicitly
tests/unit/test_simon_classical_ranking.py::test_tie_count_and_collision_count_semantics
tests/unit/test_simon_classical_ranking.py::test_rank_classical_domain_rescores_all_64_pairs
tests/unit/test_simon_confirmation_analysis.py::test_extract_record_success
tests/unit/test_simon_confirmation_analysis.py::test_compute_method_summary
tests/unit/test_simon_confirmation_analysis.py::test_analyze_all_comparators_complete
tests/unit/test_simon_confirmation_analysis.py::test_analyze_instance_mismatch_fails_closed
tests/unit/test_simon_confirmation_analysis.py::test_load_ledger_records_formats
tests/unit/test_simon_confirmation_analysis.py::test_generate_markdown_audit_report
tests/unit/test_simon_confirmation_analysis.py::test_missing_oracle_fails_closed
tests/unit/test_simon_confirmation_analysis.py::test_corrupt_sidecar_fails_closed
tests/unit/test_simon_confirmation_analysis.py::test_drift_rejected_in_main
tests/unit/test_simon_confirmation_guards.py::test_unapproved_manifest_rejected
tests/unit/test_simon_confirmation_guards.py::test_unauthorized_execution_rejected
tests/unit/test_simon_confirmation_guards.py::test_null_seed_rejected
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[260926]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[406737]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[98001]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[880001]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[880002]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[999999]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[669981]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[101001]
tests/unit/test_simon_confirmation_guards.py::test_invalid_selected_instances_rejected[None]
tests/unit/test_simon_confirmation_guards.py::test_invalid_selected_instances_rejected[16]
tests/unit/test_simon_confirmation_guards.py::test_invalid_selected_instances_rejected[48]
tests/unit/test_simon_confirmation_guards.py::test_invalid_selected_instances_rejected[128]
tests/unit/test_simon_confirmation_guards.py::test_invalid_selected_instances_rejected[32]
tests/unit/test_simon_confirmation_guards.py::test_missing_or_modified_methods_rejected
tests/unit/test_simon_confirmation_guards.py::test_modified_queue_control_seed_rejected
tests/unit/test_simon_confirmation_guards.py::test_spy_verifies_zero_worker_calls_on_preflight_rejection
tests/unit/test_simon_confirmation_guards.py::test_provenance_verification_fails_closed_on_hash_drift
tests/unit/test_simon_confirmation_guards.py::test_disjoint_family_certificate_rejects_duplicate_keys
tests/unit/test_simon_confirmation_guards.py::test_disjoint_family_certificate_detects_development_overlap
tests/unit/test_simon_confirmation_guards.py::test_disjoint_family_certificate_rejects_non_confirmation_role
tests/unit/test_simon_confirmation_guards.py::test_disjoint_family_certificate_build_and_verify_passes_on_fresh_keys
tests/unit/test_simon_confirmation_guards.py::test_resume_guards_reject_unflagged_rerun_on_existing_output
tests/unit/test_simon_confirmation_guards.py::test_resume_guards_reject_missing_checksum_sidecar
tests/unit/test_simon_confirmation_guards.py::test_resume_guards_reject_corrupted_checksum_sidecar
tests/unit/test_simon_confirmation_guards.py::test_resume_guards_reject_tampered_execution_identity
tests/unit/test_simon_confirmation_guards.py::test_resume_guards_reject_duplicate_cell
tests/unit/test_simon_confirmation_guards.py::test_resume_guards_reject_unverified_claimed_recovery
tests/unit/test_simon_confirmation_guards.py::test_resume_correctly_filters_completed_cells
tests/unit/test_simon_confirmation_guards.py::test_forged_zero_certificate_rejected_by_verify_disjoint_family_certificate
tests/unit/test_simon_confirmation_guards.py::test_generate_confirmation_dataset_fails_closed_without_authorization
tests/unit/test_simon_confirmation_guards.py::test_generate_confirmation_dataset_succeeds_when_authorized
tests/unit/test_simon_confirmation_guards.py::test_derive_confirmation_queue_seed_properties
tests/unit/test_simon_confirmation_launch.py::test_nested_profile_drift_is_rejected[resource_policy-solver_threads-2]
tests/unit/test_simon_confirmation_launch.py::test_nested_profile_drift_is_rejected[random_control_stream-prng_algorithm-another generator]
tests/unit/test_simon_confirmation_launch.py::test_nested_profile_drift_is_rejected[statistical_analysis_plan-analysis_freeze_status-changed]
tests/unit/test_simon_confirmation_launch.py::test_nested_profile_drift_is_rejected[candidate_sat_specification-formula_variables-1857]
tests/unit/test_simon_confirmation_launch.py::test_opened_eight_pair_demo_is_in_overlap_union
tests/unit/test_simon_confirmation_launch.py::test_generation_rejects_unapproved_before_any_data_draw
tests/unit/test_simon_confirmation_launch.py::test_generation_refuses_existing_artifact
tests/unit/test_simon_confirmation_launch.py::test_execution_pins_reject_source_change_before_runtime_probe
tests/unit/test_simon_confirmation_launch.py::test_analysis_cannot_accept_missing_confirmation_identity
tests/unit/test_simon_confirmation_launch.py::test_success_flag_cannot_replace_missing_fresh_verification
tests/unit/test_simon_confirmation_launch.py::test_missing_component_time_is_unknown
tests/unit/test_simon_confirmation_launch.py::test_training_replay_matches_original_generator[None]
tests/unit/test_simon_confirmation_launch.py::test_training_replay_matches_original_generator[64]
tests/unit/test_simon_confirmation_launch.py::test_real_frozen_sample_size_cannot_drift
tests/unit/test_simon_confirmation_launch.py::test_same_package_version_changed_native_bytes_changes_identity
tests/unit/test_simon_confirmation_launch.py::test_contradictory_authorization_is_rejected[seed]
tests/unit/test_simon_confirmation_launch.py::test_contradictory_authorization_is_rejected[gate]
tests/unit/test_simon_confirmation_launch.py::test_contradictory_authorization_is_rejected[blocker]
tests/unit/test_simon_confirmation_launch.py::test_confirmation_analysis_rejects_equal_partial_grid
tests/unit/test_simon_exact_ddt_equivalence.py::test_toy_width_exact_transition_equivalence
tests/unit/test_simon_exact_ddt_equivalence.py::test_queried_row_reconstruction_toy
tests/unit/test_simon_exact_ddt_equivalence.py::test_full_width_16bit_affine_basis_rank
tests/unit/test_simon_faithful_b1.py::test_simon_gohr_resnet_shapes_and_param_count
tests/unit/test_simon_faithful_b1.py::test_shared_simon_pair_encoder
tests/unit/test_simon_faithful_b1.py::test_simon_pair_group_ranker_permutation_invariance
tests/unit/test_simon_faithful_b1.py::test_simon_pair_group_ranker_flexible_pair_count[1]
tests/unit/test_simon_faithful_b1.py::test_simon_pair_group_ranker_flexible_pair_count[16]
tests/unit/test_simon_faithful_b1.py::test_simon_pair_group_ranker_flexible_pair_count[32]
tests/unit/test_simon_faithful_b1.py::test_simon_pair_group_ranker_flexible_pair_count[64]
tests/unit/test_simon_faithful_b1.py::test_simon_pair_group_ranker_flexible_pair_count[128]
tests/unit/test_simon_faithful_b1.py::test_b0_to_b1_weight_transfer
tests/unit/test_simon_faithful_b1.py::test_batch_norm_single_sample_fallback
tests/unit/test_simon_faithful_b1.py::test_faithful_feature_extraction
tests/unit/test_simon_faithful_b1.py::test_true_r10_key_features_equal_independent_nine_round_oracle
tests/unit/test_simon_faithful_b1.py::test_b1_group_mean_is_invariant_to_fourfold_pair_duplication
tests/unit/test_simon_faithful_b1.py::test_faithful_ranking_pair_count_enforcement
tests/unit/test_simon_faithful_b1.py::test_faithful_deterministic_tie_breaking
tests/unit/test_simon_faithful_b1.py::test_faithful_hybrid_8parent_queue
tests/unit/test_simon_faithful_b1.py::test_audit_contrast_faithful_vs_toy
tests/unit/test_simon_higher_round.py::test_simons11_schedule_word_order_and_chronological_trailing_window
tests/unit/test_simon_higher_round.py::test_peel_order_matches_independent_reduced_round_reference
tests/unit/test_simon_higher_round.py::test_queue_order_ties_are_deterministic_and_duplicate_paths_keep_first
tests/unit/test_simon_higher_round.py::test_native_simon11_completion_checks_128_known_blocks_and_two_fresh_pairs
tests/unit/test_simon_higher_round.py::test_run_cell_success_uses_stubbed_rankers_and_fresh_verification
tests/unit/test_simon_higher_round.py::test_run_cell_rejects_completion_that_fails_fresh_verification
tests/unit/test_simon_higher_round.py::test_run_cell_stops_dispatch_after_ranking_crosses_deadline
tests/unit/test_simon_higher_round.py::test_run_cell_does_not_accept_verification_completed_after_deadline
tests/unit/test_simon_higher_round.py::test_run_cell_rejects_malformed_instance_before_ranking[rounds]
tests/unit/test_simon_higher_round.py::test_run_cell_rejects_malformed_instance_before_ranking[pairs]
tests/unit/test_simon_higher_round.py::test_run_cell_rejects_malformed_instance_before_ranking[verification_pairs]
tests/unit/test_simon_higher_round.py::test_run_cell_rejects_malformed_instance_before_ranking[overlap]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_accepts_only_demo_without_explicit_role
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-98001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-97001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-99001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-101001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-201001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-300001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-406737]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-669981]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-6503664708853956672]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-701004]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-98001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-97001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-99001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-101001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-201001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-300001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-406737]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-669981]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-6503664708853956672]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-701004]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-98001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-97001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-99001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-101001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-201001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-300001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-406737]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-669981]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-6503664708853956672]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-701004]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-98001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-97001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-99001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-101001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-201001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-300001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-406737]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-669981]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-6503664708853956672]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-701004]
tests/unit/test_simon_higher_round.py::test_parse_args_requires_run_parameters_and_limits_tensorflow_threads
tests/unit/test_simon_higher_round.py::test_ledger_creation_is_exclusive_and_append_requires_existing_file
tests/unit/test_simon_higher_round.py::test_classical_and_hybrid_stages_batch_parents_once[S11-C]
tests/unit/test_simon_higher_round.py::test_classical_and_hybrid_stages_batch_parents_once[S11-H]
tests/unit/test_simon_higher_round.py::test_row_score_ties_use_word_order_and_nonfinite_scores_fail
tests/unit/test_simon_higher_round.py::test_bad_table_header_is_rejected_without_dense_allocation
tests/unit/test_simon_higher_round.py::test_batch_rows_reject_invalid_probabilities[-1.0]
tests/unit/test_simon_higher_round.py::test_batch_rows_reject_invalid_probabilities[nan]
tests/unit/test_simon_higher_round.py::test_batch_rows_reject_invalid_probabilities[1.01]
tests/unit/test_simon_higher_round.py::test_header_pins_sources_and_only_assets_used_by_selected_arm
tests/unit/test_simon_higher_round.py::test_rejected_cli_never_draws_data_or_initializes_workers[existing_ledger]
tests/unit/test_simon_higher_round.py::test_rejected_cli_never_draws_data_or_initializes_workers[protected_seed]
tests/unit/test_simon_higher_round.py::test_rejected_cli_never_draws_data_or_initializes_workers[missing_threads]
tests/unit/test_simon_higher_round.py::test_rejected_cli_never_draws_data_or_initializes_workers[role_specific_seed]
tests/unit/test_simon_higher_round.py::test_rejected_cli_never_draws_data_or_initializes_workers[unfrozen_role]
tests/unit/test_simon_higher_round.py::test_loader_time_is_charged_and_no_rank_starts_at_exact_deadline
tests/unit/test_simon_higher_round.py::test_vv10_pin_registration_is_scoped_to_one_load
tests/unit/test_simon_higher_round.py::test_serial_schedule_rotates_by_absolute_key_index_with_stubbed_cells
tests/unit/test_simon_higher_round.py::test_wrong_input_difference_is_rejected_before_model_load
tests/unit/test_simon_higher_round.py::test_posthoc_truth_time_does_not_turn_ontime_recovery_into_deadline
tests/unit/test_simon_operating_points_map.py::test_cohort_identity_slicing_and_divergence
tests/unit/test_simon_operating_points_map.py::test_deterministic_replay
tests/unit/test_simon_operating_points_map.py::test_independent_solver_construction
tests/unit/test_simon_operating_points_map.py::test_worker_solve_budget_exhausted
tests/unit/test_simon_operating_points_map.py::test_worker_solve_timeout_classification
tests/unit/test_simon_operating_points_map.py::test_fresh_pair_verification_rejects_wrong_key
tests/unit/test_simon_operating_points_map.py::test_run_map_workers_guard
tests/unit/test_simon_operating_points_map.py::test_parent_deadline_kills_overrunning_worker
tests/unit/test_simon_operating_points_map.py::test_worker_finishes_just_within_deadline
tests/unit/test_simon_operating_points_map.py::test_deadline_classifications_separate_outcomes
tests/unit/test_simon_operating_points_map.py::test_run_map_r10_serial_profile_cells
tests/unit/test_simon_operating_points_map.py::test_resume_rejects_changed_profile_or_worker_count
tests/unit/test_simon_operating_window.py::test_observations_are_ciphertexts_not_differential_pairs
tests/unit/test_simon_operating_window.py::test_demo_keys_are_distinct_and_pairing_is_stable
tests/unit/test_simon_operating_window.py::test_wrong_tuple_probe_is_unique_truth_excluded_and_reproducible
tests/unit/test_simon_operating_window.py::test_grid_screen_applies_paired_counts_and_censored_median
tests/unit/test_simon_operating_window.py::test_checkpoint_refuses_overwrite_and_bad_resume
tests/unit/test_simon_operating_window.py::test_one_true_key_control_matches_reference_and_fresh_verification
tests/unit/test_simon_operating_window.py::test_solver_timeout_is_unresolved_not_unsat
tests/unit/test_simon_operating_window.py::test_detached_checksum_refuses_modified_artifact
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[0-3]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[0-4]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[0-6]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[1-3]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[1-4]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[1-6]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[3-3]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[3-4]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[3-6]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[7-3]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[7-4]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[7-6]
tests/unit/test_simon_predicted_ddt.py::test_full_width_affine_support_matches_exhaustive_scalar_histogram[0]
tests/unit/test_simon_predicted_ddt.py::test_full_width_affine_support_matches_exhaustive_scalar_histogram[1]
tests/unit/test_simon_predicted_ddt.py::test_full_width_affine_support_matches_exhaustive_scalar_histogram[64]
tests/unit/test_simon_predicted_ddt.py::test_full_width_affine_support_matches_exhaustive_scalar_histogram[128]
tests/unit/test_simon_predicted_ddt.py::test_full_width_affine_support_matches_exhaustive_scalar_histogram[4660]
tests/unit/test_simon_predicted_ddt.py::test_full_width_affine_support_matches_exhaustive_scalar_histogram[65535]
tests/unit/test_simon_predicted_ddt.py::test_affine_kernel_matches_pinned_source_probability_for_all_outputs[0]
tests/unit/test_simon_predicted_ddt.py::test_affine_kernel_matches_pinned_source_probability_for_all_outputs[1]
tests/unit/test_simon_predicted_ddt.py::test_affine_kernel_matches_pinned_source_probability_for_all_outputs[64]
tests/unit/test_simon_predicted_ddt.py::test_affine_kernel_matches_pinned_source_probability_for_all_outputs[4660]
tests/unit/test_simon_predicted_ddt.py::test_affine_kernel_matches_pinned_source_probability_for_all_outputs[65535]
tests/unit/test_simon_predicted_ddt.py::test_reference_rejects_full_width_generation_and_malformed_inputs
tests/unit/test_simon_published_vv_ranking.py::test_published_log_odds_polarity_matches_source_formula
tests/unit/test_simon_published_vv_ranking.py::test_published_log_odds_rejects_malformed_outputs[malformed0]
tests/unit/test_simon_published_vv_ranking.py::test_published_log_odds_rejects_malformed_outputs[malformed1]
tests/unit/test_simon_published_vv_ranking.py::test_published_log_odds_rejects_malformed_outputs[malformed2]
tests/unit/test_simon_published_vv_ranking.py::test_published_log_odds_rejects_malformed_outputs[malformed3]
tests/unit/test_simon_published_vv_ranking.py::test_published_log_odds_rejects_malformed_outputs[malformed4]
tests/unit/test_simon_published_vv_ranking.py::test_complete_domain_uses_candidate_word_ties[None]
tests/unit/test_simon_published_vv_ranking.py::test_complete_domain_uses_candidate_word_ties[7]
tests/unit/test_simon_published_vv_ranking.py::test_neural_ranker_rejects_invalid_pair_budget
tests/unit/test_simon_published_vv_ranking.py::test_neural_candidate_pair_order_matches_independent_scalar_oracle[None]
tests/unit/test_simon_published_vv_ranking.py::test_neural_candidate_pair_order_matches_independent_scalar_oracle[22136]
tests/unit/test_simon_published_vv_ranking.py::test_vv9_and_vv8_feature_and_score_aggregation_pinned_reference
tests/unit/test_simon_published_vv_ranking.py::test_rank_neural_domain_rescores_all_64_pairs
tests/unit/test_simon_ranking_features.py::test_source_word_and_msb_bit_order
tests/unit/test_simon_ranking_features.py::test_vectorized_candidate_states_match_scalar_and_independent_oracle[None]
tests/unit/test_simon_ranking_features.py::test_vectorized_candidate_states_match_scalar_and_independent_oracle[0]
tests/unit/test_simon_ranking_features.py::test_vectorized_candidate_states_match_scalar_and_independent_oracle[65535]
tests/unit/test_simon_ranking_features.py::test_correct_keys_have_independent_seven_and_six_round_differences
tests/unit/test_simon_ranking_features.py::test_direct_difference_expression_and_candidate_collision
tests/unit/test_simon_ranking_features.py::test_malformed_candidate_feature_inputs_are_rejected[pairs0-candidates0-None-ValueError]
tests/unit/test_simon_ranking_features.py::test_malformed_candidate_feature_inputs_are_rejected[pairs1-candidates1-None-ValueError]
tests/unit/test_simon_ranking_features.py::test_malformed_candidate_feature_inputs_are_rejected[pairs2-candidates2-None-ValueError]
tests/unit/test_simon_ranking_features.py::test_malformed_candidate_feature_inputs_are_rejected[pairs3-candidates3-None-TypeError]
tests/unit/test_simon_ranking_features.py::test_malformed_candidate_feature_inputs_are_rejected[pairs4-candidates4--1-ValueError]
tests/unit/test_simon_ranking_features.py::test_malformed_candidate_feature_inputs_are_rejected[pairs5-candidates5-None-ValueError]
tests/unit/test_simon_sat.py::test_simon_sat_model_matches_reference_cipher_on_kat[1]
tests/unit/test_simon_sat.py::test_simon_sat_model_matches_reference_cipher_on_kat[2]
tests/unit/test_simon_sat.py::test_simon_sat_model_matches_reference_cipher_on_kat[4]
tests/unit/test_simon_sat.py::test_simon_sat_model_matches_reference_cipher_on_kat[6]
tests/unit/test_simon_sat.py::test_simon_sat_model_matches_reference_cipher_on_kat[8]
tests/unit/test_simon_sat.py::test_simon_sat_exact_master_key_recovery_with_trailing_assumptions
tests/unit/test_simon_sat.py::test_solver_isolation_between_models
tests/unit/test_simon_sat.py::test_conflicting_assumptions_yield_unsat
tests/unit/test_simon_sat.py::test_timeout_state_classification
tests/unit/test_simon_sat.py::test_master_key_verification_fails_closed
tests/unit/test_simon_sat.py::test_dimacs_export_and_native_cli_solver
tests/unit/test_simon_sat.py::test_simon_pure_cnf_export_solved_by_cadical_and_cms
tests/unit/test_simon_sat_300s_control.py::test_worker_receives_only_attack_and_verification_observations
tests/unit/test_simon_sat_300s_control.py::test_worker_classifies_limit_as_unresolved
tests/unit/test_simon_sat_300s_control.py::test_worker_rejects_truth_bearing_or_wrong_sized_payload
tests/unit/test_simon_sat_300s_control.py::test_parent_watchdog_is_unconditional_failure
tests/unit/test_simon_sat_300s_control.py::test_resume_refuses_source_drift_and_duplicate_records
tests/unit/test_simon_stage2_control.py::test_fixed_job_grid_and_ordered_ciphertext_only_projection
tests/unit/test_simon_stage2_control.py::test_public_payload_and_words_reject_extra_truth_and_non_integer_values
tests/unit/test_simon_stage2_control.py::test_payload_contract_and_rank_sum_queue_order
tests/unit/test_simon_stage2_control.py::test_full_binding_accepts_d053_and_rejects_old_pilot[D-051]
tests/unit/test_simon_stage2_control.py::test_full_binding_accepts_d053_and_rejects_old_pilot[D-052]
tests/unit/test_simon_stage2_control.py::test_full_analyzer_keeps_8_key_denominators_and_uses_phase_costs
tests/unit/test_simon_stage2_control.py::test_analyzer_separates_structural_parent_miss_from_partial_stage2_unavailability
tests/unit/test_simon_stage2_control.py::test_tombstones_keep_unknown_dispatch_and_proven_no_worker_distinct
tests/unit/test_simon_stage2_control.py::test_worker_cleanup_overshoot_is_preserved_without_violating_phase_caps
tests/unit/test_simon_stage2_control.py::test_interrupted_success_receipt_does_not_invent_queue_persistence
tests/unit/test_simon_stage2_control.py::test_only_exact_sealed_orphan_tombstone_is_analyzable
tests/unit/test_simon_stage2_control.py::test_pilot_returns_non_scientific_without_touching_truth_and_closure_is_strict
tests/unit/test_simon_stage2_control.py::test_interpreter_file_pin_preserves_lexical_symlink
tests/unit/test_simon_stage2_receipts.py::test_completed_pilot_accepts_bound_rss_policy_and_ac_receipts
tests/unit/test_simon_stage2_receipts.py::test_battery_is_valid_under_explicit_author_policy
tests/unit/test_simon_stage2_receipts.py::test_pilot_requires_observed_worker_member
tests/unit/test_simon_stage2_receipts.py::test_rss_stop_accepts_only_a_real_over_limit_sample
tests/unit/test_simon_stage2_receipts.py::test_telemetry_must_match_observed_runtime_policy[pid-3999-Telemetry PID]
tests/unit/test_simon_stage2_receipts.py::test_telemetry_must_match_observed_runtime_policy[parent_pid-2999-parent PID]
tests/unit/test_simon_stage2_receipts.py::test_telemetry_must_match_observed_runtime_policy[git_commit-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb-source commit]
tests/unit/test_simon_stage2_receipts.py::test_telemetry_must_match_observed_runtime_policy[tmpdir_safe-False-TMPDIR]
tests/unit/test_simon_stage2_receipts.py::test_telemetry_must_match_observed_runtime_policy[thread_environment-value4-thread environment]
tests/unit/test_simon_stage2_receipts.py::test_telemetry_must_match_observed_runtime_policy[qos_class-9-QoS]
tests/unit/test_simon_stage2_receipts.py::test_process_policy_errno_is_not_hidden_by_accepted_flag
tests/unit/test_simon_stage2_receipts.py::test_sleep_assertion_owner_is_bound_to_guardian
tests/unit/test_simon_stage2_receipts.py::test_corrupt_sample_sidecar_fails_closed
tests/unit/test_simon_stage2_receipts.py::test_partial_power_only_sample_is_allowed_without_worker_observation
tests/unit/test_simon_stage2_receipts.py::test_missing_samples_on_partial_row_remain_unobserved_not_zero_rss
tests/unit/test_simon_stage2_watchdog.py::test_accepts_loaded_then_all_eight_calls_and_advances_deadline
tests/unit/test_simon_stage2_watchdog.py::test_waits_briefly_for_event_sidecar_without_rejecting_complete_event
tests/unit/test_simon_stage2_watchdog.py::test_deadline_stops_hangs_and_retains_only_completed_calls[hang_load-load_timeout-0]
tests/unit/test_simon_stage2_watchdog.py::test_deadline_stops_hangs_and_retains_only_completed_calls[hang_call-rank_timeout-1]
tests/unit/test_simon_stage2_watchdog.py::test_deadline_stops_hangs_and_retains_only_completed_calls[hang_finalize-rank_timeout-2]
tests/unit/test_simon_stage2_watchdog.py::test_corrupt_or_nonsequential_journal_fails_closed[future_event]
tests/unit/test_simon_stage2_watchdog.py::test_corrupt_or_nonsequential_journal_fails_closed[bad_result_hash]
tests/unit/test_simon_stage2_watchdog.py::test_corrupt_or_nonsequential_journal_fails_closed[wrong_result_identity]
tests/unit/test_simon_stage2_watchdog.py::test_corrupt_or_nonsequential_journal_fails_closed[wrong_event_identity]
tests/unit/test_simon_stage2_watchdog.py::test_corrupt_or_nonsequential_journal_fails_closed[backdated_second_call]
tests/unit/test_simon_stage2_watchdog.py::test_campaign_deadline_caps_loading_deadline
tests/unit/test_simon_stage2_watchdog.py::test_configuration_drift_is_rejected_before_worker_dispatch
tests/unit/test_simon_stage2_watchdog.py::test_coordinator_loss_before_dispatch_writes_stop_receipt
tests/unit/test_simon_stage2_watchdog.py::test_coordinator_loss_after_dispatch_stops_worker_group
tests/unit/test_simon_stage2_watchdog.py::test_live_groups_matches_only_exact_stage2_argv
tests/unit/test_simon_stage2_worker.py::test_call_schedule_preserves_all_eight_parent_ordinals
tests/unit/test_simon_stage2_worker.py::test_rank_arguments_keep_the_frozen_screen_rescore_and_depths
tests/unit/test_simon_stage2_worker.py::test_checkpoint_loader_selects_vv9_vv8_and_ddt7_depths
tests/unit/test_simon_stage2_worker.py::test_rank_result_rejects_duplicate_and_wrong_width_orders
tests/unit/test_simon_stage2_worker.py::test_real_ranker_uint16_array_crosses_strict_json_boundary[vv9-None]
tests/unit/test_simon_stage2_worker.py::test_real_ranker_uint16_array_crosses_strict_json_boundary[vv8-123]
tests/unit/test_simon_stage2_worker.py::test_real_ranker_uint16_array_crosses_strict_json_boundary[ddt7-123]
tests/unit/test_simon_stage2_worker.py::test_float_rank_array_is_not_silently_truncated_to_words
tests/unit/test_simon_stage2_worker.py::test_worker_loads_one_checkpoint_and_passes_ciphertext_only_to_eight_calls
tests/unit/test_simon_stage2_worker.py::test_namespace_bootstrap_skips_both_package_initializers
tests/unit/test_simon_stage2_worker.py::test_import_blocker_rejects_solver_and_verifier_modules
tests/unit/test_simon_stage2_worker.py::test_importing_worker_module_does_not_import_project_or_solver_modules
tests/unit/test_solver_isolation.py::test_every_compared_method_receives_distinct_object_identity
tests/unit/test_solver_isolation.py::test_reordering_method_execution_does_not_alter_results
tests/unit/test_solver_isolation.py::test_candidate_assumptions_remain_incremental_within_one_method
tests/unit/test_solver_isolation.py::test_solver_state_cannot_be_passed_across_methods
tests/unit/test_solver_isolation.py::test_resume_logic_recreates_method_local_solver_state
tests/unit/test_solver_isolation.py::test_seed_role_validation_rejects_quarantined_seed_97001
tests/unit/test_solver_isolation.py::test_seed_role_validation_rejects_retired_seed_99001
tests/unit/test_solver_isolation.py::test_cli_default_contains_no_protected_research_seed
tests/unit/test_solver_isolation.py::test_historical_seed_98001_rejected_for_model_selection
tests/unit/test_solver_isolation.py::test_seed_role_validation_accepts_owner_seeds_for_their_roles
tests/unit/test_solver_isolation.py::test_seed_role_validation_rejects_cross_role_contamination
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[   PreventSystemSleep              1\n-True]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[\tPreventUserIdleSystemSleep\t1\t\n-True]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[PreventSystemSleep 0\nPreventUserIdleSystemSleep 1\n-True]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[PreventSystemSleep 1-True]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[PreventSystemSleep 0\nPreventUserIdleSystemSleep 0\n-False]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[PreventSystemSleep 10\n-False]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[OtherPreventSystemSleep 1\n-False]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[pid 42: PreventSystemSleep 1\n-False]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[PreventSystemSleep 1 owner detail\n-False]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[\n-False]
tests/unit/test_worker_telemetry.py::test_sleep_assertions_unsupported_off_darwin
tests/unit/test_worker_telemetry.py::test_live_worker_telemetry_observation
tests/unit/test_worker_telemetry.py::test_adversarial_qos9_zero_priority_rejected
tests/unit/test_worker_telemetry.py::test_unspecified_or_failed_qos_rejected
tests/unit/test_worker_telemetry.py::test_failed_priority_or_io_queries_rejected
tests/unit/test_worker_telemetry.py::test_darwin_bg_state_rejected
tests/unit/test_worker_telemetry.py::test_io_throttle_rejected
tests/unit/test_worker_telemetry.py::test_degraded_process_priority_rejected
tests/unit/test_worker_telemetry.py::test_unsafe_tmpdir_rejected
tests/unit/test_worker_telemetry.py::test_symlink_escape_tmpdir_rejected
tests/unit/test_worker_telemetry.py::test_valid_worker_certification_passes
tests/unit/test_worker_telemetry.py::test_save_worker_receipt_with_sidecar
tests/unit/test_worker_telemetry.py::test_stale_or_mismatched_receipt_identity
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[False-1]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[False-2]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[False-3]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[False-4]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[False-5]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[True-1]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[True-2]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[True-3]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[True-4]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[True-5]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[False-4]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[False-5]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[False-6]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[False-7]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[True-4]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[True-5]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[True-6]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[True-7]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_edge_cases
tests/unit/test_xor_to_cnf.py::test_convert_xor_clauses_to_cnf

1466 tests collected in 10.57s

Exit code: 0
Elapsed seconds: 12.063

```

```text
COMMAND: <work>/hamza-venv/Scripts/python.exe -m pytest --collect-only -q tests/unit
tests/unit/test_attack_instances.py::test_attack_generation_is_identical_for_a_fixed_seed
tests/unit/test_attack_instances.py::test_different_attack_seeds_produce_different_instances
tests/unit/test_attack_instances.py::test_attack_pairs_share_one_key_and_the_selected_difference
tests/unit/test_attack_instances.py::test_attack_split_keys_do_not_overlap
tests/unit/test_attack_instances.py::test_attack_export_checksums_and_independent_load
tests/unit/test_attack_instances.py::test_attack_loader_rejects_tampered_ciphertext
tests/unit/test_attack_instances.py::test_invalid_attack_configuration_is_rejected[0-2-7-train]
tests/unit/test_attack_instances.py::test_invalid_attack_configuration_is_rejected[2-0-7-train]
tests/unit/test_attack_instances.py::test_invalid_attack_configuration_is_rejected[2-2-0-train]
tests/unit/test_attack_instances.py::test_invalid_attack_configuration_is_rejected[2-2-23-train]
tests/unit/test_attack_instances.py::test_invalid_attack_configuration_is_rejected[2-2-7-holdout]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-observation]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[truth-leaked-into-public]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-trailing-key-unit]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-master-truth]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-verification-pair]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-variable-map]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-bit-order]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-native-formula]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_single_forged_artifact[forged-pure-cnf-formula]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_forged_catalog_metadata[counts]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_forged_catalog_metadata[known-observations]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_rejects_forged_catalog_metadata[assumption-units]
tests/unit/test_backend_artifact_semantics.py::test_semantic_validator_accepts_canonical_minimal_bundle
tests/unit/test_backend_artifact_semantics.py::test_catalog_observations_are_checked_against_formula_not_file_hash_only
tests/unit/test_backend_artifacts.py::test_solver_public_boundary_and_fresh_verification[speck32_64-7]
tests/unit/test_backend_artifacts.py::test_solver_public_boundary_and_fresh_verification[simon32_64-10]
tests/unit/test_backend_completed_evidence.py::test_signed_xor_parity_and_unit_clause[x1 2 0]
tests/unit/test_backend_completed_evidence.py::test_signed_xor_parity_and_unit_clause[x 1 2 0]
tests/unit/test_backend_completed_evidence.py::test_negative_xor_literal_changes_parity
tests/unit/test_backend_completed_evidence.py::test_unsatisfied_disjunction_or_empty_constraint_is_rejected[1 -2 0]
tests/unit/test_backend_completed_evidence.py::test_unsatisfied_disjunction_or_empty_constraint_is_rejected[0]
tests/unit/test_backend_completed_evidence.py::test_unsatisfied_disjunction_or_empty_constraint_is_rejected[x0]
tests/unit/test_backend_completed_evidence.py::test_incomplete_nonboolean_and_out_of_range_assignments_are_rejected[assignment0]
tests/unit/test_backend_completed_evidence.py::test_incomplete_nonboolean_and_out_of_range_assignments_are_rejected[assignment1]
tests/unit/test_backend_completed_evidence.py::test_incomplete_nonboolean_and_out_of_range_assignments_are_rejected[assignment2]
tests/unit/test_backend_completed_evidence.py::test_malformed_header_range_or_record_is_rejected[p cnf 2 2\n1 0\n]
tests/unit/test_backend_completed_evidence.py::test_malformed_header_range_or_record_is_rejected[p cnf 2 1\n3 0\n]
tests/unit/test_backend_completed_evidence.py::test_malformed_header_range_or_record_is_rejected[p cnf 2 1\n1 0 2 0\n]
tests/unit/test_backend_completed_evidence.py::test_decode_is_lsb_first_and_uses_variable_map_order
tests/unit/test_backend_completed_evidence.py::test_forced_extension_handles_negative_xor_and_auxiliary_clause
tests/unit/test_backend_completed_evidence.py::test_contradictory_reference_witness_is_rejected[cnf0-xors0]
tests/unit/test_backend_completed_evidence.py::test_contradictory_reference_witness_is_rejected[cnf1-xors1]
tests/unit/test_backend_completed_evidence.py::test_unforced_reference_extension_is_unavailable_not_filled_arbitrarily
tests/unit/test_backend_environment.py::test_power_observation_requires_successful_ac_and_sleep[Now drawing from 'AC Power'-PreventSystemSleep 1-0-True]
tests/unit/test_backend_environment.py::test_power_observation_requires_successful_ac_and_sleep[Now drawing from 'Battery Power'-PreventSystemSleep 1-0-False]
tests/unit/test_backend_environment.py::test_power_observation_requires_successful_ac_and_sleep[Now drawing from 'AC Power'-PreventSystemSleep 0-0-False]
tests/unit/test_backend_environment.py::test_power_observation_requires_successful_ac_and_sleep[Now drawing from 'AC Power'-PreventSystemSleep 1-1-False]
tests/unit/test_backend_environment.py::test_unavailable_power_query_rejects
tests/unit/test_backend_environment.py::test_process_policy_does_not_substitute_for_thread_qos[0-0-0-True]
tests/unit/test_backend_environment.py::test_process_policy_does_not_substitute_for_thread_qos[0-1-0-False]
tests/unit/test_backend_environment.py::test_process_policy_does_not_substitute_for_thread_qos[1-0-0-False]
tests/unit/test_backend_environment.py::test_process_policy_does_not_substitute_for_thread_qos[0-0-3-False]
tests/unit/test_backend_environment.py::test_unsupported_platform_is_explicit
tests/unit/test_backend_evidence.py::test_audit_reparses_complete_native_sat_model
tests/unit/test_backend_evidence.py::test_audit_accepts_native_unsat_as_recorded_alarm
tests/unit/test_backend_evidence.py::test_wrapper_audit_does_not_claim_a_captured_native_model
tests/unit/test_backend_evidence.py::test_late_worker_recovery_is_preserved_but_not_credited
tests/unit/test_backend_evidence.py::test_interruption_reports_unavailable_time_without_normal_receipts
tests/unit/test_backend_evidence.py::test_missing_environment_result_requires_parent_stop_proof[coordinator_stop-True]
tests/unit/test_backend_evidence.py::test_missing_environment_result_requires_parent_stop_proof[coordinator_stop-False]
tests/unit/test_backend_evidence.py::test_missing_environment_result_requires_parent_stop_proof[worker_exit-True]
tests/unit/test_backend_evidence.py::test_coordinator_record_mutation_is_rejected
tests/unit/test_backend_evidence.py::test_nonzero_worker_exit_receipt_is_rejected_even_when_resealed
tests/unit/test_backend_evidence.py::test_unsealed_or_tampered_raw_solver_output_is_rejected
tests/unit/test_backend_evidence.py::test_guardian_must_itself_certify_group_absence[False]
tests/unit/test_backend_evidence.py::test_guardian_must_itself_certify_group_absence[None]
tests/unit/test_backend_evidence.py::test_guardian_must_itself_certify_group_absence[true]
tests/unit/test_backend_evidence.py::test_resealed_solver_output_must_still_match_worker_status
tests/unit/test_backend_evidence.py::test_raw_worker_key_must_match_coordinator_result
tests/unit/test_backend_evidence.py::test_worker_directory_cannot_be_redirected_or_symlinked
tests/unit/test_backend_evidence.py::test_raw_artifact_paths_must_be_basenames
tests/unit/test_backend_evidence.py::test_interrupted_record_must_preserve_original_group_and_invocation
tests/unit/test_backend_formula.py::test_formula_has_independent_reference_witness_and_chronological_assumptions[speck32_64-4-key0]
tests/unit/test_backend_formula.py::test_formula_has_independent_reference_witness_and_chronological_assumptions[simon32_64-5-key1]
tests/unit/test_backend_formula.py::test_unguided_build_never_reads_truth_or_verification_fields[speck32_64-4-key0]
tests/unit/test_backend_formula.py::test_unguided_build_never_reads_truth_or_verification_fields[simon32_64-5-key1]
tests/unit/test_backend_formula.py::test_observation_prefix_does_not_read_unselected_pairs
tests/unit/test_backend_formula.py::test_reference_key_conflicts_with_corrupted_known_ciphertext
tests/unit/test_backend_formula.py::test_speck_native_output_matches_existing_encoder_bytes
tests/unit/test_backend_formula.py::test_speck_conditioned_output_matches_existing_pinned_key_units
tests/unit/test_backend_formula.py::test_simon_native_output_matches_existing_recorder_bytes
tests/unit/test_backend_formula.py::test_simon_conditioned_output_matches_existing_assumption_serializer
tests/unit/test_backend_formula.py::test_simon_builder_never_constructs_solver
tests/unit/test_backend_formula.py::test_native_only_build_skips_pure_cnf_conversion
tests/unit/test_backend_formula.py::test_renderer_preserves_signed_rhs_and_empty_xor_semantics
tests/unit/test_backend_formula.py::test_long_xor_conversion_allocates_fresh_auxiliaries_repeatably
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[0-0-fields0]
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[3-0-fields1]
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[6-0-fields2]
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[2--1-fields3]
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[2-5-fields4]
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[2-1-fields5]
tests/unit/test_backend_formula.py::test_invalid_counts_and_trailing_values_fail_closed[2-1-fields6]
tests/unit/test_backend_formula.py::test_invalid_rounds_and_observation_words_fail_closed
tests/unit/test_backend_formula.py::test_repeated_builds_are_deterministic_and_do_not_mutate_inputs
tests/unit/test_backend_ranking_depth_figure.py::test_plot_totals_agree_with_independent_raw_audit
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[duplicate]
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[missing]
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[string_flag]
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[wrong_outcome]
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[late]
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[nan]
tests/unit/test_backend_ranking_depth_figure.py::test_reject_misleading_figure_inputs[premature]
tests/unit/test_backend_ranking_depth_figure.py::test_corrupted_sidecar_rejected
tests/unit/test_backend_ranking_depth_n32_figure.py::test_canonical_source_hash_and_sidecar
tests/unit/test_backend_ranking_depth_n32_figure.py::test_n32_plot_totals_and_par2_agree_with_audit
tests/unit/test_backend_ranking_depth_n32_figure.py::test_containment_invariants
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[duplicate_cell]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[missing_cell]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[string_flag]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[wrong_outcome_recovered_but_false]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[wrong_outcome_unrecovered_but_true]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[unexpected_unrecovered_outcome]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[late_recovery_over_300s]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[premature_deadline_under_300s]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[nan_wall_time]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[negative_wall_time]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[unmeasured_containment]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[containment_key_mismatch]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_reject_defective_figure_inputs[wrong_cell_count]
tests/unit/test_backend_ranking_depth_n32_figure.py::test_corrupted_sidecar_or_hash_rejected
tests/unit/test_backend_ranking_depth_n32_figure.py::test_generate_end_to_end_in_tmp_path
tests/unit/test_backend_solver_io.py::test_native_gauss_profiles_differ_only_by_matrix_limit
tests/unit/test_backend_solver_io.py::test_native_default_keeps_cms_defaults_and_pins_one_thread
tests/unit/test_backend_solver_io.py::test_cnf_profile_changes_only_input_representation_from_x
tests/unit/test_backend_solver_io.py::test_cadical_uses_quiet_mode_and_preserves_path_as_one_argument
tests/unit/test_backend_solver_io.py::test_solver_arguments_reject_unknown_profiles_and_missing_paths[unknown-/bin/cms-/bin/cadical-formula.cnf]
tests/unit/test_backend_solver_io.py::test_solver_arguments_reject_unknown_profiles_and_missing_paths[cms_native_default--/bin/cadical-formula.cnf]
tests/unit/test_backend_solver_io.py::test_solver_arguments_reject_unknown_profiles_and_missing_paths[cadical_cnf-/bin/cms--formula.cnf]
tests/unit/test_backend_solver_io.py::test_solver_arguments_reject_unknown_profiles_and_missing_paths[cms_native_default-/bin/cms-/bin/cadical-]
tests/unit/test_backend_solver_io.py::test_sat_model_can_span_lines_and_returns_raw_gauss_diagnostics
tests/unit/test_backend_solver_io.py::test_cadical_signed_dimacs_model_and_comments
tests/unit/test_backend_solver_io.py::test_empty_unsat_result_has_empty_assignment[cms]
tests/unit/test_backend_solver_io.py::test_empty_unsat_result_has_empty_assignment[cadical]
tests/unit/test_backend_solver_io.py::test_unknown_outcomes[cms-15-s INDETERMINATE\n]
tests/unit/test_backend_solver_io.py::test_unknown_outcomes[cadical-0-s UNKNOWN\n]
tests/unit/test_backend_solver_io.py::test_unknown_needs_real_exit_semantics_and_explicit_output[cms-0-s UNKNOWN\n]
tests/unit/test_backend_solver_io.py::test_unknown_needs_real_exit_semantics_and_explicit_output[cadical-0-]
tests/unit/test_backend_solver_io.py::test_exit_code_status_disagreement_is_error
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-v 1 0\n-required0-1-no explicit solver status]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\ns SATISFIABLE\nv 1 0\n-required1-1-repeated]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 1 0 2\n-required2-2-after terminal zero]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 1 0\nv -1 0\n-required3-1-after terminal zero]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 1 -1 0\n-required4-1-conflicting]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 1 1_0 0\n-required5-10-non-integer]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 4 0\n-required6-3-outside]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 1 -2\n-required7-2-truncated]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE\nv 1 0\n-required8-2-omits required]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[20-s UNSATISFIABLE\nv 0\n-required9-0-UNSAT result unexpectedly]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[0-s UNKNOWN\nv 1 0\n-required10-1-UNKNOWN result unexpectedly]
tests/unit/test_backend_solver_io.py::test_malformed_or_partial_solver_results_fail_closed[10-s SATISFIABLE extra\nv 1 0\n-required11-1-malformed status]
tests/unit/test_backend_solver_io.py::test_invalid_requested_variable_sets_are_rejected[required0-2]
tests/unit/test_backend_solver_io.py::test_invalid_requested_variable_sets_are_rejected[required1-2]
tests/unit/test_backend_solver_io.py::test_invalid_requested_variable_sets_are_rejected[required2-2]
tests/unit/test_backend_solver_io.py::test_invalid_requested_variable_sets_are_rejected[required3-2]
tests/unit/test_backend_solver_io.py::test_invalid_requested_variable_sets_are_rejected[required4--1]
tests/unit/test_backend_solver_io.py::test_unknown_engine_and_unexpected_exit_code_are_errors
tests/unit/test_beam_adjudication.py::test_seed_governance_prohibitions
tests/unit/test_beam_adjudication.py::test_draw_csprng_seed_returns_valid_seed
tests/unit/test_beam_adjudication.py::test_configuration_parameter_invariants
tests/unit/test_beam_adjudication.py::test_counterbalancing_schedule
tests/unit/test_beam_adjudication.py::test_selection_rule_tier1_higher_recovery_wins
tests/unit/test_beam_adjudication.py::test_selection_rule_tier2_lower_par2_wins_on_tie
tests/unit/test_beam_adjudication.py::test_selection_rule_tier4_reference_a_retained_on_complete_tie
tests/unit/test_beam_adjudication.py::test_exact_mcnemar_on_discordant_pairs
tests/unit/test_beam_adjudication.py::test_mismatched_instance_sets_rejected
tests/unit/test_benchmark_formula_export.py::test_benchmark_formula_manifest_integrity
tests/unit/test_benchmark_formula_export.py::test_speck_subcritical_formula_solvable_by_cadical
tests/unit/test_bruteforce_completion.py::test_fixtures_match_task_known_answer_values
tests/unit/test_bruteforce_completion.py::test_official_known_answers_through_c_encryption
tests/unit/test_bruteforce_completion.py::test_official_known_answers_recovered_by_completion
tests/unit/test_bruteforce_completion.py::test_schedule_inversion_round_trip[speck-rounds_range0]
tests/unit/test_bruteforce_completion.py::test_schedule_inversion_round_trip[simon-rounds_range1]
tests/unit/test_bruteforce_completion.py::test_arbitrary_windows_invert_consistently[speck-22]
tests/unit/test_bruteforce_completion.py::test_arbitrary_windows_invert_consistently[simon-32]
tests/unit/test_bruteforce_completion.py::test_full_m3_space_visits_distinct_master_keys[speck-8]
tests/unit/test_bruteforce_completion.py::test_full_m3_space_visits_distinct_master_keys[simon-11]
tests/unit/test_bruteforce_completion.py::test_expand_and_encrypt_agree_with_python_reference[speck-22]
tests/unit/test_bruteforce_completion.py::test_expand_and_encrypt_agree_with_python_reference[simon-32]
tests/unit/test_bruteforce_completion.py::test_speck_expand_agrees_with_gohr_convention
tests/unit/test_bruteforce_completion.py::test_full_space_m3_completion_finds_true_key[speck-8]
tests/unit/test_bruteforce_completion.py::test_full_space_m3_completion_finds_true_key[simon-11]
tests/unit/test_bruteforce_completion.py::test_narrow_ranges_for_partial_trailing_keys[speck-8-2]
tests/unit/test_bruteforce_completion.py::test_narrow_ranges_for_partial_trailing_keys[speck-8-1]
tests/unit/test_bruteforce_completion.py::test_narrow_ranges_for_partial_trailing_keys[simon-11-2]
tests/unit/test_bruteforce_completion.py::test_narrow_ranges_for_partial_trailing_keys[simon-11-1]
tests/unit/test_bruteforce_completion.py::test_wrong_trailing_key_is_rejected_over_full_space[speck-8]
tests/unit/test_bruteforce_completion.py::test_wrong_trailing_key_is_rejected_over_full_space[simon-11]
tests/unit/test_bruteforce_completion.py::test_every_block_is_verified_not_only_the_first
tests/unit/test_bruteforce_completion.py::test_invalid_arguments_are_rejected
tests/unit/test_bruteforce_completion.py::test_build_uses_frozen_flags_and_produces_library
tests/unit/test_bruteforce_completion.py::test_quick_benchmark_cli_writes_complete_record
tests/unit/test_cipher.py::test_one_round_encryption_matches_regression_state
tests/unit/test_cipher.py::test_one_round_decryption_is_the_inverse
tests/unit/test_cipher.py::test_full_known_answer_vector
tests/unit/test_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[1-expected0]
tests/unit/test_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[2-expected1]
tests/unit/test_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[7-expected2]
tests/unit/test_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[11-expected3]
tests/unit/test_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[22-expected4]
tests/unit/test_cipher.py::test_deterministic_random_round_trips
tests/unit/test_cipher.py::test_round_and_block_inputs_are_explicitly_masked
tests/unit/test_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block-0]
tests/unit/test_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block--1]
tests/unit/test_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block-23]
tests/unit/test_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block-0]
tests/unit/test_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block--1]
tests/unit/test_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block-23]
tests/unit/test_cipher.py::test_non_integer_round_counts_are_rejected[encrypt_block-1.0]
tests/unit/test_cipher.py::test_non_integer_round_counts_are_rejected[encrypt_block-True]
tests/unit/test_cipher.py::test_non_integer_round_counts_are_rejected[decrypt_block-1.0]
tests/unit/test_cipher.py::test_non_integer_round_counts_are_rejected[decrypt_block-True]
tests/unit/test_cipher.py::test_malformed_round_state_is_rejected
tests/unit/test_compute_mcnemar_power.py::test_exact_binomial_twoside_pvalue_symmetry
tests/unit/test_compute_mcnemar_power.py::test_compute_exact_mcnemar_power_null_hypothesis
tests/unit/test_compute_mcnemar_power.py::test_compute_exact_mcnemar_power_monotonicity
tests/unit/test_compute_mcnemar_power.py::test_compute_exact_mcnemar_power_invalid_inputs
tests/unit/test_compute_mcnemar_power.py::test_generate_sensitivity_grid
tests/unit/test_compute_mcnemar_power.py::test_parse_args
tests/unit/test_conditioning_final_audit.py::test_original_formula_check_rejects_parity_and_conditioning_errors
tests/unit/test_conditioning_final_audit.py::test_duplicate_json_fields_are_rejected
tests/unit/test_conditioning_final_audit.py::test_checksum_and_sidecar_filename_are_both_required
tests/unit/test_conditioning_final_audit.py::test_summary_keeps_failures_and_recomputes_paired_denominators
tests/unit/test_conditioning_worker.py::test_production_solver_factory_uses_the_fixed_wrapper_options
tests/unit/test_conditioning_worker.py::test_worker_loads_identical_formula_and_changes_only_conditioning_interface[permanent_units-speck32_64]
tests/unit/test_conditioning_worker.py::test_worker_loads_identical_formula_and_changes_only_conditioning_interface[permanent_units-simon32_64]
tests/unit/test_conditioning_worker.py::test_worker_loads_identical_formula_and_changes_only_conditioning_interface[temporary_assumptions-speck32_64]
tests/unit/test_conditioning_worker.py::test_worker_loads_identical_formula_and_changes_only_conditioning_interface[temporary_assumptions-simon32_64]
tests/unit/test_conditioning_worker.py::test_temporary_assumptions_are_removed_and_permanent_units_persist
tests/unit/test_conditioning_worker.py::test_reference_known_check_uses_cipher_and_key_schedule[speck32_64]
tests/unit/test_conditioning_worker.py::test_reference_known_check_uses_cipher_and_key_schedule[simon32_64]
tests/unit/test_conditioning_worker.py::test_invalid_model_is_correctness_alarm_and_truth_is_not_read
tests/unit/test_conditioning_worker.py::test_known_check_failure_stops_before_truth_load
tests/unit/test_conditioning_worker.py::test_valid_sat_model_without_fresh_match_is_retained_as_nonunique_outcome
tests/unit/test_conditioning_worker.py::test_truth_hash_is_checked_only_after_known_model_check
tests/unit/test_conditioning_worker.py::test_true_key_unsat_is_a_distinct_correctness_alarm
tests/unit/test_conditioning_worker.py::test_environment_rejection_keeps_initial_telemetry_and_power_sample
tests/unit/test_conditioning_worker.py::test_thread_environment_mismatch_stops_before_solver_certification
tests/unit/test_conditioning_worker.py::test_invalid_fixed_cell_is_rejected_before_environment_or_solver[rounds-True]
tests/unit/test_conditioning_worker.py::test_invalid_fixed_cell_is_rejected_before_environment_or_solver[key_index-8]
tests/unit/test_conditioning_worker.py::test_invalid_fixed_cell_is_rejected_before_environment_or_solver[observations-2]
tests/unit/test_conditioning_worker.py::test_invalid_fixed_cell_is_rejected_before_environment_or_solver[trailing_keys-0]
tests/unit/test_conditioning_worker.py::test_invalid_fixed_cell_is_rejected_before_environment_or_solver[arm-unknown]
tests/unit/test_conditioning_worker.py::test_main_reads_checked_payload_and_atomically_persists_result
tests/unit/test_confidence.py::test_features_use_ordered_scores_and_have_expected_values
tests/unit/test_confidence.py::test_serialized_features_and_target_membership_are_deterministic
tests/unit/test_confidence.py::test_pair_subset_stability_is_exposed_when_present
tests/unit/test_confidence.py::test_tied_scores_keep_the_supplied_deterministic_order
tests/unit/test_confidence.py::test_malformed_rankings_are_rejected[candidate_values0-candidate_scores0-candidate_order0]
tests/unit/test_confidence.py::test_malformed_rankings_are_rejected[candidate_values1-candidate_scores1-candidate_order1]
tests/unit/test_confidence.py::test_malformed_rankings_are_rejected[candidate_values2-candidate_scores2-candidate_order2]
tests/unit/test_confidence.py::test_malformed_rankings_are_rejected[candidate_values3-candidate_scores3-candidate_order3]
tests/unit/test_confidence.py::test_malformed_rankings_are_rejected[candidate_values4-candidate_scores4-candidate_order4]
tests/unit/test_confidence.py::test_invalid_top_k_is_rejected[0]
tests/unit/test_confidence.py::test_invalid_top_k_is_rejected[1]
tests/unit/test_confidence.py::test_invalid_top_k_is_rejected[5]
tests/unit/test_confidence.py::test_invalid_top_k_is_rejected[True]
tests/unit/test_confidence.py::test_invalid_top_k_is_rejected[2.0]
tests/unit/test_confidence.py::test_risk_coverage_is_validation_only_and_deterministic
tests/unit/test_confidence.py::test_extended_risk_coverage_requires_stability_data
tests/unit/test_confidence.py::test_risk_coverage_rejects_unknown_or_empty_inputs
tests/unit/test_data_generator.py::test_word_encoding_is_most_significant_bit_first
tests/unit/test_data_generator.py::test_generation_is_identical_for_a_fixed_configuration_and_seed
tests/unit/test_data_generator.py::test_different_seeds_produce_different_samples
tests/unit/test_data_generator.py::test_generated_shapes_labels_and_debug_pairs_are_consistent
tests/unit/test_data_generator.py::test_positive_debug_pairs_use_the_selected_difference
tests/unit/test_data_generator.py::test_split_configs_require_distinct_seeds_and_shared_protocol
tests/unit/test_data_generator.py::test_invalid_dataset_configuration_is_rejected[0-7-1001-train]
tests/unit/test_data_generator.py::test_invalid_dataset_configuration_is_rejected[3-7-1001-train]
tests/unit/test_data_generator.py::test_invalid_dataset_configuration_is_rejected[8-0-1001-train]
tests/unit/test_data_generator.py::test_invalid_dataset_configuration_is_rejected[8-23-1001-train]
tests/unit/test_data_generator.py::test_invalid_dataset_configuration_is_rejected[8-7-1001-holdout]
tests/unit/test_dataset_export.py::test_classifier_export_is_identical_for_a_fixed_seed
tests/unit/test_dataset_export.py::test_different_seeds_change_exported_classifier_data
tests/unit/test_dataset_export.py::test_exported_classifier_metadata_and_balance_are_complete
tests/unit/test_dataset_export.py::test_classifier_benchmark_verifies_all_shards_against_manifest
tests/unit/test_dataset_export.py::test_classifier_loader_rejects_nonbinary_content
tests/unit/test_dataset_export.py::test_hex_word_rejects_noncanonical_text[ 123]
tests/unit/test_dataset_export.py::test_hex_word_rejects_noncanonical_text[+123]
tests/unit/test_dataset_export.py::test_hex_word_rejects_noncanonical_text[0x12]
tests/unit/test_dataset_export.py::test_hex_word_rejects_noncanonical_text[12-3]
tests/unit/test_dataset_export.py::test_exported_classifier_loads_without_the_generator
tests/unit/test_dataset_export.py::test_classifier_checksums_detect_a_tampered_shard
tests/unit/test_dataset_export.py::test_loaded_classifier_shard_matches_in_memory_generation
tests/unit/test_dataset_export.py::test_split_validation_still_requires_one_role_each
tests/unit/test_dataset_v1_freeze.py::test_classifier_freeze_matches_code_and_fixture
tests/unit/test_dataset_v1_freeze.py::test_attack_freeze_uses_plaintext_difference
tests/unit/test_dataset_v1_freeze.py::test_classifier_and_attack_remain_separate_frozen_datasets
tests/unit/test_demo_depth_dataset.py::test_dataset_sidecar_checksum_valid
tests/unit/test_demo_depth_dataset.py::test_cipher_known_answer_tests
tests/unit/test_demo_depth_dataset.py::test_dataset_structure_and_invariants
tests/unit/test_depth_analysis.py::test_timeouts_are_unavailable_and_keep_failure_penalty
tests/unit/test_depth_analysis.py::test_analysis_rejects_grid_or_metric_inconsistency[missing]
tests/unit/test_depth_analysis.py::test_analysis_rejects_grid_or_metric_inconsistency[identity]
tests/unit/test_depth_analysis.py::test_analysis_rejects_grid_or_metric_inconsistency[flag]
tests/unit/test_depth_analysis.py::test_analysis_rejects_grid_or_metric_inconsistency[counts]
tests/unit/test_depth_analysis.py::test_analysis_rejects_grid_or_metric_inconsistency[late]
tests/unit/test_depth_analysis.py::test_hash_corruption_cannot_be_analyzed
tests/unit/test_depth_map_runner.py::test_manifest_validation_passes
tests/unit/test_depth_map_runner.py::test_manifest_validation_rejects_tampering
tests/unit/test_depth_map_runner.py::test_schedule_properties
tests/unit/test_depth_map_runner.py::test_parent_watchdog_terminates_hanging_worker
tests/unit/test_depth_map_runner.py::test_unsat_triggers_correctness_alarm
tests/unit/test_depth_map_runner.py::test_atomic_journal_and_interrupted_cell
tests/unit/test_depth_map_runner.py::test_single_cell_real_solve
tests/unit/test_depth_map_runner.py::test_simon_single_cell_real_solve
tests/unit/test_depth_map_runner.py::test_late_recovery_rejected
tests/unit/test_dimacs.py::test_dimacs_export_and_solve_recovers_master_key
tests/unit/test_dimacs.py::test_dimacs_export_with_pinned_round_key
tests/unit/test_encryption_calibration.py::test_native_kats_all_prefixes_and_public_fixture
tests/unit/test_encryption_calibration.py::test_effective_cpu_policy_is_checked_without_timing
tests/unit/test_encryption_calibration.py::test_observer_deadline_is_passed_to_subprocess
tests/unit/test_encryption_calibration.py::test_pinned_source_is_the_actual_imported_module
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[0 7 1 2\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[0 7 1 2 3 4 5 6 7\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[2 7 1 2 3 4 5 6\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[0 23 1 2 3 4 5 6\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[1 33 1 2 3 4 5 6\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[0 7 -1 2 3 4 5 6\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_malformed_verification[0 7 184467440737095516161 2 3 4 5 6\n]
tests/unit/test_encryption_calibration.py::test_native_rejects_invalid_arguments_before_work[args0]
tests/unit/test_encryption_calibration.py::test_native_rejects_invalid_arguments_before_work[args1]
tests/unit/test_encryption_calibration.py::test_native_rejects_invalid_arguments_before_work[args2]
tests/unit/test_encryption_calibration.py::test_native_rejects_invalid_arguments_before_work[args3]
tests/unit/test_encryption_calibration.py::test_native_rejects_invalid_arguments_before_work[args4]
tests/unit/test_encryption_calibration.py::test_native_rejects_invalid_arguments_before_work[args5]
tests/unit/test_encryption_calibration.py::test_full_fixture_has_distinct_keys_and_blocks_and_known_digest
tests/unit/test_encryption_calibration.py::test_measured_conversion_uses_all_repetitions_and_one_profile_rate
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[elapsed_seconds-nan]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[elapsed_seconds--1]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[elapsed_seconds-1.99]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[batches-True]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[blocks-1]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[checksum-wrong]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[qos_class-9]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[nice-5]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[threads-8]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[peak_rss_bytes-0]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[peak_rss_bytes-100000000]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[identity_sha256-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb]
tests/unit/test_encryption_calibration.py::test_sample_tampering_rejected[ordinal-3]
tests/unit/test_encryption_calibration.py::test_missing_duplicate_and_wrong_order_fail_closed
tests/unit/test_encryption_calibration.py::test_frozen_profile_drift_rejected[threads-True]
tests/unit/test_encryption_calibration.py::test_frozen_profile_drift_rejected[repetitions-5]
tests/unit/test_encryption_calibration.py::test_frozen_profile_drift_rejected[minimum_measurement_seconds-1]
tests/unit/test_encryption_calibration.py::test_frozen_profile_drift_rejected[compiler_flags-value3]
tests/unit/test_encryption_calibration.py::test_frozen_profile_drift_rejected[protected_data_access-True]
tests/unit/test_evaluation_metrics.py::test_calculate_global_par2_basic
tests/unit/test_evaluation_metrics.py::test_calculate_global_par2_solved_exceeds_cutoff
tests/unit/test_evaluation_metrics.py::test_calculate_global_par2_solved_exact_cutoff
tests/unit/test_evaluation_metrics.py::test_calculate_global_par2_empty_and_errors
tests/unit/test_evaluation_metrics.py::test_wilson_score_interval_exact
tests/unit/test_evaluation_metrics.py::test_wilson_score_interval_validation
tests/unit/test_evaluation_metrics.py::test_exact_mcnemar_test_basic
tests/unit/test_evaluation_metrics.py::test_calculate_cactus_curve_monotonic
tests/unit/test_fallback_redesign.py::test_invariant_1_g1_feature_extraction_is_deterministic
tests/unit/test_fallback_redesign.py::test_invariant_2_g1_handles_degenerate_distributions_gracefully
tests/unit/test_fallback_redesign.py::test_invariant_3_normalized_logit_margin_matches_cattelan_silva
tests/unit/test_fallback_redesign.py::test_invariant_4_r1_hard_negative_miner_labels_correctly
tests/unit/test_fallback_redesign.py::test_invariant_5_r1_shortlist_builder_preserves_true_subkey
tests/unit/test_fallback_redesign.py::test_invariant_6_rescue_ceiling_bounded_by_top_m_recall
tests/unit/test_fallback_redesign.py::test_invariant_7_cascade_routes_to_b1_when_reliability_high
tests/unit/test_fallback_redesign.py::test_invariant_8_cascade_routes_to_r1_when_reliability_low
tests/unit/test_fallback_redesign.py::test_invariant_9_lazy_b0_never_computed_unless_primary_fails
tests/unit/test_fallback_redesign.py::test_invariant_10_solver_isolation_maintained_in_cascade
tests/unit/test_fallback_redesign.py::test_invariant_11_state_machine_tracks_cumulative_time
tests/unit/test_fallback_redesign.py::test_invariant_12_seed_validation_rejects_quarantined_seeds
tests/unit/test_fallback_redesign.py::test_invariant_13_locked_validation_role_cannot_enter_training_or_calibration
tests/unit/test_fallback_redesign.py::test_invariant_14_g1_optimized_reference_equivalence
tests/unit/test_fallback_redesign.py::test_invariant_15_r1_shortlist_budget_selection
tests/unit/test_fallback_redesign.py::test_invariant_16_terminal_fallback_deduplication_and_cumulative_budget
tests/unit/test_fallback_redesign.py::test_invariant_17_g1_failure_risk_routing_polarity
tests/unit/test_final_publication_figures.py::test_source_hash_mismatch_fails_closed
tests/unit/test_final_publication_figures.py::test_attack_duplicate_or_missing_pair_rejected[speck32_64]
tests/unit/test_final_publication_figures.py::test_attack_duplicate_or_missing_pair_rejected[simon32_64]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[verified_on_fresh_pairs-false]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[verified_on_fresh_pairs-False]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[complete_online_wall_seconds-0]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[complete_online_wall_seconds-nan]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[complete_online_wall_seconds-301]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[instance_id-another_key]
tests/unit/test_final_publication_figures.py::test_attack_contradictions_rejected[outcome-fabricated]
tests/unit/test_final_publication_figures.py::test_speck_boolean_and_late_flags_rejected[counts_as_verified_recovery-false]
tests/unit/test_final_publication_figures.py::test_speck_boolean_and_late_flags_rejected[success-False]
tests/unit/test_final_publication_figures.py::test_speck_boolean_and_late_flags_rejected[late_verified_recovery-True]
tests/unit/test_final_publication_figures.py::test_paired_intervals_match_literature_fixtures[counts0-expected0]
tests/unit/test_final_publication_figures.py::test_paired_intervals_match_literature_fixtures[counts1-expected1]
tests/unit/test_final_publication_figures.py::test_failure_matrix_rejects_wrong_total_and_duplicate
tests/unit/test_final_publication_figures.py::test_actual_table_totals_provenance_and_replay
tests/unit/test_final_publication_figures.py::test_recorded_no_entry_is_zero_but_depth_timeout_is_unavailable
tests/unit/test_final_publication_figures.py::test_corrected_paired_ci_and_frozen_holm_families
tests/unit/test_final_publication_figures.py::test_backend_interruption_excluded_only_from_affected_pairs
tests/unit/test_final_publication_figures.py::test_diagnostic_grids_reject_duplicates_and_bad_flags[depth-depth_raw]
tests/unit/test_final_publication_figures.py::test_diagnostic_grids_reject_duplicates_and_bad_flags[backend-backend_raw]
tests/unit/test_final_publication_figures.py::test_main_figures_only_two_panels_and_paper_names
tests/unit/test_g2_pipeline.py::test_partition_masks_properties
tests/unit/test_g2_pipeline.py::test_distributional_features_shift_invariance
tests/unit/test_g2_pipeline.py::test_stability_plus_features
tests/unit/test_g2_pipeline.py::test_g2_feature_vector_extraction
tests/unit/test_g2_pipeline.py::test_g2_models_train_and_predict
tests/unit/test_g2_pipeline.py::test_distributional_exact_arithmetic
tests/unit/test_g2_pipeline.py::test_stability_plus_exact_arithmetic
tests/unit/test_hierarchical_multi_round.py::test_hierarchical_stage_rejects_invalid_arguments
tests/unit/test_hierarchical_multi_round.py::test_attack_rejects_invalid_arguments
tests/unit/test_hierarchical_multi_round.py::test_single_stage_attack_recovers_five_round_instance_key
tests/unit/test_hierarchical_multi_round.py::test_two_stage_attack_refutes_a_wrong_pair_before_verifying
tests/unit/test_hierarchical_multi_round.py::test_identical_runs_produce_identical_search_and_order
tests/unit/test_hierarchical_multi_round.py::test_beam_width_bounds_the_number_of_tested_tuples
tests/unit/test_hierarchical_multi_round.py::test_pruned_first_stage_candidate_reports_ranking_failure
tests/unit/test_hierarchical_multi_round.py::test_pruning_inside_the_final_stage_reports_ranking_failure
tests/unit/test_hierarchical_multi_round.py::test_search_does_not_read_declared_ground_truth
tests/unit/test_hierarchical_multi_round.py::test_frozen_b1_checkpoint_satisfies_the_stage_interface
tests/unit/test_hierarchical_multi_round.py::test_screened_hierarchical_stage_validation_and_execution
tests/unit/test_hierarchical_multi_round.py::test_stage_cache_and_excluded_paths_cascade
tests/unit/test_hierarchical_ranking.py::test_h1_preserves_b1_window_set_and_tail
tests/unit/test_hierarchical_ranking.py::test_h1_b0_ties_use_ascending_candidate_value
tests/unit/test_hierarchical_ranking.py::test_h1_invokes_b0_only_for_the_sixteen_candidate_window
tests/unit/test_hierarchical_ranking.py::test_h1_uses_existing_b0_sum_logits_sorting
tests/unit/test_hierarchical_ranking.py::test_h1_scalar_and_batched_window_scores_match
tests/unit/test_hierarchical_ranking.py::test_h1_rerun_is_deterministic
tests/unit/test_hierarchical_ranking.py::test_h1_serialization_round_trip
tests/unit/test_hierarchical_ranking.py::test_h1_deserialization_rejects_order_inconsistent_with_b0_scores
tests/unit/test_hierarchical_ranking.py::test_h1_deserialization_rejects_corrupt_protocol_metadata[tie_rule]
tests/unit/test_hierarchical_ranking.py::test_h1_deserialization_rejects_corrupt_protocol_metadata[candidate_domain]
tests/unit/test_hierarchical_ranking.py::test_h1_timing_fields_are_finite_and_non_negative
tests/unit/test_hierarchical_ranking.py::test_h1_ordering_api_does_not_accept_or_serialize_true_key
tests/unit/test_hierarchical_ranking.py::test_h1_rejects_incomplete_or_duplicate_b1_order
tests/unit/test_hierarchical_ranking.py::test_h1_rejects_unfrozen_budget_and_invalid_batch_size[kwargs0-window_size]
tests/unit/test_hierarchical_ranking.py::test_h1_rejects_unfrozen_budget_and_invalid_batch_size[kwargs1-b0_candidate_batch_size]
tests/unit/test_hierarchical_ranking.py::test_h1_rejects_wrong_instance_pair_count
tests/unit/test_hierarchical_ranking.py::test_h1_rejects_wrong_ranking_type
tests/unit/test_hierarchical_ranking.py::test_h1_rejects_non_model_b0_input
tests/unit/test_key_schedule.py::test_full_key_schedule_matches_regression_values
tests/unit/test_key_schedule.py::test_reduced_key_schedule_is_a_prefix[1]
tests/unit/test_key_schedule.py::test_reduced_key_schedule_is_a_prefix[2]
tests/unit/test_key_schedule.py::test_reduced_key_schedule_is_a_prefix[7]
tests/unit/test_key_schedule.py::test_reduced_key_schedule_is_a_prefix[11]
tests/unit/test_key_schedule.py::test_reduced_key_schedule_is_a_prefix[22]
tests/unit/test_key_schedule.py::test_key_schedule_is_deterministic
tests/unit/test_key_schedule.py::test_key_schedule_outputs_16_bit_words
tests/unit/test_key_schedule.py::test_key_schedule_constants_match_verified_parameters
tests/unit/test_key_schedule.py::test_master_key_words_are_explicitly_masked
tests/unit/test_key_schedule.py::test_invalid_round_counts_are_rejected[0]
tests/unit/test_key_schedule.py::test_invalid_round_counts_are_rejected[-1]
tests/unit/test_key_schedule.py::test_invalid_round_counts_are_rejected[23]
tests/unit/test_key_schedule.py::test_non_integer_round_count_is_rejected
tests/unit/test_key_schedule.py::test_boolean_round_count_is_rejected
tests/unit/test_key_schedule.py::test_wrong_master_key_length_is_rejected[master_key0]
tests/unit/test_key_schedule.py::test_wrong_master_key_length_is_rejected[master_key1]
tests/unit/test_key_schedule.py::test_wrong_master_key_length_is_rejected[master_key2]
tests/unit/test_key_schedule.py::test_non_integer_master_key_word_is_rejected
tests/unit/test_multi_round_recovery.py::test_assumptions_for_round_key_pins_every_bit
tests/unit/test_multi_round_recovery.py::test_assumptions_reject_invalid_inputs
tests/unit/test_multi_round_recovery.py::test_assumptions_for_round_keys_are_order_independent
tests/unit/test_multi_round_recovery.py::test_trailing_round_key_indices
tests/unit/test_multi_round_recovery.py::test_multi_round_recovery_verifies_true_candidate
tests/unit/test_multi_round_recovery.py::test_multi_round_recovery_skips_wrong_candidate
tests/unit/test_multi_round_recovery.py::test_multi_round_recovery_reports_exhausted_candidates
tests/unit/test_multi_round_recovery.py::test_partial_decrypt_instance_matches_shorter_encryption
tests/unit/test_multi_round_recovery.py::test_partial_decrypt_instance_rejects_invalid_inputs
tests/unit/test_multi_round_recovery.py::test_multi_round_recovery_requires_uniform_candidate_width
tests/unit/test_multi_round_recovery.py::test_seven_round_recovery_needs_two_assumed_round_keys
tests/unit/test_neural_dataset.py::test_features_to_tensor_reshapes_64_bits_to_4_words
tests/unit/test_neural_dataset.py::test_dataset_from_fixture_json
tests/unit/test_neural_dataset.py::test_features_to_tensor_rejects_nonbinary_values
tests/unit/test_neural_dataset.py::test_classifier_splits_load_and_verify_fixture
tests/unit/test_neural_dataset.py::test_dataset_from_generated_in_memory
tests/unit/test_neural_model.py::test_gohr_resnet_forward_shapes
tests/unit/test_neural_model.py::test_gohr_resnet_probabilities_in_range
tests/unit/test_neural_model.py::test_predict_proba_supports_one_sample_without_changing_training_mode
tests/unit/test_neural_model.py::test_single_sample_training_batch_is_supported
tests/unit/test_neural_model.py::test_invalid_resnet_configuration_is_rejected
tests/unit/test_neural_model.py::test_batched_vs_unbatched_inference_equivalence
tests/unit/test_neural_model.py::test_gradient_flows_through_all_parameters
tests/unit/test_neural_train.py::test_roc_auc_metric_perfect_and_random
tests/unit/test_neural_train.py::test_metrics_reject_nonbinary_labels_and_invalid_probabilities
tests/unit/test_neural_train.py::test_train_distinguisher_runs_and_improves_loss
tests/unit/test_neural_train.py::test_seeded_model_training_is_reproducible
tests/unit/test_neural_train.py::test_save_and_load_checkpoint
tests/unit/test_numpy_inference.py::test_numpy_pair_group_outputs_match_pytorch[b1_mean]
tests/unit/test_numpy_inference.py::test_numpy_pair_group_outputs_match_pytorch[b2_moments]
tests/unit/test_numpy_inference.py::test_numpy_pair_group_outputs_match_pytorch[b3_weighted]
tests/unit/test_numpy_inference.py::test_numpy_b0_outputs_match_pytorch
tests/unit/test_numpy_inference.py::test_numpy_adapter_returns_torch_logits
tests/unit/test_numpy_inference.py::test_numpy_backend_rejects_wrong_dtype
tests/unit/test_numpy_inference.py::test_numpy_backend_rejects_wrong_group_shape
tests/unit/test_numpy_inference.py::test_vectorized_final_round_features_match_scalar_reference
tests/unit/test_pair_group_data.py::test_pair_group_generation_is_deterministic_and_typed
tests/unit/test_pair_group_data.py::test_different_pair_group_seed_changes_content
tests/unit/test_pair_group_data.py::test_group_keys_and_positive_negative_construction_are_correct
tests/unit/test_pair_group_data.py::test_group_split_keys_are_disjoint
tests/unit/test_pair_group_data.py::test_pair_group_export_load_and_manifest_verification
tests/unit/test_pair_group_data.py::test_phase6a_attack_export_is_validation_only
tests/unit/test_pair_group_model.py::test_pair_group_forward_shapes_and_permutation_invariance[b1_mean]
tests/unit/test_pair_group_model.py::test_pair_group_forward_shapes_and_permutation_invariance[b2_moments]
tests/unit/test_pair_group_model.py::test_pair_group_forward_shapes_and_permutation_invariance[b3_weighted]
tests/unit/test_pair_group_model.py::test_pair_group_initialization_is_deterministic_and_transfers_b0_encoder
tests/unit/test_pair_group_model.py::test_pair_group_rejects_wrong_dtype_and_shape
tests/unit/test_pair_group_model.py::test_pair_group_gradients_are_finite
tests/unit/test_pair_group_model.py::test_pair_group_checkpoint_round_trip
tests/unit/test_pair_group_model.py::test_pair_group_training_records_finite_validation_bce
tests/unit/test_pair_group_ranking.py::test_pair_group_candidate_batching_and_mode_restore
tests/unit/test_pair_group_ranking.py::test_pair_group_ties_use_ascending_candidate_value
tests/unit/test_pair_group_ranking.py::test_numpy_pair_group_ranking_matches_reference_on_small_domain
tests/unit/test_paired_stats.py::TestContingency2x2::test_compute_2x2_contingency_basic
tests/unit/test_paired_stats.py::TestContingency2x2::test_compute_2x2_contingency_mismatch
tests/unit/test_paired_stats.py::TestExactMcNemar::test_exact_binomial_twoside_pvalue
tests/unit/test_paired_stats.py::TestExactMcNemar::test_exact_mcnemar_literature_values
tests/unit/test_paired_stats.py::TestExactMcNemar::test_exact_mcnemar_errors
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_altman_table_6_2_literature_fixture
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_altman_table_6_2_reversed_symmetry
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_newcombe_table_3_literature_fixture
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_zero_discordance_ties
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_different_concordant_splits
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_negative_association
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_boundary_proportions_full_success
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_boundary_proportions_both_zero
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_boundary_proportions_both_one
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_complete_separation
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_small_sample_sizes_n1
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_reversal_symmetry_general
tests/unit/test_paired_stats.py::TestNewcombeMethod10PairedCI::test_validation_errors
tests/unit/test_paired_stats.py::TestHolmBonferroniAdjustment::test_two_primary_contrasts_both_rejected
tests/unit/test_paired_stats.py::TestHolmBonferroniAdjustment::test_two_primary_contrasts_neither_rejected
tests/unit/test_paired_stats.py::TestHolmBonferroniAdjustment::test_two_primary_contrasts_one_rejected
tests/unit/test_paired_stats.py::TestHolmBonferroniAdjustment::test_order_invariance
tests/unit/test_paired_stats.py::TestHolmBonferroniAdjustment::test_boundary_cases
tests/unit/test_paired_stats.py::TestMcNemarPowerSensitivity::test_protocol_power_fixtures
tests/unit/test_paired_stats.py::TestMcNemarPowerSensitivity::test_power_edge_cases
tests/unit/test_partial.py::test_true_final_round_key_recovers_the_pre_final_state
tests/unit/test_partial.py::test_wrong_final_round_key_does_not_recover_the_pre_final_state
tests/unit/test_partial.py::test_partial_decryption_matches_inverse_round_for_seeded_states
tests/unit/test_phase10_vtc.py::test_solver_isolation_invariant
tests/unit/test_phase10_vtc.py::test_vtc_early_termination_on_b1_success
tests/unit/test_phase10_vtc.py::test_vtc_escalation_on_b1_exhaustion
tests/unit/test_phase10_vtc.py::test_vtc_candidate_deduplication
tests/unit/test_phase10_vtc.py::test_vtc_budget_capping
tests/unit/test_phase10_vtc.py::test_vtc_serialization_roundtrip
tests/unit/test_phase11b_prime_pair_group.py::test_six_round_generation_is_deterministic
tests/unit/test_phase11b_prime_pair_group.py::test_six_round_metadata_records_six_rounds_and_owned_dataset_id
tests/unit/test_phase11b_prime_pair_group.py::test_six_round_features_use_six_round_encryption
tests/unit/test_phase11b_prime_pair_group.py::test_six_round_config_rejects_five_rounds
tests/unit/test_phase11b_prime_pair_group.py::test_frozen_five_round_config_still_rejects_six_rounds
tests/unit/test_phase11b_prime_pair_group.py::test_frozen_five_round_identifiers_are_unchanged
tests/unit/test_phase11b_prime_pair_group.py::test_six_round_export_manifest_and_shard_round_trip
tests/unit/test_phase11b_prime_pair_group.py::test_five_round_path_rejects_six_round_artefacts
tests/unit/test_phase11b_prime_pair_group.py::test_five_round_export_cannot_emit_six_round_configurations
tests/unit/test_phase11b_prime_pair_group.py::test_stage1_seed_role_accepts_its_own_seeds
tests/unit/test_phase11b_prime_pair_group.py::test_stage1_seed_role_rejects_history_and_other_protected_seeds
tests/unit/test_phase11b_prime_pair_group.py::test_protected_stage1_seeds_are_rejected_by_other_roles
tests/unit/test_phase6b_proposal_figures.py::test_phase6b_summary_is_validation_only_and_complete
tests/unit/test_phase6b_proposal_figures.py::test_phase6b_result_figure_uses_selected_b1_metrics
tests/unit/test_phase6b_proposal_figures.py::test_phase6b_methodology_marks_planned_components
tests/unit/test_phase6b_proposal_figures.py::test_all_eight_proposal_figures_generate_valid_svg
tests/unit/test_phase6b_proposal_figures.py::test_proposal_figures_generation_in_temp_dir
tests/unit/test_phase6b_proposal_figures.py::test_phase6b_figure_manifest_is_reproducible
tests/unit/test_phase6c_calibration.py::test_phase6c_feature_uses_ordered_top8_and_all_score_std
tests/unit/test_phase6c_calibration.py::test_phase6c_feature_has_zero_score_std_guard
tests/unit/test_phase6c_calibration.py::test_platt_fit_is_deterministic_bounded_and_serializable
tests/unit/test_phase6c_calibration.py::test_platt_fit_rejects_nonconvergence_and_single_class
tests/unit/test_phase6c_calibration.py::test_threshold_selection_is_unique_deterministic_and_fails_closed
tests/unit/test_phase6c_calibration.py::test_ece_boundaries_and_auc_tie_credit_are_explicit
tests/unit/test_phase6c_calibration.py::test_calibration_report_contains_prevalence_brier_and_risk_coverage
tests/unit/test_phase6c_calibration.py::test_bootstrap_mean_interval_is_seed_deterministic
tests/unit/test_phase6c_fallback.py::test_fallback_alternates_and_fills_exactly_32_unique_candidates
tests/unit/test_phase6c_fallback.py::test_fallback_first_occurrence_and_duplicate_fill_are_deterministic
tests/unit/test_phase6c_fallback.py::test_route_boundary_uses_trusted_route_at_threshold
tests/unit/test_phase6c_fallback.py::test_fallback_rejects_non_complete_inputs[b10-b00]
tests/unit/test_phase6c_fallback.py::test_fallback_rejects_non_complete_inputs[b11-b01]
tests/unit/test_phase6c_fallback.py::test_fallback_rejects_non_complete_inputs[b12-b02]
tests/unit/test_phase7_evaluation_figures.py::test_phase7_summary_is_complete_and_has_all_five_methods
tests/unit/test_phase7_evaluation_figures.py::test_phase7_figures_render_valid_svg_strings
tests/unit/test_phase7_evaluation_figures.py::test_phase7_latex_and_markdown_tables_are_valid
tests/unit/test_phase7_protocol.py::test_phase7_config_is_held_out_test_and_seed_disjoint
tests/unit/test_phase7_protocol.py::test_phase7_methods_include_all_five_configurations
tests/unit/test_phase7_protocol.py::test_phase7_counterbalanced_order_is_cyclic_permutation
tests/unit/test_phase7_protocol.py::test_phase7_atomic_write_and_summary_aggregation
tests/unit/test_primitives.py::test_mask_word_wraps_to_16_bits[0-0]
tests/unit/test_primitives.py::test_mask_word_wraps_to_16_bits[1-1]
tests/unit/test_primitives.py::test_mask_word_wraps_to_16_bits[65535-65535]
tests/unit/test_primitives.py::test_mask_word_wraps_to_16_bits[65536-0]
tests/unit/test_primitives.py::test_mask_word_wraps_to_16_bits[-1-65535]
tests/unit/test_primitives.py::test_rotations_use_16_bit_words[4660-0-4660-4660]
tests/unit/test_primitives.py::test_rotations_use_16_bit_words[4660-4-9025-16675]
tests/unit/test_primitives.py::test_rotations_use_16_bit_words[4660-16-4660-4660]
tests/unit/test_primitives.py::test_rotations_use_16_bit_words[65535-7-65535-65535]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-0]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-1]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-2]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-7]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-15]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-16]
tests/unit/test_primitives.py::test_rotations_are_inverses[0-31]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-0]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-1]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-2]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-7]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-15]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-16]
tests/unit/test_primitives.py::test_rotations_are_inverses[1-31]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-0]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-1]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-2]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-7]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-15]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-16]
tests/unit/test_primitives.py::test_rotations_are_inverses[4660-31]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-0]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-1]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-2]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-7]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-15]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-16]
tests/unit/test_primitives.py::test_rotations_are_inverses[65535-31]
tests/unit/test_primitives.py::test_rotations_normalise_negative_counts
tests/unit/test_primitives.py::test_add_words_wraps
tests/unit/test_primitives.py::test_subtract_words_wraps
tests/unit/test_primitives.py::test_addition_and_subtraction_are_inverses[0-0]
tests/unit/test_primitives.py::test_addition_and_subtraction_are_inverses[1-2]
tests/unit/test_primitives.py::test_addition_and_subtraction_are_inverses[4660-65244]
tests/unit/test_primitives.py::test_addition_and_subtraction_are_inverses[65535-65535]
tests/unit/test_primitives.py::test_non_integer_word_inputs_are_rejected[mask_word]
tests/unit/test_primitives.py::test_non_integer_word_inputs_are_rejected[rotate_left]
tests/unit/test_primitives.py::test_non_integer_word_inputs_are_rejected[rotate_right]
tests/unit/test_primitives.py::test_boolean_inputs_are_rejected
tests/unit/test_protocol_manifest.py::test_authoritative_manifest_passes_validation
tests/unit/test_protocol_manifest.py::test_seed_governance_rejection
tests/unit/test_protocol_manifest.py::test_hash_tamper_rejection
tests/unit/test_protocol_manifest.py::test_parameter_tamper_rejection
tests/unit/test_protocol_manifest.py::test_comparator_methods_completeness
tests/unit/test_protocol_manifest.py::test_duplicate_method_ids_rejection
tests/unit/test_protocol_manifest.py::test_method_parameter_bounds_rejection
tests/unit/test_protocol_manifest.py::test_missing_required_section_rejection
tests/unit/test_protocol_manifest.py::test_failure_taxonomy_completeness_and_distinction
tests/unit/test_protocol_manifest.py::test_statistical_plan_and_multiplicity_rejection
tests/unit/test_protocol_manifest.py::test_formula_metrics_and_environment_rejection
tests/unit/test_protocol_manifest.py::test_overwrite_protection
tests/unit/test_protocol_manifest.py::test_prospective_execution_refused_on_proposed_status
tests/unit/test_protocol_manifest.py::test_prospective_execution_refused_on_null_seed
tests/unit/test_protocol_manifest.py::test_prospective_execution_accepted_on_frozen_authorized
tests/unit/test_protocol_manifest.py::test_d030_rejected_for_prospective_authorization
tests/unit/test_protocol_manifest.py::test_retired_seed_rejection_in_prospective
tests/unit/test_protocol_manifest.py::test_failure_taxonomy_orthogonal_fields_validation
tests/unit/test_protocol_manifest.py::test_classify_attack_outcome_mapping
tests/unit/test_protocol_manifest.py::test_prospective_manifest_rejects_seed_406737
tests/unit/test_protocol_manifest.py::test_prospective_dataset_validation_rejects_seed_406737
tests/unit/test_protocol_manifest.py::test_development_dataset_inspection_allowed_without_prospective_status
tests/unit/test_protocol_manifest.py::test_seed_policy_governance_for_406737
tests/unit/test_protocol_manifest.py::test_critical_code_mutation_invalidates_manifest
tests/unit/test_publication_evidence_package.py::test_relocation_retains_original_bytes_and_truth_separation
tests/unit/test_publication_evidence_package.py::test_nonportable_paths_fail_closed[.]
tests/unit/test_publication_evidence_package.py::test_nonportable_paths_fail_closed[../bad]
tests/unit/test_publication_evidence_package.py::test_nonportable_paths_fail_closed[/absolute]
tests/unit/test_publication_evidence_package.py::test_nonportable_paths_fail_closed[a/../b]
tests/unit/test_publication_evidence_package.py::test_nonportable_paths_fail_closed[a\\b]
tests/unit/test_publication_evidence_package.py::test_nonportable_paths_fail_closed[a//b]
tests/unit/test_publication_evidence_package.py::test_output_overwrite_and_source_tampering_are_rejected
tests/unit/test_publication_evidence_package.py::test_package_corruption_and_unexpected_files_are_rejected[missing]
tests/unit/test_publication_evidence_package.py::test_package_corruption_and_unexpected_files_are_rejected[extra]
tests/unit/test_publication_evidence_package.py::test_package_corruption_and_unexpected_files_are_rejected[altered]
tests/unit/test_publication_evidence_package.py::test_package_corruption_and_unexpected_files_are_rejected[symlink]
tests/unit/test_publication_evidence_package.py::test_package_corruption_and_unexpected_files_are_rejected[root_license]
tests/unit/test_publication_evidence_package.py::test_package_corruption_and_unexpected_files_are_rejected[source_license]
tests/unit/test_publication_evidence_package.py::test_license_provenance_detects_copies_changed_with_rehashed_file_manifest
tests/unit/test_publication_evidence_package.py::test_schema1_bundle_without_license_remains_verifiable
tests/unit/test_publication_evidence_package.py::test_build_requires_unchanged_committed_root_license_before_output[missing]
tests/unit/test_publication_evidence_package.py::test_build_requires_unchanged_committed_root_license_before_output[modified]
tests/unit/test_publication_evidence_package.py::test_build_requires_unchanged_committed_root_license_before_output[staged]
tests/unit/test_publication_evidence_package.py::test_build_requires_unchanged_committed_root_license_before_output[untracked]
tests/unit/test_publication_evidence_package.py::test_build_rechecks_license_before_creating_output
tests/unit/test_publication_evidence_package.py::test_portable_metadata_cannot_be_changed_even_if_local_hashes_updated
tests/unit/test_publication_evidence_package.py::test_duplicate_json_keys_rejected
tests/unit/test_publication_evidence_package.py::test_original_detached_sidecar_corruption_rejects_packaging
tests/unit/test_publication_evidence_package.py::test_formula_symlink_outside_pinned_directory_rejected
tests/unit/test_publication_evidence_package.py::test_scalar_namespace_loads_leaf_without_eager_training_exports
tests/unit/test_pure_cnf.py::test_speck_pure_cnf_export_solved_by_cadical_and_cms
tests/unit/test_pure_cnf.py::test_speck_pure_cnf_export_with_pinned_keys
tests/unit/test_pure_cnf_adversarial.py::test_empty_contradiction_survives_conversion[0]
tests/unit/test_pure_cnf_adversarial.py::test_empty_contradiction_survives_conversion[x 0]
tests/unit/test_pure_cnf_adversarial.py::test_empty_contradiction_survives_conversion[x0]
tests/unit/test_pure_cnf_adversarial.py::test_empty_formula_is_tautology
tests/unit/test_pure_cnf_adversarial.py::test_generic_empty_xor_preserves_both_rhs_values[False]
tests/unit/test_pure_cnf_adversarial.py::test_generic_empty_xor_preserves_both_rhs_values[True]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals0]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals1]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals2]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals3]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals4]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals5]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals6]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals7]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[False-literals8]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals0]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals1]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals2]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals3]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals4]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals5]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals6]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals7]
tests/unit/test_pure_cnf_adversarial.py::test_signed_repeated_xors_have_exact_projected_semantics[True-literals8]
tests/unit/test_pure_cnf_adversarial.py::test_multiple_long_xors_use_disjoint_fresh_auxiliaries
tests/unit/test_pure_cnf_adversarial.py::test_declared_unused_variables_are_reserved
tests/unit/test_pure_cnf_adversarial.py::test_wrapped_and_multiple_records_preserve_semantics
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[1 0\n-1-header]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[-0-header]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\np cnf 1 1\n1 0\n-1-header]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 0\n1 0\n-1-record count]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 2\n1 0\n-1-record count]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\n1\n-1-unterminated]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\nx 1\n-1-unterminated]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\nx\n-1-unterminated]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 2\n1 x 1 0\n-1-unterminated]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\n2 0\n-1-range]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\nx -2 0\n-1-range]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 6 1\nx 1 2 3 4 5 6 0\n-5-match]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf -1 0\n-0-negative]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 -1\n-1-negative]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf one 0\n-1-noninteger]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1_0 0\n-10-noninteger]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\n\u0661 0\n-1-noninteger]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 10 1\n1_0 0\n-10-noninteger]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 0 trailing\n-1-expected header]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p other 1 0\n-1-expected header]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\nword 0\n-1-noninteger]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\nx 1 0 0\n-1-record count]
tests/unit/test_pure_cnf_adversarial.py::test_malformed_input_rejected_before_conversion[p cnf 1 1\n1 0 1\n-1-unterminated]
tests/unit/test_pure_cnf_adversarial.py::test_invalid_supplied_variable_count_rejected[True]
tests/unit/test_pure_cnf_adversarial.py::test_invalid_supplied_variable_count_rejected[False]
tests/unit/test_pure_cnf_adversarial.py::test_invalid_supplied_variable_count_rejected[-1]
tests/unit/test_pure_cnf_adversarial.py::test_invalid_supplied_variable_count_rejected[1.0]
tests/unit/test_pure_cnf_adversarial.py::test_invalid_supplied_variable_count_rejected[1]
tests/unit/test_pure_cnf_adversarial.py::test_invalid_supplied_variable_count_rejected[None]
tests/unit/test_queue_order_environment.py::test_author_allows_battery_or_ac_with_observed_sleep_protection[False]
tests/unit/test_queue_order_environment.py::test_author_allows_battery_or_ac_with_observed_sleep_protection[True]
tests/unit/test_queue_order_environment.py::test_battery_allowance_keeps_telemetry_and_sleep_gates[supported-False]
tests/unit/test_queue_order_environment.py::test_battery_allowance_keeps_telemetry_and_sleep_gates[sleep_assertion_observed-False]
tests/unit/test_queue_order_environment.py::test_battery_allowance_keeps_telemetry_and_sleep_gates[assertions_returncode-1]
tests/unit/test_queue_order_environment.py::test_battery_allowance_keeps_telemetry_and_sleep_gates[power_returncode-1]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[Now drawing from 'Unknown Power'\n-False]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[-False]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[None-False]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[Now drawing from 'Battery Power'\nNow drawing from 'AC Power'\n-True]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[Now drawing from 'AC Power'\nNow drawing from 'Unknown Power'\n-True]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[Now drawing from 'Battery Power'\n-True]
tests/unit/test_queue_order_environment.py::test_unrecognized_ambiguous_or_inconsistent_power_source_rejected[Now drawing from 'AC Power'\n-False]
tests/unit/test_queue_order_environment.py::test_interpreter_alias_accepts_same_venv_but_rejects_other_venv_or_base
tests/unit/test_queue_order_ranking.py::test_speck_composition_matches_original_one_based_stage_ties
tests/unit/test_queue_order_ranking.py::test_simon_composition_matches_original_zero_based_numpy_orders
tests/unit/test_queue_order_ranking.py::test_composers_reject_malformed_prefixes[stage10-stage20]
tests/unit/test_queue_order_ranking.py::test_composers_reject_malformed_prefixes[stage11-stage21]
tests/unit/test_queue_order_ranking.py::test_composers_reject_malformed_prefixes[stage12-stage22]
tests/unit/test_queue_order_ranking.py::test_composers_reject_malformed_prefixes[stage13-stage23]
tests/unit/test_queue_order_ranking.py::test_speck_rank_queue_uses_frozen_stage_calls_and_public_only_instance
tests/unit/test_queue_order_ranking.py::test_speck_ranking_fails_closed_without_mps
tests/unit/test_queue_order_ranking.py::test_simon_classical_uses_fixed_public_ranking_settings_and_numpy_orders
tests/unit/test_queue_order_ranking.py::test_simon_bao_path_uses_existing_pinned_subprocess_builder
tests/unit/test_queue_order_ranking.py::test_rank_queue_rejects_truth_fields_and_policy_drift_before_dispatch
tests/unit/test_queue_order_worker.py::test_traversal_reuses_model_after_unsat_but_starts_a_fresh_model_per_arm
tests/unit/test_queue_order_worker.py::test_unresolved_status_rebuilds_before_next_candidate[solver_timeout]
tests/unit/test_queue_order_worker.py::test_unresolved_status_rebuilds_before_next_candidate[unsat_or_timeout]
tests/unit/test_queue_order_worker.py::test_verification_failure_rebuilds_and_is_not_reported_as_recovery
tests/unit/test_queue_order_worker.py::test_traverse_rejects_wrong_observation_counts_before_model_construction[127-2]
tests/unit/test_queue_order_worker.py::test_traverse_rejects_wrong_observation_counts_before_model_construction[128-1]
tests/unit/test_queue_order_worker.py::test_new_model_uses_only_the_leading_four_known_observations[speck32_64]
tests/unit/test_queue_order_worker.py::test_new_model_uses_only_the_leading_four_known_observations[simon32_64]
tests/unit/test_queue_order_worker.py::test_traverse_builds_chronological_round_key_assumptions[speck32_64]
tests/unit/test_queue_order_worker.py::test_traverse_builds_chronological_round_key_assumptions[simon32_64]
tests/unit/test_queue_order_worker.py::test_four_round_cipher_sat_trailing_assumptions_and_reference_verification[speck32_64]
tests/unit/test_queue_order_worker.py::test_four_round_cipher_sat_trailing_assumptions_and_reference_verification[simon32_64]
tests/unit/test_ranking.py::test_true_candidate_maps_to_the_expected_pre_final_state
tests/unit/test_ranking.py::test_correct_candidate_ranks_first_and_model_mode_is_restored
tests/unit/test_ranking.py::test_candidate_batching_produces_identical_scores_and_order
tests/unit/test_ranking.py::test_score_ties_use_ascending_candidate_value
tests/unit/test_ranking.py::test_invalid_candidate_sets_are_rejected[candidate_values0]
tests/unit/test_ranking.py::test_invalid_candidate_sets_are_rejected[candidate_values1]
tests/unit/test_ranking.py::test_invalid_candidate_sets_are_rejected[candidate_values2]
tests/unit/test_ranking.py::test_invalid_candidate_sets_are_rejected[candidate_values3]
tests/unit/test_ranking.py::test_invalid_candidate_sets_are_rejected[candidate_values4]
tests/unit/test_ranking.py::test_empty_instance_is_rejected
tests/unit/test_ranking.py::test_model_must_return_one_logit_per_row
tests/unit/test_ranking.py::test_default_candidate_set_is_the_complete_16_bit_domain
tests/unit/test_ranking.py::test_evaluation_reports_missing_true_candidate_without_guessing
tests/unit/test_ranking.py::test_pair_subset_stability_reuses_per_pair_logits
tests/unit/test_ranking.py::test_invalid_pair_subset_stability_is_rejected[pair_subsets0]
tests/unit/test_ranking.py::test_invalid_pair_subset_stability_is_rejected[pair_subsets1]
tests/unit/test_ranking.py::test_invalid_pair_subset_stability_is_rejected[pair_subsets2]
tests/unit/test_ranking.py::test_invalid_pair_subset_stability_is_rejected[pair_subsets3]
tests/unit/test_ranking.py::test_invalid_pair_subset_stability_is_rejected[pair_subsets4]
tests/unit/test_ranking.py::test_invalid_pair_subset_stability_is_rejected[pair_subsets5]
tests/unit/test_ranking_depth_ranking.py::test_speck_stage1_spy_receives_only_ciphertext_and_fixed_mps_batch
tests/unit/test_ranking_depth_ranking.py::test_speck_sparse_r5_stage2_uses_all_fixed_parents_and_existing_queue_order
tests/unit/test_ranking_depth_ranking.py::test_one_and_two_key_simon_queues_share_persisted_stage1_prefix
tests/unit/test_ranking_depth_ranking.py::test_rejects_nonfrozen_trailing_key_counts[0]
tests/unit/test_ranking_depth_ranking.py::test_rejects_nonfrozen_trailing_key_counts[3]
tests/unit/test_ranking_depth_ranking.py::test_rejects_nonfrozen_trailing_key_counts[True]
tests/unit/test_ranking_depth_ranking.py::test_rejects_nonfrozen_trailing_key_counts[1.0]
tests/unit/test_ranking_depth_ranking.py::test_accepts_only_exact_public_ciphertext_pair_shape[bad_pairs0]
tests/unit/test_ranking_depth_ranking.py::test_accepts_only_exact_public_ciphertext_pair_shape[bad_pairs1]
tests/unit/test_ranking_depth_ranking.py::test_accepts_only_exact_public_ciphertext_pair_shape[bad_pairs2]
tests/unit/test_ranking_depth_ranking.py::test_accepts_only_exact_public_ciphertext_pair_shape[bad_pairs3]
tests/unit/test_ranking_depth_ranking.py::test_rejects_bool_and_duplicate_rank_words
tests/unit/test_ranking_depth_ranking.py::test_tf_child_profile_is_stage1_only_and_has_no_truth_inputs
tests/unit/test_ranking_depth_ranking.py::test_tf_import_boundary_bypasses_simon_init_and_blocks_sat
tests/unit/test_ranking_depth_ranking.py::test_source_receipts_hash_project_package_code_not_runtime_packages
tests/unit/test_reference_encryption_report.py::test_inclusive_boundaries_and_failure_denominator
tests/unit/test_reference_encryption_report.py::test_invalid_cohort_record_rejected[id-b]
tests/unit/test_reference_encryption_report.py::test_invalid_cohort_record_rejected[success-false]
tests/unit/test_reference_encryption_report.py::test_invalid_cohort_record_rejected[wall-nan]
tests/unit/test_reference_encryption_report.py::test_invalid_cohort_record_rejected[wall-True]
tests/unit/test_reference_encryption_report.py::test_invalid_cohort_record_rejected[wall-0.0]
tests/unit/test_reference_encryption_report.py::test_invalid_cohort_record_rejected[wall-300.1]
tests/unit/test_reference_encryption_report.py::test_missing_record_rejected
tests/unit/test_sat_encoding.py::test_four_bit_modular_addition_truth_table
tests/unit/test_sat_encoding.py::test_native_xor_encoder_matches_truth_table
tests/unit/test_screened_group_ranking.py::test_screened_ranking_invariants
tests/unit/test_screened_group_ranking.py::test_screened_ranking_validations
tests/unit/test_seven_round_analysis.py::test_holm_bonferroni_correction
tests/unit/test_seven_round_analysis.py::test_compute_paired_contingency
tests/unit/test_seven_round_analysis.py::test_paired_risk_difference_ci
tests/unit/test_seven_round_analysis.py::test_analyze_records_end_to_end
tests/unit/test_seven_round_analysis.py::test_shuffled_instance_order_alignment
tests/unit/test_seven_round_analysis.py::test_duplicate_instance_id_rejection
tests/unit/test_seven_round_analysis.py::test_missing_method_rejection
tests/unit/test_seven_round_analysis.py::test_mismatched_instance_sets_rejection
tests/unit/test_seven_round_analysis.py::test_contradictory_record_integrity
tests/unit/test_seven_round_analysis.py::test_prospective_requires_n_128
tests/unit/test_seven_round_analysis.py::test_even_n_median_correctness
tests/unit/test_seven_round_analysis.py::test_newcombe_method_10_edge_cases
tests/unit/test_seven_round_analysis.py::test_analyzer_prospective_validation_gate
tests/unit/test_seven_round_comparators.py::test_stage_model_validation
tests/unit/test_seven_round_comparators.py::test_classical_stage_conditional_scoring
tests/unit/test_seven_round_comparators.py::test_truth_free_invariant
tests/unit/test_seven_round_comparators.py::test_global_timeout_during_ranking
tests/unit/test_seven_round_comparators.py::test_global_timeout_in_multi_round_recovery
tests/unit/test_seven_round_comparators.py::test_random_multi_round_candidates
tests/unit/test_seven_round_comparators.py::test_run_random_multi_round_recovery
tests/unit/test_seven_round_comparators.py::test_run_multi_round_sat_only_recovery
tests/unit/test_seven_round_comparators.py::test_solver_isolation_across_comparators
tests/unit/test_seven_round_comparators.py::test_resume_preserves_unselected_methods_behavioral_main
tests/unit/test_seven_round_comparators.py::test_resume_without_output_fails_loudly
tests/unit/test_seven_round_comparators.py::test_method_alias_normalization
tests/unit/test_seven_round_comparators.py::test_candidate_budget_defaults_to_m1_times_m2
tests/unit/test_seven_round_comparators.py::test_sat_only_respects_global_timeout
tests/unit/test_seven_round_comparators.py::test_recovery_continues_after_single_candidate_timeout
tests/unit/test_seven_round_comparators.py::test_solver_rebuilt_on_candidate_timeout
tests/unit/test_seven_round_comparators.py::test_distinct_outcome_categories
tests/unit/test_seven_round_comparators.py::test_resume_fail_closed_validation
tests/unit/test_seven_round_comparators.py::test_sat_only_observation_counts_and_formula_sizes
tests/unit/test_seven_round_comparators.py::test_model_construction_consumes_method_budget_and_clamps_solver_timeout
tests/unit/test_seven_round_comparators.py::test_method_budget_isolation_and_order_invariance
tests/unit/test_seven_round_comparators.py::test_solver_timeout_leaves_instance_unverified_with_cause
tests/unit/test_seven_round_comparators.py::test_runner_prospective_gate_fails_closed_on_proposed_manifest
tests/unit/test_seven_round_comparators.py::test_runner_flags_conflict_rejection
tests/unit/test_seven_round_comparators.py::test_runner_rebuilding_seconds_and_summary
tests/unit/test_seven_round_comparators.py::test_runner_non_interruptible_overshoot_reclassification
tests/unit/test_seven_round_comparators.py::test_runner_resume_validation_fail_closed
tests/unit/test_seven_round_comparators.py::test_runner_late_verified_recovery_reclassification
tests/unit/test_seven_round_comparators.py::test_zero_remaining_time_clean_exit
tests/unit/test_seven_round_comparators.py::test_verify_record_integrity_contradictions
tests/unit/test_seven_round_comparators.py::test_delayed_diagnostics_regression
tests/unit/test_seven_round_comparators.py::test_disjoint_timing_fields
tests/unit/test_seven_round_comparators.py::test_solver_exception_handling
tests/unit/test_seven_round_comparators.py::test_device_enforcement_prospective
tests/unit/test_seven_round_comparators.py::test_counterbalanced_method_order_cyclic
tests/unit/test_seven_round_comparators.py::test_resume_cohort_and_code_drift_validation
tests/unit/test_seven_round_prospective_data.py::test_draw_csprng_seed_rejection_and_policy
tests/unit/test_seven_round_prospective_data.py::test_generation_config_hash_invariance
tests/unit/test_seven_round_prospective_data.py::test_prohibited_seeds_contain_all_critical_seeds
tests/unit/test_seven_round_prospective_data.py::test_small_cohort_generation_and_deterministic_replay
tests/unit/test_simon_attack_instance.py::test_demo_instance_generation_is_deterministic
tests/unit/test_simon_attack_instance.py::test_input_difference_structure_and_encryption
tests/unit/test_simon_attack_instance.py::test_fresh_verification_pairs_are_independently_valid
tests/unit/test_simon_attack_instance.py::test_generator_rejects_attack_and_verification_plaintext_collisions
tests/unit/test_simon_attack_instance.py::test_instance_rejects_overlap_after_deserialization_or_manual_construction
tests/unit/test_simon_attack_instance.py::test_zero_effective_input_difference_is_rejected[input_difference0]
tests/unit/test_simon_attack_instance.py::test_zero_effective_input_difference_is_rejected[input_difference1]
tests/unit/test_simon_attack_instance.py::test_impossible_unique_plaintext_request_is_rejected
tests/unit/test_simon_attack_instance.py::test_opened_demo_instances_replay_unchanged[8-8a762dee8ab2f4d0e8e96c10f9f3fb3e00a681fa419bfd18104570cadb1f16e8]
tests/unit/test_simon_attack_instance.py::test_opened_demo_instances_replay_unchanged[9-b294ee5fed9e0b6ae79437ef976da67b5e061b5f3318858948fd9bfcea98bca4]
tests/unit/test_simon_attack_instance.py::test_opened_demo_instances_replay_unchanged[10-e4e9b03e21a67c28d7764ab8e759f6c7bb2d8377f32c89e164946e45b3357718]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[98001]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[97001]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[99001]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[100001]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[101001]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[300001]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[406737]
tests/unit/test_simon_attack_instance.py::test_seed_governance_rejects_protected_seeds[669981]
tests/unit/test_simon_attack_instance.py::test_role_governance_rejects_unauthorized_roles
tests/unit/test_simon_attack_instance.py::test_demo_role_requires_cli_demo_seed
tests/unit/test_simon_attack_instance.py::test_development_role_cannot_relabel_opened_demo_seed
tests/unit/test_simon_attack_instance.py::test_zero_or_negative_fresh_verification_count_is_rejected[0]
tests/unit/test_simon_attack_instance.py::test_zero_or_negative_fresh_verification_count_is_rejected[-1]
tests/unit/test_simon_attack_instance.py::test_noninteger_verification_count_is_rejected[True]
tests/unit/test_simon_attack_instance.py::test_noninteger_verification_count_is_rejected[1.5]
tests/unit/test_simon_attack_instance.py::test_noninteger_verification_count_is_rejected[2]
tests/unit/test_simon_attack_instance.py::test_ground_truth_isolation_properties
tests/unit/test_simon_attack_instance.py::test_serialization_round_trip
tests/unit/test_simon_b1_data.py::test_training_key_role_audit_rejects_duplicates_and_cross_role_overlap
tests/unit/test_simon_b1_data.py::test_fixed_9round_direct_pair_construction
tests/unit/test_simon_b1_data.py::test_group_master_key_sharing
tests/unit/test_simon_b1_data.py::test_strict_master_key_splitting
tests/unit/test_simon_b1_data.py::test_label_balance_and_permutation
tests/unit/test_simon_b1_data.py::test_deterministic_replay
tests/unit/test_simon_b1_data.py::test_feature_layout_and_bit_slicing
tests/unit/test_simon_b1_data.py::test_to_b0_data_flattening
tests/unit/test_simon_b1_data.py::test_pytorch_datasets_and_dataloaders
tests/unit/test_simon_b1_data.py::test_train_and_eval_step_b0
tests/unit/test_simon_b1_data.py::test_train_and_eval_step_b1
tests/unit/test_simon_b1_data.py::test_b0_weight_transfer_initialize_from_b0
tests/unit/test_simon_b1_data.py::test_checkpoint_save_and_load
tests/unit/test_simon_b1_data.py::test_synthetic_overfit_check_b1
tests/unit/test_simon_b1_data.py::test_synthetic_overfit_check_b0
tests/unit/test_simon_b1_data.py::test_proposed_training_manifest_specification
tests/unit/test_simon_b1_training_runner.py::test_proposed_manifest_loads_and_validates
tests/unit/test_simon_b1_training_runner.py::test_manifest_validation_rejects_missing_file_and_invalid_json
tests/unit/test_simon_b1_training_runner.py::test_manifest_validation_rejects_missing_required_keys
tests/unit/test_simon_b1_training_runner.py::test_manifest_validation_rejects_invalid_governing_decisions
tests/unit/test_simon_b1_training_runner.py::test_manifest_validation_rejects_invalid_architectures
tests/unit/test_simon_b1_training_runner.py::test_manifest_validation_rejects_forbidden_seeds
tests/unit/test_simon_b1_training_runner.py::test_manifest_validation_rejects_invalid_hyperparameters
tests/unit/test_simon_b1_training_runner.py::test_key_audit_rejects_duplicates_before_training
tests/unit/test_simon_b1_training_runner.py::test_key_audit_rejects_train_val_overlap_before_training
tests/unit/test_simon_b1_training_runner.py::test_preflight_cryptography_scalar_and_oracle_verification
tests/unit/test_simon_b1_training_runner.py::test_bounded_smoke_test_execution_and_checkpoint_reload
tests/unit/test_simon_b1_training_runner.py::test_deterministic_checkpoint_selection_on_tied_and_improving_losses
tests/unit/test_simon_b1_training_runner.py::test_deterministic_checkpoint_selection_in_b0_stage
tests/unit/test_simon_b1_training_runner.py::test_weak_model_retained_and_labeled_descriptive
tests/unit/test_simon_b1_training_runner.py::test_nan_loss_aborts_immediately
tests/unit/test_simon_b1_training_runner.py::test_zero_gradients_aborts_immediately
tests/unit/test_simon_b1_training_runner.py::test_non_finite_weights_aborts_immediately
tests/unit/test_simon_b1_training_runner.py::test_cli_requires_smoke_or_confirmation
tests/unit/test_simon_b1_training_runner.py::test_cli_smoke_test_runs_cleanly
tests/unit/test_simon_candidate_history_observation.py::test_complete_grid_derives_only_matched_target_with_unsat_history
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[duplicate-Duplicate method/key]
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[missing-Incomplete method/key]
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[flag-Non-Boolean]
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[false_flag-Outcome/flag conflict]
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[wrong_target-Target tuple differs]
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[rebuild-Intervening solver rebuild]
tests/unit/test_simon_candidate_history_observation.py::test_corrupt_or_non_history_evidence_is_rejected[unknown_prefix-non-UNSAT]
tests/unit/test_simon_candidate_history_observation.py::test_false_master_key_and_failed_fresh_verification_are_rejected
tests/unit/test_simon_candidate_history_observation.py::test_frozen_hash_sidecar_and_duplicate_fields
tests/unit/test_simon_cipher.py::test_one_round_encryption_matches_regression_state
tests/unit/test_simon_cipher.py::test_one_round_decryption_is_the_inverse
tests/unit/test_simon_cipher.py::test_full_known_answer_vector
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[1-expected0]
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[2-expected1]
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[4-expected2]
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[8-expected3]
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[16-expected4]
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[24-expected5]
tests/unit/test_simon_cipher.py::test_reduced_round_outputs_use_the_round_key_prefix[32-expected6]
tests/unit/test_simon_cipher.py::test_key_schedule_matches_all_expected_round_keys
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[1]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[2]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[3]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[4]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[5]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[6]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[7]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[8]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[9]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[10]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[11]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[12]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[13]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[14]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[15]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[16]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[17]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[18]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[19]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[20]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[21]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[22]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[23]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[24]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[25]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[26]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[27]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[28]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[29]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[30]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[31]
tests/unit/test_simon_cipher.py::test_key_schedule_prefix_consistency[32]
tests/unit/test_simon_cipher.py::test_partial_decrypt_final_round_matches_prior_round
tests/unit/test_simon_cipher.py::test_simon_pair_difference_needs_candidate_sensitive_descent
tests/unit/test_simon_cipher.py::test_direct_pair_ranking_difference_has_r_minus_two_then_r_minus_three_depth
tests/unit/test_simon_cipher.py::test_deterministic_random_round_trips
tests/unit/test_simon_cipher.py::test_independent_reference_oracle_kat
tests/unit/test_simon_cipher.py::test_independent_reference_oracle_differential_random_comparison
tests/unit/test_simon_cipher.py::test_round_and_block_inputs_are_explicitly_masked
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block-0]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block--1]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block-33]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[encrypt_block-100]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block-0]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block--1]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block-33]
tests/unit/test_simon_cipher.py::test_invalid_round_counts_are_rejected[decrypt_block-100]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[encrypt_block-1.0]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[encrypt_block-True]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[encrypt_block-32]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[encrypt_block-None]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[decrypt_block-1.0]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[decrypt_block-True]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[decrypt_block-32]
tests/unit/test_simon_cipher.py::test_non_integer_round_counts_are_rejected[decrypt_block-None]
tests/unit/test_simon_cipher.py::test_malformed_round_state_is_rejected
tests/unit/test_simon_cipher.py::test_key_schedule_validation
tests/unit/test_simon_cipher.py::test_primitives_properties
tests/unit/test_simon_classical_ranking.py::test_ondemand_stage1_removes_temporary_output_on_success_and_failure
tests/unit/test_simon_classical_ranking.py::test_sparse_full_distribution_row_has_exact_zero_bins
tests/unit/test_simon_classical_ranking.py::test_row_gather_score_matches_independent_scalar_source_index[None]
tests/unit/test_simon_classical_ranking.py::test_row_gather_score_matches_independent_scalar_source_index[0]
tests/unit/test_simon_classical_ranking.py::test_row_gather_score_matches_independent_scalar_source_index[4660]
tests/unit/test_simon_classical_ranking.py::test_complete_screen_and_rescore_use_deterministic_ascending_ties[None-7]
tests/unit/test_simon_classical_ranking.py::test_complete_screen_and_rescore_use_deterministic_ascending_ties[7-6]
tests/unit/test_simon_classical_ranking.py::test_nan_row_and_wrong_stage_fail_explicitly
tests/unit/test_simon_classical_ranking.py::test_tie_count_and_collision_count_semantics
tests/unit/test_simon_classical_ranking.py::test_rank_classical_domain_rescores_all_64_pairs
tests/unit/test_simon_confirmation_analysis.py::test_extract_record_success
tests/unit/test_simon_confirmation_analysis.py::test_compute_method_summary
tests/unit/test_simon_confirmation_analysis.py::test_analyze_all_comparators_complete
tests/unit/test_simon_confirmation_analysis.py::test_analyze_instance_mismatch_fails_closed
tests/unit/test_simon_confirmation_analysis.py::test_load_ledger_records_formats
tests/unit/test_simon_confirmation_analysis.py::test_generate_markdown_audit_report
tests/unit/test_simon_confirmation_analysis.py::test_missing_oracle_fails_closed
tests/unit/test_simon_confirmation_analysis.py::test_corrupt_sidecar_fails_closed
tests/unit/test_simon_confirmation_analysis.py::test_drift_rejected_in_main
tests/unit/test_simon_confirmation_guards.py::test_unapproved_manifest_rejected
tests/unit/test_simon_confirmation_guards.py::test_unauthorized_execution_rejected
tests/unit/test_simon_confirmation_guards.py::test_null_seed_rejected
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[260926]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[406737]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[98001]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[880001]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[880002]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[999999]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[669981]
tests/unit/test_simon_confirmation_guards.py::test_ineligible_seeds_rejected[101001]
tests/unit/test_simon_confirmation_guards.py::test_invalid_selected_instances_rejected[None]
tests/unit/test_simon_confirmation_guards.py::test_invalid_selected_instances_rejected[16]
tests/unit/test_simon_confirmation_guards.py::test_invalid_selected_instances_rejected[48]
tests/unit/test_simon_confirmation_guards.py::test_invalid_selected_instances_rejected[128]
tests/unit/test_simon_confirmation_guards.py::test_invalid_selected_instances_rejected[32]
tests/unit/test_simon_confirmation_guards.py::test_missing_or_modified_methods_rejected
tests/unit/test_simon_confirmation_guards.py::test_modified_queue_control_seed_rejected
tests/unit/test_simon_confirmation_guards.py::test_spy_verifies_zero_worker_calls_on_preflight_rejection
tests/unit/test_simon_confirmation_guards.py::test_provenance_verification_fails_closed_on_hash_drift
tests/unit/test_simon_confirmation_guards.py::test_disjoint_family_certificate_rejects_duplicate_keys
tests/unit/test_simon_confirmation_guards.py::test_disjoint_family_certificate_detects_development_overlap
tests/unit/test_simon_confirmation_guards.py::test_disjoint_family_certificate_rejects_non_confirmation_role
tests/unit/test_simon_confirmation_guards.py::test_disjoint_family_certificate_build_and_verify_passes_on_fresh_keys
tests/unit/test_simon_confirmation_guards.py::test_resume_guards_reject_unflagged_rerun_on_existing_output
tests/unit/test_simon_confirmation_guards.py::test_resume_guards_reject_missing_checksum_sidecar
tests/unit/test_simon_confirmation_guards.py::test_resume_guards_reject_corrupted_checksum_sidecar
tests/unit/test_simon_confirmation_guards.py::test_resume_guards_reject_tampered_execution_identity
tests/unit/test_simon_confirmation_guards.py::test_resume_guards_reject_duplicate_cell
tests/unit/test_simon_confirmation_guards.py::test_resume_guards_reject_unverified_claimed_recovery
tests/unit/test_simon_confirmation_guards.py::test_resume_correctly_filters_completed_cells
tests/unit/test_simon_confirmation_guards.py::test_forged_zero_certificate_rejected_by_verify_disjoint_family_certificate
tests/unit/test_simon_confirmation_guards.py::test_generate_confirmation_dataset_fails_closed_without_authorization
tests/unit/test_simon_confirmation_guards.py::test_generate_confirmation_dataset_succeeds_when_authorized
tests/unit/test_simon_confirmation_guards.py::test_derive_confirmation_queue_seed_properties
tests/unit/test_simon_confirmation_launch.py::test_nested_profile_drift_is_rejected[resource_policy-solver_threads-2]
tests/unit/test_simon_confirmation_launch.py::test_nested_profile_drift_is_rejected[random_control_stream-prng_algorithm-another generator]
tests/unit/test_simon_confirmation_launch.py::test_nested_profile_drift_is_rejected[statistical_analysis_plan-analysis_freeze_status-changed]
tests/unit/test_simon_confirmation_launch.py::test_nested_profile_drift_is_rejected[candidate_sat_specification-formula_variables-1857]
tests/unit/test_simon_confirmation_launch.py::test_opened_eight_pair_demo_is_in_overlap_union
tests/unit/test_simon_confirmation_launch.py::test_generation_rejects_unapproved_before_any_data_draw
tests/unit/test_simon_confirmation_launch.py::test_generation_refuses_existing_artifact
tests/unit/test_simon_confirmation_launch.py::test_execution_pins_reject_source_change_before_runtime_probe
tests/unit/test_simon_confirmation_launch.py::test_analysis_cannot_accept_missing_confirmation_identity
tests/unit/test_simon_confirmation_launch.py::test_success_flag_cannot_replace_missing_fresh_verification
tests/unit/test_simon_confirmation_launch.py::test_missing_component_time_is_unknown
tests/unit/test_simon_confirmation_launch.py::test_training_replay_matches_original_generator[None]
tests/unit/test_simon_confirmation_launch.py::test_training_replay_matches_original_generator[64]
tests/unit/test_simon_confirmation_launch.py::test_real_frozen_sample_size_cannot_drift
tests/unit/test_simon_confirmation_launch.py::test_same_package_version_changed_native_bytes_changes_identity
tests/unit/test_simon_confirmation_launch.py::test_contradictory_authorization_is_rejected[seed]
tests/unit/test_simon_confirmation_launch.py::test_contradictory_authorization_is_rejected[gate]
tests/unit/test_simon_confirmation_launch.py::test_contradictory_authorization_is_rejected[blocker]
tests/unit/test_simon_confirmation_launch.py::test_confirmation_analysis_rejects_equal_partial_grid
tests/unit/test_simon_exact_ddt_equivalence.py::test_toy_width_exact_transition_equivalence
tests/unit/test_simon_exact_ddt_equivalence.py::test_queried_row_reconstruction_toy
tests/unit/test_simon_exact_ddt_equivalence.py::test_full_width_16bit_affine_basis_rank
tests/unit/test_simon_faithful_b1.py::test_simon_gohr_resnet_shapes_and_param_count
tests/unit/test_simon_faithful_b1.py::test_shared_simon_pair_encoder
tests/unit/test_simon_faithful_b1.py::test_simon_pair_group_ranker_permutation_invariance
tests/unit/test_simon_faithful_b1.py::test_simon_pair_group_ranker_flexible_pair_count[1]
tests/unit/test_simon_faithful_b1.py::test_simon_pair_group_ranker_flexible_pair_count[16]
tests/unit/test_simon_faithful_b1.py::test_simon_pair_group_ranker_flexible_pair_count[32]
tests/unit/test_simon_faithful_b1.py::test_simon_pair_group_ranker_flexible_pair_count[64]
tests/unit/test_simon_faithful_b1.py::test_simon_pair_group_ranker_flexible_pair_count[128]
tests/unit/test_simon_faithful_b1.py::test_b0_to_b1_weight_transfer
tests/unit/test_simon_faithful_b1.py::test_batch_norm_single_sample_fallback
tests/unit/test_simon_faithful_b1.py::test_faithful_feature_extraction
tests/unit/test_simon_faithful_b1.py::test_true_r10_key_features_equal_independent_nine_round_oracle
tests/unit/test_simon_faithful_b1.py::test_b1_group_mean_is_invariant_to_fourfold_pair_duplication
tests/unit/test_simon_faithful_b1.py::test_faithful_ranking_pair_count_enforcement
tests/unit/test_simon_faithful_b1.py::test_faithful_deterministic_tie_breaking
tests/unit/test_simon_faithful_b1.py::test_faithful_hybrid_8parent_queue
tests/unit/test_simon_faithful_b1.py::test_audit_contrast_faithful_vs_toy
tests/unit/test_simon_higher_round.py::test_simons11_schedule_word_order_and_chronological_trailing_window
tests/unit/test_simon_higher_round.py::test_peel_order_matches_independent_reduced_round_reference
tests/unit/test_simon_higher_round.py::test_queue_order_ties_are_deterministic_and_duplicate_paths_keep_first
tests/unit/test_simon_higher_round.py::test_native_simon11_completion_checks_128_known_blocks_and_two_fresh_pairs
tests/unit/test_simon_higher_round.py::test_run_cell_success_uses_stubbed_rankers_and_fresh_verification
tests/unit/test_simon_higher_round.py::test_run_cell_rejects_completion_that_fails_fresh_verification
tests/unit/test_simon_higher_round.py::test_run_cell_stops_dispatch_after_ranking_crosses_deadline
tests/unit/test_simon_higher_round.py::test_run_cell_does_not_accept_verification_completed_after_deadline
tests/unit/test_simon_higher_round.py::test_run_cell_rejects_malformed_instance_before_ranking[rounds]
tests/unit/test_simon_higher_round.py::test_run_cell_rejects_malformed_instance_before_ranking[pairs]
tests/unit/test_simon_higher_round.py::test_run_cell_rejects_malformed_instance_before_ranking[verification_pairs]
tests/unit/test_simon_higher_round.py::test_run_cell_rejects_malformed_instance_before_ranking[overlap]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_accepts_only_demo_without_explicit_role
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-98001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-97001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-99001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-101001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-201001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-300001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-406737]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-669981]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-6503664708853956672]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[demo-701004]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-98001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-97001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-99001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-101001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-201001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-300001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-406737]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-669981]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-6503664708853956672]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[development-701004]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-98001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-97001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-99001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-101001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-201001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-300001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-406737]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-669981]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-6503664708853956672]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[feasibility-701004]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-98001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-97001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-99001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-101001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-201001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-300001]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-406737]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-669981]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-6503664708853956672]
tests/unit/test_simon_higher_round.py::test_validate_seed_role_rejects_registered_or_cross_cipher_opened_seeds[confirmation-701004]
tests/unit/test_simon_higher_round.py::test_parse_args_requires_run_parameters_and_limits_tensorflow_threads
tests/unit/test_simon_higher_round.py::test_ledger_creation_is_exclusive_and_append_requires_existing_file
tests/unit/test_simon_higher_round.py::test_classical_and_hybrid_stages_batch_parents_once[S11-C]
tests/unit/test_simon_higher_round.py::test_classical_and_hybrid_stages_batch_parents_once[S11-H]
tests/unit/test_simon_higher_round.py::test_row_score_ties_use_word_order_and_nonfinite_scores_fail
tests/unit/test_simon_higher_round.py::test_bad_table_header_is_rejected_without_dense_allocation
tests/unit/test_simon_higher_round.py::test_batch_rows_reject_invalid_probabilities[-1.0]
tests/unit/test_simon_higher_round.py::test_batch_rows_reject_invalid_probabilities[nan]
tests/unit/test_simon_higher_round.py::test_batch_rows_reject_invalid_probabilities[1.01]
tests/unit/test_simon_higher_round.py::test_header_pins_sources_and_only_assets_used_by_selected_arm
tests/unit/test_simon_higher_round.py::test_rejected_cli_never_draws_data_or_initializes_workers[existing_ledger]
tests/unit/test_simon_higher_round.py::test_rejected_cli_never_draws_data_or_initializes_workers[protected_seed]
tests/unit/test_simon_higher_round.py::test_rejected_cli_never_draws_data_or_initializes_workers[missing_threads]
tests/unit/test_simon_higher_round.py::test_rejected_cli_never_draws_data_or_initializes_workers[role_specific_seed]
tests/unit/test_simon_higher_round.py::test_rejected_cli_never_draws_data_or_initializes_workers[unfrozen_role]
tests/unit/test_simon_higher_round.py::test_loader_time_is_charged_and_no_rank_starts_at_exact_deadline
tests/unit/test_simon_higher_round.py::test_vv10_pin_registration_is_scoped_to_one_load
tests/unit/test_simon_higher_round.py::test_serial_schedule_rotates_by_absolute_key_index_with_stubbed_cells
tests/unit/test_simon_higher_round.py::test_wrong_input_difference_is_rejected_before_model_load
tests/unit/test_simon_higher_round.py::test_posthoc_truth_time_does_not_turn_ontime_recovery_into_deadline
tests/unit/test_simon_operating_points_map.py::test_cohort_identity_slicing_and_divergence
tests/unit/test_simon_operating_points_map.py::test_deterministic_replay
tests/unit/test_simon_operating_points_map.py::test_independent_solver_construction
tests/unit/test_simon_operating_points_map.py::test_worker_solve_budget_exhausted
tests/unit/test_simon_operating_points_map.py::test_worker_solve_timeout_classification
tests/unit/test_simon_operating_points_map.py::test_fresh_pair_verification_rejects_wrong_key
tests/unit/test_simon_operating_points_map.py::test_run_map_workers_guard
tests/unit/test_simon_operating_points_map.py::test_parent_deadline_kills_overrunning_worker
tests/unit/test_simon_operating_points_map.py::test_worker_finishes_just_within_deadline
tests/unit/test_simon_operating_points_map.py::test_deadline_classifications_separate_outcomes
tests/unit/test_simon_operating_points_map.py::test_run_map_r10_serial_profile_cells
tests/unit/test_simon_operating_points_map.py::test_resume_rejects_changed_profile_or_worker_count
tests/unit/test_simon_operating_window.py::test_observations_are_ciphertexts_not_differential_pairs
tests/unit/test_simon_operating_window.py::test_demo_keys_are_distinct_and_pairing_is_stable
tests/unit/test_simon_operating_window.py::test_wrong_tuple_probe_is_unique_truth_excluded_and_reproducible
tests/unit/test_simon_operating_window.py::test_grid_screen_applies_paired_counts_and_censored_median
tests/unit/test_simon_operating_window.py::test_checkpoint_refuses_overwrite_and_bad_resume
tests/unit/test_simon_operating_window.py::test_one_true_key_control_matches_reference_and_fresh_verification
tests/unit/test_simon_operating_window.py::test_solver_timeout_is_unresolved_not_unsat
tests/unit/test_simon_operating_window.py::test_detached_checksum_refuses_modified_artifact
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[0-3]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[0-4]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[0-6]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[1-3]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[1-4]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[1-6]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[3-3]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[3-4]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[3-6]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[7-3]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[7-4]
tests/unit/test_simon_predicted_ddt.py::test_full_toy_distribution_matches_exhaustive_scalar_enumeration[7-6]
tests/unit/test_simon_predicted_ddt.py::test_full_width_affine_support_matches_exhaustive_scalar_histogram[0]
tests/unit/test_simon_predicted_ddt.py::test_full_width_affine_support_matches_exhaustive_scalar_histogram[1]
tests/unit/test_simon_predicted_ddt.py::test_full_width_affine_support_matches_exhaustive_scalar_histogram[64]
tests/unit/test_simon_predicted_ddt.py::test_full_width_affine_support_matches_exhaustive_scalar_histogram[128]
tests/unit/test_simon_predicted_ddt.py::test_full_width_affine_support_matches_exhaustive_scalar_histogram[4660]
tests/unit/test_simon_predicted_ddt.py::test_full_width_affine_support_matches_exhaustive_scalar_histogram[65535]
tests/unit/test_simon_predicted_ddt.py::test_affine_kernel_matches_pinned_source_probability_for_all_outputs[0]
tests/unit/test_simon_predicted_ddt.py::test_affine_kernel_matches_pinned_source_probability_for_all_outputs[1]
tests/unit/test_simon_predicted_ddt.py::test_affine_kernel_matches_pinned_source_probability_for_all_outputs[64]
tests/unit/test_simon_predicted_ddt.py::test_affine_kernel_matches_pinned_source_probability_for_all_outputs[4660]
tests/unit/test_simon_predicted_ddt.py::test_affine_kernel_matches_pinned_source_probability_for_all_outputs[65535]
tests/unit/test_simon_predicted_ddt.py::test_reference_rejects_full_width_generation_and_malformed_inputs
tests/unit/test_simon_published_vv_ranking.py::test_published_log_odds_polarity_matches_source_formula
tests/unit/test_simon_published_vv_ranking.py::test_published_log_odds_rejects_malformed_outputs[malformed0]
tests/unit/test_simon_published_vv_ranking.py::test_published_log_odds_rejects_malformed_outputs[malformed1]
tests/unit/test_simon_published_vv_ranking.py::test_published_log_odds_rejects_malformed_outputs[malformed2]
tests/unit/test_simon_published_vv_ranking.py::test_published_log_odds_rejects_malformed_outputs[malformed3]
tests/unit/test_simon_published_vv_ranking.py::test_published_log_odds_rejects_malformed_outputs[malformed4]
tests/unit/test_simon_published_vv_ranking.py::test_complete_domain_uses_candidate_word_ties[None]
tests/unit/test_simon_published_vv_ranking.py::test_complete_domain_uses_candidate_word_ties[7]
tests/unit/test_simon_published_vv_ranking.py::test_neural_ranker_rejects_invalid_pair_budget
tests/unit/test_simon_published_vv_ranking.py::test_neural_candidate_pair_order_matches_independent_scalar_oracle[None]
tests/unit/test_simon_published_vv_ranking.py::test_neural_candidate_pair_order_matches_independent_scalar_oracle[22136]
tests/unit/test_simon_published_vv_ranking.py::test_vv9_and_vv8_feature_and_score_aggregation_pinned_reference
tests/unit/test_simon_published_vv_ranking.py::test_rank_neural_domain_rescores_all_64_pairs
tests/unit/test_simon_ranking_features.py::test_source_word_and_msb_bit_order
tests/unit/test_simon_ranking_features.py::test_vectorized_candidate_states_match_scalar_and_independent_oracle[None]
tests/unit/test_simon_ranking_features.py::test_vectorized_candidate_states_match_scalar_and_independent_oracle[0]
tests/unit/test_simon_ranking_features.py::test_vectorized_candidate_states_match_scalar_and_independent_oracle[65535]
tests/unit/test_simon_ranking_features.py::test_correct_keys_have_independent_seven_and_six_round_differences
tests/unit/test_simon_ranking_features.py::test_direct_difference_expression_and_candidate_collision
tests/unit/test_simon_ranking_features.py::test_malformed_candidate_feature_inputs_are_rejected[pairs0-candidates0-None-ValueError]
tests/unit/test_simon_ranking_features.py::test_malformed_candidate_feature_inputs_are_rejected[pairs1-candidates1-None-ValueError]
tests/unit/test_simon_ranking_features.py::test_malformed_candidate_feature_inputs_are_rejected[pairs2-candidates2-None-ValueError]
tests/unit/test_simon_ranking_features.py::test_malformed_candidate_feature_inputs_are_rejected[pairs3-candidates3-None-TypeError]
tests/unit/test_simon_ranking_features.py::test_malformed_candidate_feature_inputs_are_rejected[pairs4-candidates4--1-ValueError]
tests/unit/test_simon_ranking_features.py::test_malformed_candidate_feature_inputs_are_rejected[pairs5-candidates5-None-ValueError]
tests/unit/test_simon_sat.py::test_simon_sat_model_matches_reference_cipher_on_kat[1]
tests/unit/test_simon_sat.py::test_simon_sat_model_matches_reference_cipher_on_kat[2]
tests/unit/test_simon_sat.py::test_simon_sat_model_matches_reference_cipher_on_kat[4]
tests/unit/test_simon_sat.py::test_simon_sat_model_matches_reference_cipher_on_kat[6]
tests/unit/test_simon_sat.py::test_simon_sat_model_matches_reference_cipher_on_kat[8]
tests/unit/test_simon_sat.py::test_simon_sat_exact_master_key_recovery_with_trailing_assumptions
tests/unit/test_simon_sat.py::test_solver_isolation_between_models
tests/unit/test_simon_sat.py::test_conflicting_assumptions_yield_unsat
tests/unit/test_simon_sat.py::test_timeout_state_classification
tests/unit/test_simon_sat.py::test_master_key_verification_fails_closed
tests/unit/test_simon_sat.py::test_dimacs_export_and_native_cli_solver
tests/unit/test_simon_sat.py::test_simon_pure_cnf_export_solved_by_cadical_and_cms
tests/unit/test_simon_sat_300s_control.py::test_worker_receives_only_attack_and_verification_observations
tests/unit/test_simon_sat_300s_control.py::test_worker_classifies_limit_as_unresolved
tests/unit/test_simon_sat_300s_control.py::test_worker_rejects_truth_bearing_or_wrong_sized_payload
tests/unit/test_simon_sat_300s_control.py::test_parent_watchdog_is_unconditional_failure
tests/unit/test_simon_sat_300s_control.py::test_resume_refuses_source_drift_and_duplicate_records
tests/unit/test_simon_stage2_control.py::test_fixed_job_grid_and_ordered_ciphertext_only_projection
tests/unit/test_simon_stage2_control.py::test_public_payload_and_words_reject_extra_truth_and_non_integer_values
tests/unit/test_simon_stage2_control.py::test_payload_contract_and_rank_sum_queue_order
tests/unit/test_simon_stage2_control.py::test_full_binding_accepts_d053_and_rejects_old_pilot[D-051]
tests/unit/test_simon_stage2_control.py::test_full_binding_accepts_d053_and_rejects_old_pilot[D-052]
tests/unit/test_simon_stage2_control.py::test_full_analyzer_keeps_8_key_denominators_and_uses_phase_costs
tests/unit/test_simon_stage2_control.py::test_analyzer_separates_structural_parent_miss_from_partial_stage2_unavailability
tests/unit/test_simon_stage2_control.py::test_tombstones_keep_unknown_dispatch_and_proven_no_worker_distinct
tests/unit/test_simon_stage2_control.py::test_worker_cleanup_overshoot_is_preserved_without_violating_phase_caps
tests/unit/test_simon_stage2_control.py::test_interrupted_success_receipt_does_not_invent_queue_persistence
tests/unit/test_simon_stage2_control.py::test_only_exact_sealed_orphan_tombstone_is_analyzable
tests/unit/test_simon_stage2_control.py::test_pilot_returns_non_scientific_without_touching_truth_and_closure_is_strict
tests/unit/test_simon_stage2_control.py::test_interpreter_file_pin_preserves_lexical_symlink
tests/unit/test_simon_stage2_receipts.py::test_completed_pilot_accepts_bound_rss_policy_and_ac_receipts
tests/unit/test_simon_stage2_receipts.py::test_battery_is_valid_under_explicit_author_policy
tests/unit/test_simon_stage2_receipts.py::test_pilot_requires_observed_worker_member
tests/unit/test_simon_stage2_receipts.py::test_rss_stop_accepts_only_a_real_over_limit_sample
tests/unit/test_simon_stage2_receipts.py::test_telemetry_must_match_observed_runtime_policy[pid-3999-Telemetry PID]
tests/unit/test_simon_stage2_receipts.py::test_telemetry_must_match_observed_runtime_policy[parent_pid-2999-parent PID]
tests/unit/test_simon_stage2_receipts.py::test_telemetry_must_match_observed_runtime_policy[git_commit-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb-source commit]
tests/unit/test_simon_stage2_receipts.py::test_telemetry_must_match_observed_runtime_policy[tmpdir_safe-False-TMPDIR]
tests/unit/test_simon_stage2_receipts.py::test_telemetry_must_match_observed_runtime_policy[thread_environment-value4-thread environment]
tests/unit/test_simon_stage2_receipts.py::test_telemetry_must_match_observed_runtime_policy[qos_class-9-QoS]
tests/unit/test_simon_stage2_receipts.py::test_process_policy_errno_is_not_hidden_by_accepted_flag
tests/unit/test_simon_stage2_receipts.py::test_sleep_assertion_owner_is_bound_to_guardian
tests/unit/test_simon_stage2_receipts.py::test_corrupt_sample_sidecar_fails_closed
tests/unit/test_simon_stage2_receipts.py::test_partial_power_only_sample_is_allowed_without_worker_observation
tests/unit/test_simon_stage2_receipts.py::test_missing_samples_on_partial_row_remain_unobserved_not_zero_rss
tests/unit/test_simon_stage2_watchdog.py::test_accepts_loaded_then_all_eight_calls_and_advances_deadline
tests/unit/test_simon_stage2_watchdog.py::test_waits_briefly_for_event_sidecar_without_rejecting_complete_event
tests/unit/test_simon_stage2_watchdog.py::test_deadline_stops_hangs_and_retains_only_completed_calls[hang_load-load_timeout-0]
tests/unit/test_simon_stage2_watchdog.py::test_deadline_stops_hangs_and_retains_only_completed_calls[hang_call-rank_timeout-1]
tests/unit/test_simon_stage2_watchdog.py::test_deadline_stops_hangs_and_retains_only_completed_calls[hang_finalize-rank_timeout-2]
tests/unit/test_simon_stage2_watchdog.py::test_corrupt_or_nonsequential_journal_fails_closed[future_event]
tests/unit/test_simon_stage2_watchdog.py::test_corrupt_or_nonsequential_journal_fails_closed[bad_result_hash]
tests/unit/test_simon_stage2_watchdog.py::test_corrupt_or_nonsequential_journal_fails_closed[wrong_result_identity]
tests/unit/test_simon_stage2_watchdog.py::test_corrupt_or_nonsequential_journal_fails_closed[wrong_event_identity]
tests/unit/test_simon_stage2_watchdog.py::test_corrupt_or_nonsequential_journal_fails_closed[backdated_second_call]
tests/unit/test_simon_stage2_watchdog.py::test_campaign_deadline_caps_loading_deadline
tests/unit/test_simon_stage2_watchdog.py::test_configuration_drift_is_rejected_before_worker_dispatch
tests/unit/test_simon_stage2_watchdog.py::test_coordinator_loss_before_dispatch_writes_stop_receipt
tests/unit/test_simon_stage2_watchdog.py::test_coordinator_loss_after_dispatch_stops_worker_group
tests/unit/test_simon_stage2_watchdog.py::test_live_groups_matches_only_exact_stage2_argv
tests/unit/test_simon_stage2_worker.py::test_call_schedule_preserves_all_eight_parent_ordinals
tests/unit/test_simon_stage2_worker.py::test_rank_arguments_keep_the_frozen_screen_rescore_and_depths
tests/unit/test_simon_stage2_worker.py::test_checkpoint_loader_selects_vv9_vv8_and_ddt7_depths
tests/unit/test_simon_stage2_worker.py::test_rank_result_rejects_duplicate_and_wrong_width_orders
tests/unit/test_simon_stage2_worker.py::test_real_ranker_uint16_array_crosses_strict_json_boundary[vv9-None]
tests/unit/test_simon_stage2_worker.py::test_real_ranker_uint16_array_crosses_strict_json_boundary[vv8-123]
tests/unit/test_simon_stage2_worker.py::test_real_ranker_uint16_array_crosses_strict_json_boundary[ddt7-123]
tests/unit/test_simon_stage2_worker.py::test_float_rank_array_is_not_silently_truncated_to_words
tests/unit/test_simon_stage2_worker.py::test_worker_loads_one_checkpoint_and_passes_ciphertext_only_to_eight_calls
tests/unit/test_simon_stage2_worker.py::test_namespace_bootstrap_skips_both_package_initializers
tests/unit/test_simon_stage2_worker.py::test_import_blocker_rejects_solver_and_verifier_modules
tests/unit/test_simon_stage2_worker.py::test_importing_worker_module_does_not_import_project_or_solver_modules
tests/unit/test_solver_isolation.py::test_every_compared_method_receives_distinct_object_identity
tests/unit/test_solver_isolation.py::test_reordering_method_execution_does_not_alter_results
tests/unit/test_solver_isolation.py::test_candidate_assumptions_remain_incremental_within_one_method
tests/unit/test_solver_isolation.py::test_solver_state_cannot_be_passed_across_methods
tests/unit/test_solver_isolation.py::test_resume_logic_recreates_method_local_solver_state
tests/unit/test_solver_isolation.py::test_seed_role_validation_rejects_quarantined_seed_97001
tests/unit/test_solver_isolation.py::test_seed_role_validation_rejects_retired_seed_99001
tests/unit/test_solver_isolation.py::test_cli_default_contains_no_protected_research_seed
tests/unit/test_solver_isolation.py::test_historical_seed_98001_rejected_for_model_selection
tests/unit/test_solver_isolation.py::test_seed_role_validation_accepts_owner_seeds_for_their_roles
tests/unit/test_solver_isolation.py::test_seed_role_validation_rejects_cross_role_contamination
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[   PreventSystemSleep              1\n-True]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[\tPreventUserIdleSystemSleep\t1\t\n-True]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[PreventSystemSleep 0\nPreventUserIdleSystemSleep 1\n-True]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[PreventSystemSleep 1-True]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[PreventSystemSleep 0\nPreventUserIdleSystemSleep 0\n-False]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[PreventSystemSleep 10\n-False]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[OtherPreventSystemSleep 1\n-False]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[pid 42: PreventSystemSleep 1\n-False]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[PreventSystemSleep 1 owner detail\n-False]
tests/unit/test_worker_telemetry.py::test_sleep_assertion_summary_columns[\n-False]
tests/unit/test_worker_telemetry.py::test_sleep_assertions_unsupported_off_darwin
tests/unit/test_worker_telemetry.py::test_live_worker_telemetry_observation
tests/unit/test_worker_telemetry.py::test_adversarial_qos9_zero_priority_rejected
tests/unit/test_worker_telemetry.py::test_unspecified_or_failed_qos_rejected
tests/unit/test_worker_telemetry.py::test_failed_priority_or_io_queries_rejected
tests/unit/test_worker_telemetry.py::test_darwin_bg_state_rejected
tests/unit/test_worker_telemetry.py::test_io_throttle_rejected
tests/unit/test_worker_telemetry.py::test_degraded_process_priority_rejected
tests/unit/test_worker_telemetry.py::test_unsafe_tmpdir_rejected
tests/unit/test_worker_telemetry.py::test_symlink_escape_tmpdir_rejected
tests/unit/test_worker_telemetry.py::test_valid_worker_certification_passes
tests/unit/test_worker_telemetry.py::test_save_worker_receipt_with_sidecar
tests/unit/test_worker_telemetry.py::test_stale_or_mismatched_receipt_identity
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[False-1]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[False-2]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[False-3]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[False-4]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[False-5]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[True-1]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[True-2]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[True-3]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[True-4]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_direct_exhaustive_equivalence[True-5]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[False-4]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[False-5]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[False-6]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[False-7]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[True-4]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[True-5]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[True-6]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_tseitin_exhaustive_equivalence[True-7]
tests/unit/test_xor_to_cnf.py::test_xor_to_cnf_edge_cases
tests/unit/test_xor_to_cnf.py::test_convert_xor_clauses_to_cnf

1377 tests collected in 15.40s

Exit code: 0
Elapsed seconds: 17.328

```

```text
COMMAND: <work>/hamza-venv/Scripts/python.exe -m pytest --collect-only -q tests/integration
tests/integration/test_dataset_v1_fixture.py::test_classifier_fixture_matches_the_generator
tests/integration/test_dataset_v1_fixture.py::test_attack_fixture_matches_the_generator
tests/integration/test_dataset_v1_fixture.py::test_classifier_and_attack_fixtures_are_separate_schemas
tests/integration/test_dataset_v1_fixture.py::test_classifier_fixture_can_be_loaded_as_plain_json
tests/integration/test_phase5_recovery.py::test_candidate_recovery_verifies_the_decoded_master_key
tests/integration/test_phase5_recovery.py::test_sat_only_recovery_requires_independent_verification
tests/integration/test_phase5_recovery.py::test_random_candidate_order_is_seeded_without_hidden_state
tests/integration/test_phase5_recovery.py::test_candidate_recovery_rejects_invalid_orders[candidate_order0]
tests/integration/test_phase5_recovery.py::test_candidate_recovery_rejects_invalid_orders[candidate_order1]
tests/integration/test_phase5_recovery.py::test_candidate_recovery_rejects_invalid_orders[candidate_order2]
tests/integration/test_phase5_recovery.py::test_candidate_recovery_rejects_invalid_orders[candidate_order3]
tests/integration/test_sat_model.py::test_fixed_key_model_states_match_the_python_reference
tests/integration/test_sat_model.py::test_final_round_subkey_assumptions_follow_the_key_schedule
tests/integration/test_sat_model.py::test_final_round_assumptions_are_temporary
tests/integration/test_sat_model.py::test_final_round_subkey_assumption_rejects_invalid_values[True]
tests/integration/test_sat_model.py::test_final_round_subkey_assumption_rejects_invalid_values[False]
tests/integration/test_sat_model.py::test_final_round_subkey_assumption_rejects_invalid_values[-1]
tests/integration/test_sat_model.py::test_final_round_subkey_assumption_rejects_invalid_values[65536]
tests/integration/test_sat_model.py::test_final_round_subkey_assumption_rejects_invalid_values[1.0]
tests/integration/test_sat_model.py::test_unknown_key_recovery_produces_a_verifiable_candidate
tests/integration/test_sat_model.py::test_modified_ciphertext_is_unsatisfiable_under_the_true_key
tests/integration/test_sat_model.py::test_blocking_clause_exposes_multiple_solutions_when_key_is_underdetermined
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[0-3-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[0-4-3-5]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[0-6-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[0-6-63-7]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[1-3-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[1-4-3-5]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[1-6-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[1-6-63-7]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[3-3-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[3-4-3-5]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[3-6-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[3-6-63-7]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[6-3-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[6-4-3-5]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[6-6-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[6-6-63-7]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[7-3-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[7-4-3-5]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[7-6-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_complete_toy_distribution_matches_independent_python_oracle[7-6-63-7]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_full_width_rejects_unbounded_dense_round
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_record_cap_fails_without_silent_truncation
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_rejects_zero_difference_and_existing_output
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_rejects_malformed_word_width[16junk]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_rejects_malformed_word_width[-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_rejects_malformed_word_width[4294967312]
tests/integration/test_simon_predicted_ddt_cpp.py::test_cpp_full_width_three_rounds_matches_source_style_transition
tests/integration/test_simon_predicted_ddt_cpp.py::test_dense_seven_round_table_matches_independent_complete_oracle[4-3-5]
tests/integration/test_simon_predicted_ddt_cpp.py::test_dense_seven_round_table_matches_independent_complete_oracle[6-0-1]
tests/integration/test_simon_predicted_ddt_cpp.py::test_dense_seven_round_table_matches_independent_complete_oracle[6-63-7]
tests/integration/test_simon_predicted_ddt_cpp.py::test_dense_generator_rejects_corrupted_sparse_input
tests/integration/test_simon_published_vv.py::test_published_vv_loader_rejects_unapproved_round
tests/integration/test_simon_published_vv.py::test_safe_checkpoint_load_matches_all_original_weights_and_clip_formula[7]
tests/integration/test_simon_published_vv.py::test_safe_checkpoint_load_matches_all_original_weights_and_clip_formula[8]
tests/integration/test_simon_published_vv.py::test_safe_checkpoint_load_matches_all_original_weights_and_clip_formula[9]
tests/integration/test_simon_published_vv.py::test_published_vv_loader_rejects_tampered_file
tests/integration/test_tiny_dataset_fixture.py::test_tiny_fixture_matches_the_generator

59 tests collected in 3.40s

Exit code: 0
Elapsed seconds: 4.438

```
