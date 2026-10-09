# Queue-cost model evidence (D-059 analysis; D-060 frozen predictions)

Regenerate from the repository root (outputs are byte-identical):

    python3 -I scripts/extract_queue_cost_attempts.py --output-dir OUT
    python3 -I scripts/fit_queue_cost_model.py --input-dir OUT --output-dir OUT
    python3 -I scripts/queue_cost_posthoc.py --input-dir OUT
    python3 -I scripts/queue_cost_step5_predictions.py --input-dir OUT --output OUT/step5_predictions.json
    python3 -I scripts/queue_cost_simon_primary.py OUT results/simon_confirmation_replacement/simon_confirmation_comparators.json

Files: attempts.csv, cells.csv (extraction; validated 6,110 completed attempts);
predictions.csv, summary.json, fit_out.txt (protocol model); posthoc.txt
(labelled post hoc); step5_predictions.json (frozen Step 5 predictions);
step4/ (primary SIMON campaign, exploratory). Every file has a .sha256 sidecar.
Protocol SHA-256: 4cffa9b461edba379b004ec887e5af4c3c6559de4e975fbd2fc5b4a3f492dd19.
