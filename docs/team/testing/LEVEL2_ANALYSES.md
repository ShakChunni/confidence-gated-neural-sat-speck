# Level 2 rebuilt analyses

Run: 31 October 2026, Windows 11, Python 3.12.14. Source commit: 78c1bd27257a364bf822d105a298085c7b963c90 (fresh GitHub clone, hamza/testing). Existing source files were not edited.

| Item | Compared with | Match | Difference / method |
|---|---|---|---|
| D063 | `docs/paper/evidence/d063_campaign_20261011/summary.json` | Yes | Exact parsed JSON equality; no tolerance |
| D064 | `docs/paper/evidence/d064_campaign_20261011/summary.json` | Yes | Exact parsed JSON equality; no tolerance |
| D065 | `docs/paper/evidence/d065_campaign_20261012/summary.json` | Yes | Exact parsed JSON equality; no tolerance |

- Figure profile `report`: exit 0; built-in input hash and count checks passed; 7 figure entries. Output saved outside the source tree.
- Figure profile `lncs`: exit 0; built-in input hash and count checks passed; 7 figure entries. Output saved outside the source tree.

## Exact commands and outputs

The analysis JSON files accompany this note. Figure files are not included in the submission.

```text
COMMAND: C:/Users/3TEE/Documents/Codex/2026-10-30/so/work/hamza-venv/Scripts/python.exe scripts/analyze_d063.py docs/paper/evidence/d063_campaign_20261011/ledger_k00-31.jsonl
{
 "n_keys": 32,
 "arms": {
  "R7-B1": {
   "recovered": 19,
   "wilson95": [
    0.423,
    0.745
   ],
   "predicted": {
    "point": 21,
    "range": [
     15,
     27
    ],
    "verdict": "inside"
   },
   "par2_mean_s": 323.6,
   "median_total_s_recovered": 129.2,
   "outcomes": {
    "deadline": 13,
    "recovered": 19
   },
   "stage1_in_top32": 26,
   "stage1_top1": 2,
   "stage1_top8": 8,
   "stage1_median_rank_when_in_top32": 11.0,
   "true_chain_in_queue": 26,
   "in_queue_not_recovered": 7,
   "median_stage1_s": 43.7,
   "median_prep_s": 60.1,
   "median_wrong_candidate_s": 1.548,
   "verification_rejections": 0
  },
  "R7-G": {
   "recovered": 32,
   "wilson95": [
    0.893,
    1.0
   ],
   "predicted": {
    "point": 31,
    "range": [
     29,
     32
    ],
    "verdict": "inside"
   },
   "par2_mean_s": 22.8,
   "median_total_s_recovered": 21.4,
   "outcomes": {
    "recovered": 32
   },
   "stage1_in_top32": 32,
   "stage1_top1": 26,
   "stage1_top8": 32,
   "stage1_median_rank_when_in_top32": 1.0,
   "true_chain_in_queue": 32,
   "in_queue_not_recovered": 0,
   "median_stage1_s": 2.43,
   "median_prep_s": 18.8,
   "median_wrong_candidate_s": 1.588,
   "verification_rejections": 0
  },
  "R7-C": {
   "recovered": 30,
   "wilson95": [
    0.799,
    0.983
   ],
   "predicted": {
    "point": 30,
    "range": [
     27,
     32
    ],
    "verdict": "inside"
   },
   "par2_mean_s": 74.3,
   "median_total_s_recovered": 24.2,
   "outcomes": {
    "deadline": 2,
    "recovered": 30
   },
   "stage1_in_top32": 31,
   "stage1_top1": 8,
   "stage1_top8": 29,
   "stage1_median_rank_when_in_top32": 2,
   "true_chain_in_queue": 31,
   "in_queue_not_recovered": 1,
   "median_stage1_s": 2.44,
   "median_prep_s": 18.8,
   "median_wrong_candidate_s": 1.532,
   "verification_rejections": 0
  },
  "R8-G": {
   "recovered": 26,
   "wilson95": [
    0.647,
    0.911
   ],
   "predicted": {
    "point": 23,
    "range": [
     16,
     28
    ],
    "verdict": "inside"
   },
   "par2_mean_s": 247.7,
   "median_total_s_recovered": 156.8,
   "outcomes": {
    "queue_exhausted": 6,
    "recovered": 26
   },
   "stage1_in_top32": 30,
   "stage1_top1": 11,
   "stage1_top8": 25,
   "stage1_median_rank_when_in_top32": 2.0,
   "true_chain_in_queue": 26,
   "in_queue_not_recovered": 0,
   "median_stage1_s": 2.42,
   "median_prep_s": 150.3,
   "median_wrong_candidate_s": 0.255,
   "verification_rejections": 0
  },
  "R8-C": {
   "recovered": 0,
   "wilson95": [
    0.0,
    0.107
   ],
   "predicted": {
    "point": 0,
    "range": [
     0,
     2
    ],
    "verdict": "inside"
   },
   "par2_mean_s": 600.0,
   "median_total_s_recovered": null,
   "outcomes": {
    "queue_exhausted": 32
   },
   "stage1_in_top32": 0,
   "stage1_top1": 0,
   "stage1_top8": 0,
   "stage1_median_rank_when_in_top32": null,
   "true_chain_in_queue": 0,
   "in_queue_not_recovered": 0,
   "median_stage1_s": 0.5,
   "median_prep_s": 147.9,
   "median_wrong_candidate_s": 0.238,
   "verification_rejections": 0
  }
 },
 "Q1_primary": {
  "x": "R8-G",
  "y": "R8-C",
  "x_only": 26,
  "y_only": 0,
  "p_exact_two_sided": 2.9802322387695312e-08,
  "verdict": "pass"
 },
 "Q2_secondary": {
  "x": "R7-G",
  "y": "R7-C",
  "x_only": 2,
  "y_only": 0,
  "p_exact_two_sided": 0.5,
  "noninferior_R7G_ge_R7C_minus_2": true
 },
 "Q3_holm": [
  {
   "x": "R7-G",
   "y": "R7-B1",
   "x_only": 13,
   "y_only": 0,
   "p_exact_two_sided": 0.000244140625,
   "p_holm": 0.00048828125
  },
  {
   "x": "R7-C",
   "y": "R7-B1",
   "x_only": 12,
   "y_only": 1,
   "p_exact_two_sided": 0.00341796875,
   "p_holm": 0.00341796875
  }
 ],
 "par2_sign_tests": [
  {
   "x": "R8-G",
   "y": "R8-C",
   "x_cheaper": 26,
   "y_cheaper": 0,
   "ties": 6,
   "p_exact_two_sided": 2.9802322387695312e-08
  },
  {
   "x": "R7-G",
   "y": "R7-C",
   "x_cheaper": 26,
   "y_cheaper": 6,
   "ties": 0,
   "p_exact_two_sided": 0.0005350527353584766
  },
  {
   "x": "R7-G",
   "y": "R7-B1",
   "x_cheaper": 32,
   "y_cheaper": 0,
   "ties": 0,
   "p_exact_two_sided": 4.656612873077393e-10
  },
  {
   "x": "R7-C",
   "y": "R7-B1",
   "x_cheaper": 29,
   "y_cheaper": 2,
   "ties": 1,
   "p_exact_two_sided": 4.628673195838928e-07
  }
 ],
 "per_key": {
  "0": {
   "R7-B1": [
    "recovered",
    15,
    233.4
   ],
   "R7-G": [
    "recovered",
    1,
    20.8
   ],
   "R7-C": [
    "recovered",
    2,
    22.6
   ],
   "R8-G": [
    "recovered",
    1,
    153.9
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    278.4
   ]
  },
  "1": {
   "R7-B1": [
    "recovered",
    10,
    168.7
   ],
   "R7-G": [
    "recovered",
    2,
    31.1
   ],
   "R7-C": [
    "recovered",
    4,
    47.6
   ],
   "R8-G": [
    "queue_exhausted",
    2,
    280.9
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    280.2
   ]
  },
  "2": {
   "R7-B1": [
    "deadline",
    null,
    300.0
   ],
   "R7-G": [
    "recovered",
    1,
    20.3
   ],
   "R7-C": [
    "recovered",
    1,
    19.9
   ],
   "R8-G": [
    "recovered",
    4,
    154.6
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    278.7
   ]
  },
  "3": {
   "R7-B1": [
    "recovered",
    1,
    62.2
   ],
   "R7-G": [
    "recovered",
    1,
    21.8
   ],
   "R7-C": [
    "recovered",
    4,
    28.8
   ],
   "R8-G": [
    "recovered",
    3,
    158.8
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    280.3
   ]
  },
  "4": {
   "R7-B1": [
    "recovered",
    11,
    168.4
   ],
   "R7-G": [
    "recovered",
    1,
    23.6
   ],
   "R7-C": [
    "recovered",
    1,
    23.6
   ],
   "R8-G": [
    "recovered",
    1,
    151.0
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    277.1
   ]
  },
  "5": {
   "R7-B1": [
    "recovered",
    16,
    255.9
   ],
   "R7-G": [
    "recovered",
    1,
    20.5
   ],
   "R7-C": [
    "recovered",
    5,
    36.2
   ],
   "R8-G": [
    "recovered",
    2,
    157.8
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    270.8
   ]
  },
  "6": {
   "R7-B1": [
    "deadline",
    32,
    300.0
   ],
   "R7-G": [
    "recovered",
    1,
    21.1
   ],
   "R7-C": [
    "recovered",
    1,
    20.8
   ],
   "R8-G": [
    "recovered",
    1,
    152.8
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    286.1
   ]
  },
  "7": {
   "R7-B1": [
    "deadline",
    25,
    300.0
   ],
   "R7-G": [
    "recovered",
    1,
    21.6
   ],
   "R7-C": [
    "recovered",
    4,
    28.8
   ],
   "R8-G": [
    "recovered",
    12,
    196.8
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    280.5
   ]
  },
  "8": {
   "R7-B1": [
    "deadline",
    null,
    300.0
   ],
   "R7-G": [
    "recovered",
    1,
    23.9
   ],
   "R7-C": [
    "deadline",
    21,
    300.0
   ],
   "R8-G": [
    "queue_exhausted",
    2,
    292.3
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    281.0
   ]
  },
  "9": {
   "R7-B1": [
    "recovered",
    10,
    154.4
   ],
   "R7-G": [
    "recovered",
    2,
    27.5
   ],
   "R7-C": [
    "recovered",
    2,
    28.0
   ],
   "R8-G": [
    "recovered",
    1,
    153.1
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    280.7
   ]
  },
  "10": {
   "R7-B1": [
    "recovered",
    10,
    127.1
   ],
   "R7-G": [
    "recovered",
    1,
    20.5
   ],
   "R7-C": [
    "deadline",
    null,
    300.0
   ],
   "R8-G": [
    "queue_exhausted",
    null,
    274.6
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    274.9
   ]
  },
  "11": {
   "R7-B1": [
    "recovered",
    6,
    113.2
   ],
   "R7-G": [
    "recovered",
    2,
    29.6
   ],
   "R7-C": [
    "recovered",
    4,
    47.9
   ],
   "R8-G": [
    "queue_exhausted",
    3,
    276.7
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    274.2
   ]
  },
  "12": {
   "R7-B1": [
    "recovered",
    13,
    174.8
   ],
   "R7-G": [
    "recovered",
    1,
    20.6
   ],
   "R7-C": [
    "recovered",
    1,
    20.6
   ],
   "R8-G": [
    "recovered",
    8,
    165.9
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    282.9
   ]
  },
  "13": {
   "R7-B1": [
    "deadline",
    24,
    300.0
   ],
   "R7-G": [
    "recovered",
    2,
    26.6
   ],
   "R7-C": [
    "recovered",
    1,
    24.1
   ],
   "R8-G": [
    "recovered",
    9,
    179.4
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    286.1
   ]
  },
  "14": {
   "R7-B1": [
    "deadline",
    null,
    300.0
   ],
   "R7-G": [
    "recovered",
    1,
    24.0
   ],
   "R7-C": [
    "recovered",
    2,
    26.7
   ],
   "R8-G": [
    "recovered",
    2,
    153.9
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    278.4
   ]
  },
  "15": {
   "R7-B1": [
    "recovered",
    2,
    69.0
   ],
   "R7-G": [
    "recovered",
    1,
    23.6
   ],
   "R7-C": [
    "recovered",
    2,
    26.7
   ],
   "R8-G": [
    "recovered",
    1,
    152.5
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    277.0
   ]
  },
  "16": {
   "R7-B1": [
    "recovered",
    12,
    178.9
   ],
   "R7-G": [
    "recovered",
    1,
    24.5
   ],
   "R7-C": [
    "recovered",
    1,
    24.4
   ],
   "R8-G": [
    "recovered",
    1,
    152.1
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    286.3
   ]
  },
  "17": {
   "R7-B1": [
    "deadline",
    null,
    300.0
   ],
   "R7-G": [
    "recovered",
    1,
    19.8
   ],
   "R7-C": [
    "recovered",
    1,
    19.7
   ],
   "R8-G": [
    "recovered",
    9,
    168.6
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    275.5
   ]
  },
  "18": {
   "R7-B1": [
    "recovered",
    1,
    64.8
   ],
   "R7-G": [
    "recovered",
    1,
    22.3
   ],
   "R7-C": [
    "recovered",
    3,
    30.8
   ],
   "R8-G": [
    "recovered",
    1,
    154.0
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    285.9
   ]
  },
  "19": {
   "R7-B1": [
    "deadline",
    20,
    300.0
   ],
   "R7-G": [
    "recovered",
    1,
    19.2
   ],
   "R7-C": [
    "recovered",
    2,
    20.7
   ],
   "R8-G": [
    "recovered",
    3,
    155.9
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    276.7
   ]
  },
  "20": {
   "R7-B1": [
    "recovered",
    3,
    68.7
   ],
   "R7-G": [
    "recovered",
    1,
    20.7
   ],
   "R7-C": [
    "recovered",
    2,
    22.3
   ],
   "R8-G": [
    "recovered",
    7,
    169.3
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    284.9
   ]
  },
  "21": {
   "R7-B1": [
    "recovered",
    8,
    98.8
   ],
   "R7-G": [
    "recovered",
    1,
    20.8
   ],
   "R7-C": [
    "recovered",
    3,
    26.3
   ],
   "R8-G": [
    "queue_exhausted",
    null,
    279.0
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    276.3
   ]
  },
  "22": {
   "R7-B1": [
    "deadline",
    null,
    300.0
   ],
   "R7-G": [
    "recovered",
    1,
    24.4
   ],
   "R7-C": [
    "recovered",
    8,
    85.0
   ],
   "R8-G": [
    "recovered",
    1,
    175.3
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    278.9
   ]
  },
  "23": {
   "R7-B1": [
    "deadline",
    22,
    300.0
   ],
   "R7-G": [
    "recovered",
    1,
    20.4
   ],
   "R7-C": [
    "recovered",
    2,
    21.6
   ],
   "R8-G": [
    "recovered",
    5,
    161.6
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    282.0
   ]
  },
  "24": {
   "R7-B1": [
    "recovered",
    11,
    141.4
   ],
   "R7-G": [
    "recovered",
    1,
    20.2
   ],
   "R7-C": [
    "recovered",
    2,
    21.9
   ],
   "R8-G": [
    "recovered",
    1,
    150.7
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    281.0
   ]
  },
  "25": {
   "R7-B1": [
    "recovered",
    10,
    129.2
   ],
   "R7-G": [
    "recovered",
    1,
    20.9
   ],
   "R7-C": [
    "recovered",
    6,
    42.6
   ],
   "R8-G": [
    "recovered",
    20,
    232.7
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    280.0
   ]
  },
  "26": {
   "R7-B1": [
    "deadline",
    null,
    300.0
   ],
   "R7-G": [
    "recovered",
    1,
    22.6
   ],
   "R7-C": [
    "recovered",
    8,
    77.6
   ],
   "R8-G": [
    "recovered",
    2,
    176.4
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    280.1
   ]
  },
  "27": {
   "R7-B1": [
    "recovered",
    7,
    87.6
   ],
   "R7-G": [
    "recovered",
    1,
    20.4
   ],
   "R7-C": [
    "recovered",
    2,
    22.2
   ],
   "R8-G": [
    "recovered",
    2,
    151.6
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    280.1
   ]
  },
  "28": {
   "R7-B1": [
    "recovered",
    6,
    88.9
   ],
   "R7-G": [
    "recovered",
    2,
    28.3
   ],
   "R7-C": [
    "recovered",
    1,
    24.1
   ],
   "R8-G": [
    "recovered",
    25,
    233.7
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    280.2
   ]
  },
  "29": {
   "R7-B1": [
    "recovered",
    11,
    168.3
   ],
   "R7-G": [
    "recovered",
    2,
    26.9
   ],
   "R7-C": [
    "recovered",
    19,
    292.6
   ],
   "R8-G": [
    "queue_exhausted",
    1,
    282.1
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    284.2
   ]
  },
  "30": {
   "R7-B1": [
    "deadline",
    30,
    300.0
   ],
   "R7-G": [
    "recovered",
    1,
    20.5
   ],
   "R7-C": [
    "recovered",
    2,
    21.4
   ],
   "R8-G": [
    "recovered",
    6,
    162.0
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    280.3
   ]
  },
  "31": {
   "R7-B1": [
    "deadline",
    23,
    300.0
   ],
   "R7-G": [
    "recovered",
    1,
    20.2
   ],
   "R7-C": [
    "recovered",
    3,
    22.7
   ],
   "R8-G": [
    "recovered",
    1,
    152.2
   ],
   "R8-C": [
    "queue_exhausted",
    null,
    280.5
   ]
  }
 }
}

Exit code: 0
Elapsed seconds: 0.125

```

```text
COMMAND: C:/Users/3TEE/Documents/Codex/2026-10-30/so/work/hamza-venv/Scripts/python.exe scripts/analyze_d064.py docs/paper/evidence/d064_campaign_20261011/ledger_k00-31.jsonl
{
 "ledger_header": {
  "seed": 701002,
  "git_commit": "f86cf6766593f8e1392a9269cb30a6d4c71f14b3",
  "runner_dirty": false,
  "script_sha256": "2797f6209cb63cbe15d902ca21ff1f0cd83955342aff9af7b5eb4b54bde3f4d1",
  "started_local": "2026-10-11 16:55:58 AWST"
 },
 "arms": {
  "R8-G": {
   "recovered": 24,
   "wilson95": [
    0.5789,
    0.8675
   ],
   "outcomes": {
    "queue_exhausted": 8,
    "recovered": 24
   },
   "stage1_top1": 8,
   "stage1_top8": 22,
   "stage1_top32": 30,
   "stage1_median_position": 3.0,
   "stage1_max_position": 127,
   "true_chain_queued": 24,
   "recovered_when_queued": 24,
   "median_total_s_recovered": 160.2,
   "max_total_s": 288.7,
   "mean_par2_s": 279.2,
   "median_stage1_s": 2.66,
   "median_prep_s": 149.2,
   "median_wrong_median_s": 0.233,
   "mean_wrong_mean_s": 0.25,
   "verification_rejections": 0
  },
  "R8-F": {
   "recovered": 18,
   "wilson95": [
    0.3933,
    0.7183
   ],
   "outcomes": {
    "queue_exhausted": 14,
    "recovered": 18
   },
   "stage1_top1": 4,
   "stage1_top8": 14,
   "stage1_top32": 22,
   "stage1_median_position": 9.5,
   "stage1_max_position": 2872,
   "true_chain_queued": 18,
   "recovered_when_queued": 18,
   "median_total_s_recovered": 161.0,
   "max_total_s": 285.9,
   "mean_par2_s": 359.5,
   "median_stage1_s": 1.33,
   "median_prep_s": 148.2,
   "median_wrong_median_s": 0.24,
   "mean_wrong_mean_s": 0.245,
   "verification_rejections": 0,
   "stage1_load_s_median_max": [
    0.99,
    1.6
   ]
  }
 },
 "tests": {
  "Q1_stage1_sign": {
   "net_better": 19,
   "table_better": 10,
   "ties": 3,
   "p": 0.13604594767093658,
   "verdict": "no detectable difference"
  },
  "Q2_recovery_mcnemar": {
   "net_only": 6,
   "table_only": 0,
   "both": 18,
   "neither": 8,
   "p": 0.03125,
   "p_holm": 0.0625
  },
  "Q3_par2_sign": {
   "net_cheaper": 10,
   "table_cheaper": 14,
   "ties": 8,
   "p": 0.5412561893463135,
   "p_holm": 0.5412561893463135
  }
 },
 "per_key": [
  {
   "key_index": 0,
   "net_stage1_position": 1,
   "table_stage1_position": 32,
   "net_outcome": "recovered",
   "table_outcome": "recovered",
   "net_queue_position": 2,
   "table_queue_position": 459
  },
  {
   "key_index": 1,
   "net_stage1_position": 8,
   "table_stage1_position": 1,
   "net_outcome": "recovered",
   "table_outcome": "recovered",
   "net_queue_position": 158,
   "table_queue_position": 48
  },
  {
   "key_index": 2,
   "net_stage1_position": 8,
   "table_stage1_position": 6,
   "net_outcome": "recovered",
   "table_outcome": "recovered",
   "net_queue_position": 92,
   "table_queue_position": 62
  },
  {
   "key_index": 3,
   "net_stage1_position": 24,
   "table_stage1_position": 194,
   "net_outcome": "recovered",
   "table_outcome": "queue_exhausted",
   "net_queue_position": 367,
   "table_queue_position": null
  },
  {
   "key_index": 4,
   "net_stage1_position": 2,
   "table_stage1_position": 5,
   "net_outcome": "recovered",
   "table_outcome": "recovered",
   "net_queue_position": 8,
   "table_queue_position": 35
  },
  {
   "key_index": 5,
   "net_stage1_position": 3,
   "table_stage1_position": 17,
   "net_outcome": "recovered",
   "table_outcome": "recovered",
   "net_queue_position": 51,
   "table_queue_position": 276
  },
  {
   "key_index": 6,
   "net_stage1_position": 27,
   "table_stage1_position": 16,
   "net_outcome": "recovered",
   "table_outcome": "recovered",
   "net_queue_position": 389,
   "table_queue_position": 215
  },
  {
   "key_index": 7,
   "net_stage1_position": 127,
   "table_stage1_position": 2846,
   "net_outcome": "queue_exhausted",
   "table_outcome": "queue_exhausted",
   "net_queue_position": null,
   "table_queue_position": null
  },
  {
   "key_index": 8,
   "net_stage1_position": 2,
   "table_stage1_position": 2,
   "net_outcome": "recovered",
   "table_outcome": "recovered",
   "net_queue_position": 6,
   "table_queue_position": 5
  },
  {
   "key_index": 9,
   "net_stage1_position": 13,
   "table_stage1_position": 9,
   "net_outcome": "queue_exhausted",
   "table_outcome": "queue_exhausted",
   "net_queue_position": null,
   "table_queue_position": null
  },
  {
   "key_index": 10,
   "net_stage1_position": 11,
   "table_stage1_position": 8,
   "net_outcome": "recovered",
   "table_outcome": "recovered",
   "net_queue_position": 130,
   "table_queue_position": 85
  },
  {
   "key_index": 11,
   "net_stage1_position": 3,
   "table_stage1_position": 38,
   "net_outcome": "recovered",
   "table_outcome": "queue_exhausted",
   "net_queue_position": 48,
   "table_queue_position": null
  },
  {
   "key_index": 12,
   "net_stage1_position": 2,
   "table_stage1_position": 1,
   "net_outcome": "recovered",
   "table_outcome": "recovered",
   "net_queue_position": 20,
   "table_queue_position": 12
  },
  {
   "key_index": 13,
   "net_stage1_position": 3,
   "table_stage1_position": 64,
   "net_outcome": "queue_exhausted",
   "table_outcome": "queue_exhausted",
   "net_queue_position": null,
   "table_queue_position": null
  },
  {
   "key_index": 14,
   "net_stage1_position": 1,
   "table_stage1_position": 2,
   "net_outcome": "recovered",
   "table_outcome": "recovered",
   "net_queue_position": 2,
   "table_queue_position": 6
  },
  {
   "key_index": 15,
   "net_stage1_position": 3,
   "table_stage1_position": 2,
   "net_outcome": "recovered",
   "table_outcome": "recovered",
   "net_queue_position": 20,
   "table_queue_position": 13
  },
  {
   "key_index": 16,
   "net_stage1_position": 2,
   "table_stage1_position": 16,
   "net_outcome": "queue_exhausted",
   "table_outcome": "queue_exhausted",
   "net_queue_position": null,
   "table_queue_position": null
  },
  {
   "key_index": 17,
   "net_stage1_position": 1,
   "table_stage1_position": 6,
   "net_outcome": "queue_exhausted",
   "table_outcome": "queue_exhausted",
   "net_queue_position": null,
   "table_queue_position": null
  },
  {
   "key_index": 18,
   "net_stage1_position": 13,
   "table_stage1_position": 9,
   "net_outcome": "recovered",
   "table_outcome": "recovered",
   "net_queue_position": 139,
   "table_queue_position": 76
  },
  {
   "key_index": 19,
   "net_stage1_position": 1,
   "table_stage1_position": 1,
   "net_outcome": "recovered",
   "table_outcome": "recovered",
   "net_queue_position": 9,
   "table_queue_position": 11
  },
  {
   "key_index": 20,
   "net_stage1_position": 13,
   "table_stage1_position": 10,
   "net_outcome": "recovered",
   "table_outcome": "recovered",
   "net_queue_position": 151,
   "table_queue_position": 106
  },
  {
   "key_index": 21,
   "net_stage1_position": 7,
   "table_stage1_position": 17,
   "net_outcome": "recovered",
   "table_outcome": "recovered",
   "net_queue_position": 78,
   "table_queue_position": 238
  },
  {
   "key_index": 22,
   "net_stage1_position": 2,
   "table_stage1_position": 54,
   "net_outcome": "recovered",
   "table_outcome": "queue_exhausted",
   "net_queue_position": 8,
   "table_queue_position": null
  },
  {
   "key_index": 23,
   "net_stage1_position": 13,
   "table_stage1_position": 89,
   "net_outcome": "recovered",
   "table_outcome": "queue_exhausted",
   "net_queue_position": 217,
   "table_queue_position": null
  },
  {
   "key_index": 24,
   "net_stage1_position": 1,
   "table_stage1_position": 40,
   "net_outcome": "recovered",
   "table_outcome": "queue_exhausted",
   "net_queue_position": 15,
   "table_queue_position": null
  },
  {
   "key_index": 25,
   "net_stage1_position": 10,
   "table_stage1_position": 3,
   "net_outcome": "recovered",
   "table_outcome": "recovered",
   "net_queue_position": 126,
   "table_queue_position": 24
  },
  {
   "key_index": 26,
   "net_stage1_position": 1,
   "table_stage1_position": 634,
   "net_outcome": "queue_exhausted",
   "table_outcome": "queue_exhausted",
   "net_queue_position": null,
   "table_queue_position": null
  },
  {
   "key_index": 27,
   "net_stage1_position": 7,
   "table_stage1_position": 2872,
   "net_outcome": "recovered",
   "table_outcome": "queue_exhausted",
   "net_queue_position": 42,
   "table_queue_position": null
  },
  {
   "key_index": 28,
   "net_stage1_position": 43,
   "table_stage1_position": 187,
   "net_outcome": "queue_exhausted",
   "table_outcome": "queue_exhausted",
   "net_queue_position": null,
   "table_queue_position": null
  },
  {
   "key_index": 29,
   "net_stage1_position": 1,
   "table_stage1_position": 2,
   "net_outcome": "recovered",
   "table_outcome": "recovered",
   "net_queue_position": 0,
   "table_queue_position": 3
  },
  {
   "key_index": 30,
   "net_stage1_position": 4,
   "table_stage1_position": 8,
   "net_outcome": "queue_exhausted",
   "table_outcome": "queue_exhausted",
   "net_queue_position": null,
   "table_queue_position": null
  },
  {
   "key_index": 31,
   "net_stage1_position": 1,
   "table_stage1_position": 1,
   "net_outcome": "recovered",
   "table_outcome": "recovered",
   "net_queue_position": 8,
   "table_queue_position": 8
  }
 ]
}

Exit code: 0
Elapsed seconds: 0.125

```

```text
COMMAND: C:/Users/3TEE/Documents/Codex/2026-10-30/so/work/hamza-venv/Scripts/python.exe scripts/analyze_d065.py docs/paper/evidence/d065_predictions/predictions.json C:\Users\3TEE\Documents\Codex\2026-10-31\pl\work\repository\docs\team\testing\level2_d065_20261031.json docs/paper/evidence/d065_campaign_20261012/ledger_k00-31.jsonl
{
 "H1_primary": {
  "keys_sat_ahead": 28,
  "keys_bf_ahead": 0,
  "keys_tied": 4,
  "p": 3.725290298461914e-09,
  "pass": true,
  "per_ranker": {
   "R7-B1": {
    "sat_only": 23,
    "bf_only": 0,
    "p": 1.1920928955078125e-07,
    "p_holm": 3.5762786865234375e-07
   },
   "R7-G": {
    "sat_only": 6,
    "bf_only": 0,
    "p": 0.015625,
    "p_holm": 0.015625
   },
   "R7-C": {
    "sat_only": 16,
    "bf_only": 0,
    "p": 1.52587890625e-05,
    "p_holm": 3.0517578125e-05
   }
  }
 },
 "H2_coprimary": {
  "sat_recovered": 18,
  "bf_recovered": 18,
  "a_noninferior": true,
  "bf_cheaper": 17,
  "bf_dearer": 1,
  "ties": 14,
  "p_b": 7.2479248046875e-05,
  "b_pass": true,
  "discordant_recovery": {
   "sat_only": 0,
   "bf_only": 0
  },
  "pass": true
 },
 "H3_calibration": {
  "inside": 5,
  "of": 8,
  "pass": false
 },
 "H4_loss_order": {
  "loss": {
   "R7-B1": 23,
   "R7-G": 6,
   "R7-C": 16
  },
  "B1_ge_table_ge_net6": true
 },
 "verdict": "rule wins at both depths"
}
R7-B1 26 [14.0, 24.0] above PAR-2 216.7
R7-B1-BF 3 [0.0, 4.0] inside PAR-2 563.2
R7-G 32 [32.0, 32.0] inside PAR-2 24.2
R7-G-BF 26 [22.0, 29.0] inside PAR-2 208.5
R7-C 29 [26.0, 31.0] inside PAR-2 98.3
R7-C-BF 13 [12.0, 21.0] inside PAR-2 408.9
R8-G 18 [22.0, 29.0] below PAR-2 356.0
R8-G-BF 18 [22.0, 29.0] below PAR-2 345.8

Exit code: 0
Elapsed seconds: 0.125

```

```text
COMMAND: C:/Users/3TEE/Documents/Codex/2026-10-30/so/work/hamza-venv/Scripts/python.exe scripts/generate_report_figures.py --profile report --force --output-root C:\Users\3TEE\Documents\Codex\2026-10-31\pl\work\final_figcheck_report
Could not save font_manager cache [Errno 13] Permission denied: 'C:\\Users\\3TEE\\.matplotlib\\fontlist-v3.11.0.json.matplotlib-lock'
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
fig_pipeline: 6.0 x 2.74 in
fig_recovery_main: 6.0 x 2.95 in
fig_recovery_d063: 6.0 x 2.75 in
fig_recovery_d063_d066: 6.0 x 2.75 in
fig_second_stage: 6.0 x 2.55 in
fig_affordable_depth: 4.6 x 3.312 in
fig_prospective_predictions: 4.6 x 2.852 in

Exit code: 0
Elapsed seconds: 5.547

```

```text
COMMAND: C:/Users/3TEE/Documents/Codex/2026-10-30/so/work/hamza-venv/Scripts/python.exe scripts/generate_report_figures.py --profile lncs --force --output-root C:\Users\3TEE\Documents\Codex\2026-10-31\pl\work\final_figcheck_lncs
Could not save font_manager cache [Errno 13] Permission denied: 'C:\\Users\\3TEE\\.matplotlib\\fontlist-v3.11.0.json.matplotlib-lock'
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
'created' timestamp seems very low; regarding as unix timestamp
'modified' timestamp seems very low; regarding as unix timestamp
fig_pipeline: 4.75 x 2.7 in
fig_recovery_main: 4.75 x 2.85 in
fig_recovery_d063: 4.75 x 2.65 in
fig_recovery_d063_d066: 4.75 x 2.65 in
fig_second_stage: 4.75 x 2.45 in
fig_affordable_depth: 4.1 x 2.952 in
fig_prospective_predictions: 4.1 x 2.542 in

Exit code: 0
Elapsed seconds: 5.282

```

