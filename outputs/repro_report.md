# Reproducibility check

3 untouched runs at the shipped settings {'temperature': 0, 'seed': 42}, each starting cold. Reference = run 1.

Reference fingerprint: `259570e2e17c`

| run | fingerprint | matches reference | reason wording identical | LLM calls | input tokens | output tokens |
|---|---|---|---|---|---|---|
| run 1 | `259570e2e17c` | yes | yes | 2 | 4307 | 2130 |
| run 2 | `259570e2e17c` | yes | no (wording only) | 2 | 4307 | 2070 |
| run 3 | `259570e2e17c` | yes | no (wording only) | 2 | 4307 | 2070 |

**Result: PASS**: flags, numbers, verdicts and cited notes are identical in every run.

The fingerprint is a SHA-256 of these columns: route, week_of, cost_per_tonne_km, vs_own_history, vs_similar_routes, flagged, matched_note_id. The free-text `reason` is excluded; the brief allows its wording to vary.
