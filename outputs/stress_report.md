# Temperature stress check

5 cold runs with sampling switched ON (temperature 0.8, seeds 1, 2, 3, 4, 5). Reference = the shipped temperature-0 `outputs/output.csv`.

Reference fingerprint: `259570e2e17c`

| run | fingerprint | matches reference | reason wording identical | LLM calls | input tokens | output tokens |
|---|---|---|---|---|---|---|
| T=0.8, seed 1 | `259570e2e17c` | yes | no (wording only) | 2 | 4307 | 2103 |
| T=0.8, seed 2 | `259570e2e17c` | yes | no (wording only) | 2 | 4307 | 2144 |
| T=0.8, seed 3 | `259570e2e17c` | yes | no (wording only) | 2 | 4307 | 2141 |
| T=0.8, seed 4 | `259570e2e17c` | yes | no (wording only) | 2 | 4307 | 2147 |
| T=0.8, seed 5 | `259570e2e17c` | yes | no (wording only) | 2 | 4307 | 2046 |

**Result: PASS**: flags, numbers, verdicts and cited notes are identical in every run.

The fingerprint is a SHA-256 of these columns: route, week_of, cost_per_tonne_km, vs_own_history, vs_similar_routes, flagged, matched_note_id. The free-text `reason` is excluded; the brief allows its wording to vary.
