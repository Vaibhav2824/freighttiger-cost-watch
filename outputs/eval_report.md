# Evaluation against the hand-labelled sets

| check | result | pass |
|---|---|---|
| Note reading, dataset notes (raises cost + weeks covered) | 10/10 | yes |
| Note reading, adversarial probe set (informed the fixes; accuracy only) | 9/9 | yes |
| Note reading, adversarial held-out set (target >= 90%) | 11/11 | yes |
| UNSAFE misreads on adversarial notes (false raise or window too wide) | 0 | yes |
| Flagged route-weeks match gold set | missing 0, extra 0 | yes |
| Verdict + matched_note_id match gold | 20/20 | yes |
| Fabricated justifications in output | 0 | yes |
| Retrieval rank of the gold note (justified rows) | #1, #1, #1 | yes |
| Reasons that passed the grounding validator (rest used the template) | 20/20 | yes |

**Overall: PASS**
