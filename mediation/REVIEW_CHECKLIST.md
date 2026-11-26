# REVIEW_CHECKLIST - Revision-R2g

| # | Check | Status | Detail |
|---|---|---|---|
| 1 | c paths reproduce Revision-R2d diet family (M1 and M0): beta, HC3 SE, HC3 p | PASS | max abs diff 0.0e+00 (tolerance 1e-8) |
| 2 | a, b, c' (beta, HC3 SE, p) of 20 random Family S pairs = statsmodels formula fit = manual numpy HC3 | PASS | max abs diff 4.9e-15 |
| 3 | bootstrap ab of 3 pairs reproduced by an independent full-design loop on the same draws | PASS | max abs diff ab 1.4e-13, CI 1.1e-14 |
| 4 | n = 88 (15 depressive), no missing input or result values | PASS | n per family {'P': [88], 'S': [88], 'S1': [88], 'S2': [88]} |
| 5 | z-scores have mean 0 and SD 1 (ddof = 1) | PASS | max |mean| 2.1e-15, max |SD-1| 4.4e-16 |
| 6 | 188 taxa with prevalence >= 18/88; input copies = originals; loaded inputs = main-run hashes | PASS | 188 taxa; 25 copies identical; 4 main-run hashes identical |
| 7 | BH q-values: statsmodels = manual implementation | PASS | 20 q columns, max diff 5.0e-14 |
| 9 | plan hash unchanged | PASS | 9a7179af96076554... |
| 8 | Protected folders unchanged (R2b, R2c, R2b-conf, R2c-conf, R2d, R2e, R2f) | PASS | 1221 files |

## Flags
- Exploratory: Family P taxa were prioritised using the outcome.
- No pair has a q-value supporting a mediation claim.

## Changes from the plan
- None in substance. The forest plot was redrawn with `--resume` (title clipped in the first draw); no number changed.
