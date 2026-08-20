# P3 Results — Full Verdict Table

Generated from `data/p3_verdicts.json` (generated 2026-08-08T12:46:07Z, config
hash `9b8af804d62b0fec771fac4a2b7d86acc9f7fb27c639965b3fc98710482eedad`,
matching the frozen pre-registration in `data/prereg_freeze.json`). Run cost:
$20.01 across 23,008 API requests, 1,438 questions.

Verdict rule (frozen in `docs/prereg.md` before the run): CI entirely within
±2pp → **placebo**; CI excludes 0, positive → **still_works**; CI excludes 0,
negative → **actively_hurts**; otherwise → **inconclusive**. Holm-Bonferroni
across the 6 technique comparisons per model×task (family size 1 for the
adaptive-reasoning B-vs-T1 checks).

**Census: 20 placebo · 17 inconclusive · 2 actively_hurts · 0 still_works.**

Deltas and CIs are in percentage points (technique minus baseline, paired on
the same questions).

## Math (GSM8K, N=449 pairs)

| Model | Reasoning | Arm | Technique | Base % | Tech % | Δ (pp) | 95% CI (pp) | p (raw) | Holm sig. | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-4-5 | off | T1 | zero_shot_cot | 96.88 | 97.10 | +0.22 | [-0.89, +1.34] | 0.8732 | no | placebo |
| claude-haiku-4-5 | off | T2 | role_prompt | 96.88 | 97.10 | +0.22 | [-0.89, +1.34] | 0.8632 | no | placebo |
| claude-haiku-4-5 | off | T3 | emotional_stakes | 96.88 | 97.10 | +0.22 | [-0.67, +1.11] | 0.8370 | no | placebo |
| claude-haiku-4-5 | off | T4 | incentive | 96.88 | 97.55 | +0.67 | [-0.22, +1.78] | 0.2488 | no | placebo |
| claude-haiku-4-5 | off | T5 | politeness | 96.88 | 97.10 | +0.22 | [-0.89, +1.34] | 0.8566 | no | placebo |
| claude-haiku-4-5 | off | T6 | few_shot | 96.88 | 97.55 | +0.67 | [-0.45, +1.78] | 0.3384 | no | placebo |
| claude-sonnet-5 | off | T1 | zero_shot_cot | 97.33 | 97.77 | +0.45 | [-0.67, +1.56] | 0.5486 | no | placebo |
| claude-sonnet-5 | off | T2 | role_prompt | 97.33 | 97.77 | +0.45 | [-0.67, +1.56] | 0.5314 | no | placebo |
| claude-sonnet-5 | off | T3 | emotional_stakes | 97.33 | 96.88 | -0.45 | [-1.56, +0.67] | 0.5424 | no | placebo |
| claude-sonnet-5 | off | T4 | incentive | 97.33 | 98.00 | +0.67 | [0.00, +1.56] | 0.1014 | no | placebo |
| claude-sonnet-5 | off | T5 | politeness | 97.33 | 97.55 | +0.22 | [-0.67, +1.34] | 0.8234 | no | placebo |
| claude-sonnet-5 | off | T6 | few_shot | 97.33 | 98.22 | +0.89 | [0.00, +2.00] | 0.1226 | no | inconclusive |
| claude-sonnet-5 | adaptive | T1 | zero_shot_cot | 97.55 | 97.77 | +0.22 | [-0.67, +1.11] | 0.8360 | no | placebo |

## Logic (BBH date_understanding, N=250 pairs)

| Model | Reasoning | Arm | Technique | Base % | Tech % | Δ (pp) | 95% CI (pp) | p (raw) | Holm sig. | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-4-5 | off | T1 | zero_shot_cot | 93.20 | 92.40 | -0.80 | [-2.80, +1.20] | 0.5288 | no | inconclusive |
| claude-haiku-4-5 | off | T2 | role_prompt | 93.20 | 92.40 | -0.80 | [-2.40, +0.80] | 0.4600 | no | inconclusive |
| claude-haiku-4-5 | off | T3 | emotional_stakes | 93.20 | 93.20 | 0.00 | [-1.60, +1.60] | 1.0000 | no | placebo |
| claude-haiku-4-5 | off | T4 | incentive | 93.20 | 90.80 | -2.40 | [-4.80, 0.00] | 0.0656 | no | inconclusive |
| claude-haiku-4-5 | off | T5 | politeness | 93.20 | 90.80 | -2.40 | [-4.80, 0.00] | 0.0710 | no | inconclusive |
| claude-haiku-4-5 | off | T6 | few_shot | 93.20 | 95.20 | +2.00 | [-0.40, +4.40] | 0.1220 | no | inconclusive |
| claude-sonnet-5 | off | T1 | zero_shot_cot | 96.00 | 96.00 | 0.00 | [0.00, 0.00] | 1.0000 | no | placebo |
| claude-sonnet-5 | off | T2 | role_prompt | 96.00 | 95.60 | -0.40 | [-1.60, +0.80] | 0.7798 | no | placebo |
| claude-sonnet-5 | off | T3 | emotional_stakes | 96.00 | 95.60 | -0.40 | [-1.20, 0.00] | 0.7276 | no | placebo |
| claude-sonnet-5 | off | T4 | incentive | 96.00 | 95.20 | -0.80 | [-2.40, +0.80] | 0.4434 | no | inconclusive |
| claude-sonnet-5 | off | T5 | politeness | 96.00 | 96.00 | 0.00 | [-1.20, +1.20] | 1.0000 | no | placebo |
| claude-sonnet-5 | off | T6 | few_shot | 96.00 | 96.00 | 0.00 | [-1.60, +1.60] | 1.0000 | no | placebo |
| claude-sonnet-5 | adaptive | T1 | zero_shot_cot | 95.20 | 94.40 | -0.80 | [-2.00, 0.00] | 0.2740 | no | placebo |

## Procedural (seeded generator, N=739 pairs)

| Model | Reasoning | Arm | Technique | Base % | Tech % | Δ (pp) | 95% CI (pp) | p (raw) | Holm sig. | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-4-5 | off | T1 | zero_shot_cot | 48.71 | 48.17 | -0.54 | [-2.57, +1.35] | 0.6272 | no | inconclusive |
| claude-haiku-4-5 | off | T2 | role_prompt | 48.71 | 49.66 | +0.95 | [-1.22, +3.11] | 0.4272 | no | inconclusive |
| claude-haiku-4-5 | off | T3 | emotional_stakes | 48.71 | 50.34 | +1.62 | [-0.27, +3.52] | 0.1094 | no | inconclusive |
| claude-haiku-4-5 | off | T4 | incentive | 48.71 | 49.39 | +0.68 | [-1.35, +2.71] | 0.5512 | no | inconclusive |
| claude-haiku-4-5 | off | T5 | politeness | 48.71 | 48.04 | -0.68 | [-2.57, +1.22] | 0.5254 | no | inconclusive |
| claude-haiku-4-5 | off | T6 | few_shot | 48.71 | 47.23 | -1.49 | [-3.52, +0.41] | 0.1508 | no | inconclusive |
| claude-sonnet-5 | off | T1 | zero_shot_cot | 96.21 | 96.75 | +0.54 | [-1.08, +2.17] | 0.5718 | no | inconclusive |
| claude-sonnet-5 | off | T2 | role_prompt | 96.21 | 96.62 | +0.41 | [-1.22, +2.03] | 0.6938 | no | inconclusive |
| claude-sonnet-5 | off | T3 | emotional_stakes | 96.21 | 95.67 | -0.54 | [-2.17, +1.08] | 0.5772 | no | inconclusive |
| claude-sonnet-5 | off | T4 | incentive | 96.21 | 95.67 | -0.54 | [-2.30, +1.08] | 0.5856 | no | inconclusive |
| claude-sonnet-5 | off | T5 | politeness | 96.21 | 93.37 | **-2.84** | [-4.87, -0.81] | 0.0072 | **yes** | **actively_hurts** |
| claude-sonnet-5 | off | T6 | few_shot | 96.21 | 92.42 | **-3.79** | [-5.95, -1.62] | 0.0004 | **yes** | **actively_hurts** |
| claude-sonnet-5 | adaptive | T1 | zero_shot_cot | 99.32 | 99.05 | -0.27 | [-1.22, +0.68] | 0.6788 | no | placebo |

## Reading notes

- No comparison anywhere earned **still_works**. Every point estimate that
  leaned positive has a CI touching or crossing zero.
- The two **actively_hurts** results (politeness, few_shot; Sonnet 5, reasoning
  disabled, procedural) are the only Holm-significant effects in the grid
  (`holm_significant: true`); neither was downgraded by the correction
  (`downgraded_by_holm: false` on all 39 records).
- `n_pairs == n_expected` on all 39 records — no dropped pairs.
- The Sonnet-5 adaptive-reasoning row exists only for B vs. T1 (family size 1),
  per the frozen design.
