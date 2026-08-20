# prompt-placebo

Which 2023-era prompt-engineering folklore (role prompts, "think step by step,"
emotional stakes, tips, politeness, few-shot) still causally improves accuracy on
2026 models that reason natively — and which is now placebo?

Paired-delta design: every technique is measured against an identical baseline on
the *same question*, so the comparison isolates the technique's effect from
question difficulty. Bootstrap confidence intervals, Holm-Bonferroni correction
across techniques, and a pre-registration hash-freeze (the stats and verdict rule
are committed and hashed before any full run) keep the result honest.

**Status: P3 complete.** The full pre-registered run has been executed under the
frozen config (hash `9b8af804d62b0fec771fac4a2b7d86acc9f7fb27c639965b3fc98710482eedad`,
matching `data/prereg_freeze.json`): 23,008 API requests, 1,438 questions,
**$20.01 total spend** (under the $25 cap). Raw results in `data/p3_results.jsonl`;
the 39 final verdict records in `data/p3_verdicts.json`; full table in
[`docs/results.md`](docs/results.md).

- **Headline: nothing "still works." 0 of 39 comparisons earned a still_works
  verdict.** 20 are placebo (CI entirely within ±2pp), 17 inconclusive, and 2
  actively hurt.
- **The only significant effects are harms.** On claude-sonnet-5 (reasoning
  disabled), procedural tasks: **politeness** drops accuracy 2.84pp (96.21% →
  93.37%, CI [-4.87, -0.81], p=0.0072) and **few-shot** drops it 3.79pp
  (96.21% → 92.42%, CI [-5.95, -1.62], p=0.0004). Both survive Holm-Bonferroni.
- **Math is pure placebo.** All 6 techniques on claude-haiku-4-5 and 5 of 6 on
  claude-sonnet-5 (reasoning off) are placebo; the exception (few-shot on
  Sonnet, +0.89pp, CI [0.00, +2.00]) is inconclusive, not a win. The adaptive-
  reasoning CoT check is also placebo.
- **Logic is placebo-or-inconclusive.** Sonnet (reasoning off): 5 placebo, 1
  inconclusive (incentive). Haiku: 1 placebo (emotional stakes, delta exactly
  0.00), 5 inconclusive — the largest point estimates there (incentive -2.4pp,
  politeness -2.4pp, few-shot +2.0pp) all have CIs crossing zero. Adaptive CoT:
  placebo.
- **Procedural is where the action is.** Haiku: all 6 inconclusive (baseline
  48.71% — the hardened generator bites). Sonnet (reasoning off): 4
  inconclusive plus the two significant harms above. Sonnet with adaptive
  reasoning: CoT is placebo at a 99.32% baseline.
- **P2 pilot (real, committed):** N=30 per task set, every arm, both models,
  ~$1.37 total API spend (`data/pilot_results.jsonl`,
  `data/procedural_check_results.jsonl`). The pilot exists to size the full
  run via power analysis — it is deliberately too small to decide any verdict,
  and none is claimed from it.
- **Pre-registration FROZEN before P3** (`data/prereg_freeze.json`, config hash
  above): N=449 (math), N=250 (logic — capped by BBH date_understanding's
  entire 250-row test split), N=739 (procedural — budget-capped from an ideal
  1377). 449 + 250 + 739 = 1,438 questions. The verdict rule was committed in
  `docs/prereg.md` before any full-run data existed and was not changed after.
- **One genuine boundary finding from the pilot:** the original procedural
  task hit 100% accuracy on every arm/model combo (ceiling effect, zero
  variance). The generator was hardened and re-checked — in P3 the
  Sonnet-with-reasoning procedural baseline landed at 99.32%, so the cell had
  (barely) enough variance to classify; its CoT comparison is placebo.

## Layout

- `prereg/` — frozen experimental design: technique arms, models, stats constants,
  hash-freeze gate
- `harness/` — prompt construction, answer extraction, cost-metered batch runner,
  procedural task generator
- `stats/` — paired bootstrap confidence intervals
- `docs/prereg.md` — the human-readable pre-registration
- `docs/results.md` — full P3 verdict table (all 39 comparisons)
