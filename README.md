# prompt-placebo

Which 2023-era prompt-engineering folklore (role prompts, "think step by step,"
emotional stakes, tips, politeness, few-shot) still causally improves accuracy on
2026 models that reason natively — and which is now placebo?

Paired-delta design: every technique is measured against an identical baseline on
the *same question*, so the comparison isolates the technique's effect from
question difficulty. Bootstrap confidence intervals, Holm-Bonferroni correction
across techniques, and a pre-registration hash-freeze (the stats and verdict rule
are committed and hashed before any full run) keep the result honest.

**Status:** pilot done, pre-registration frozen, full run not yet executed. **No technique verdicts yet.**

- **P2 pilot (real, committed):** N=30 per task set, every arm, both models,
  ~$1.37 total API spend (`data/pilot_results.jsonl`,
  `data/procedural_check_results.jsonl`). The pilot exists to size the full
  run via power analysis — it is deliberately too small to decide any verdict,
  and none is claimed from it.
- **Pre-registration FROZEN** (`data/prereg_freeze.json`, config hash
  `9b8af804d62b0fec771fac4a2b7d86acc9f7fb27c639965b3fc98710482eedad`): N=449
  (math), N=250 (logic — capped by BBH date_understanding's entire 250-row
  test split; detects ~4-8pp instead of the 3pp target, still clear of the
  2pp placebo bound), N=739 (procedural — budget-capped from an ideal 1377).
- **One genuine boundary finding from the pilot:** the original procedural
  task hit 100% accuracy on every arm/model combo (ceiling effect, zero
  variance). The generator was hardened and re-checked — real variance
  everywhere except Sonnet 5 with reasoning ON, which stays at 100% on both
  baseline and CoT. That one comparison has no room to show an effect;
  reported as a limitation, not engineered around.
- **P3 (the full pre-registered run, ~$21-24 projected) has not been run.**
  Placebo / still-works / actively-hurts calls come only from P3 under the
  frozen verdict rule in `docs/prereg.md`.

## Layout

- `prereg/` — frozen experimental design: technique arms, models, stats constants,
  hash-freeze gate
- `harness/` — prompt construction, answer extraction, cost-metered batch runner,
  procedural task generator
- `stats/` — paired bootstrap confidence intervals
- `docs/prereg.md` — the human-readable pre-registration
