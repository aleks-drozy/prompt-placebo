# prompt-placebo

Which 2023-era prompt-engineering folklore (role prompts, "think step by step,"
emotional stakes, tips, politeness, few-shot) still causally improves accuracy on
2026 models that reason natively — and which is now placebo?

Paired-delta design: every technique is measured against an identical baseline on
the *same question*, so the comparison isolates the technique's effect from
question difficulty. Bootstrap confidence intervals, Holm-Bonferroni correction
across techniques, and a pre-registration hash-freeze (the stats and verdict rule
are committed and hashed before any full run) keep the result honest.

**Status:** harness under construction. No results yet.

## Layout

- `prereg/` — frozen experimental design: technique arms, models, stats constants,
  hash-freeze gate
- `harness/` — prompt construction, answer extraction, cost-metered batch runner,
  procedural task generator
- `stats/` — paired bootstrap confidence intervals
- `docs/prereg.md` — the human-readable pre-registration
