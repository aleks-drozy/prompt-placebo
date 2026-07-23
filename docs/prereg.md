# Pre-registration — Prompt-Technique Efficacy Audit

Written before any full experimental run. Committed to git and hash-frozen
(`prereg/freeze.py`) so the runner refuses to execute the full run if anything
below has changed since this document and `prereg/config.py` were locked in.

## The question

On 2026 models that reason natively, which classic prompting techniques still
causally change accuracy on objectively-scoreable tasks — and which are
placebo?

## Design invariant

The answer-format instruction and the answer-extraction parser are IDENTICAL
across every arm (see `prereg/arms.py`, `harness/prompt_builder.py`,
`harness/extractors.py`). The technique text is the only thing that varies
between arms. This is the load-bearing assumption of the whole experiment —
see `tests/test_prompt_builder.py::test_format_instruction_identical_across_every_arm`,
which fails the build if it's ever violated.

## Technique arms (verbatim, frozen)

Source of truth: `prereg/arms.py`. Prose summary:

| Arm | Name | Delta vs. baseline |
|---|---|---|
| B | baseline | Question + format instruction, nothing else |
| T1 | zero_shot_cot | + "Think step by step before answering." — appended *after* the format instruction (matches the original zero-shot-CoT construction: the trigger goes last, right before generation) |
| T2 | role_prompt | + "You are a world-class expert in {domain}." prepended before the question |
| T3 | emotional_stakes | + "This is very important to my career." |
| T4 | incentive | + "I'll tip $200 for a correct answer." |
| T5 | politeness | Question wrapped in a "Could you please help me... / Thank you so much!" politeness frame (baseline stays terse) |
| T6 | few_shot | 3 fixed worked examples per task set, prepended before the question (same 3 examples for every question in that task set) |

The format instruction itself asks the model to *end* its response with a
final `ANSWER: <value>` line, rather than respond with *only* that line —
this is what lets T1's reasoning instruction coexist with an identical parser
across arms without contradiction.

## Models

| Model | Reasoning | Arms run |
|---|---|---|
| `claude-haiku-4-5` | off (no `thinking` support relevant here) | all 7 |
| `claude-sonnet-5` | **explicitly disabled** (`thinking: {"type": "disabled"}`) | all 7 |
| `claude-sonnet-5` | adaptive (`thinking: {"type": "adaptive"}`) | B, T1 only |

Implementation note: Claude Sonnet 5 runs **adaptive thinking by default**
when the `thinking` parameter is omitted (unlike Sonnet 4.6, which defaulted
to thinking-off). The main-grid "reasoning off" cells therefore require
`harness/runner.py` to pass `thinking: {"type": "disabled"}` explicitly —
omitting the parameter would silently contaminate the "off" condition, which
is the whole point of the third row above.

## Task sets

1. **Math** — GSM8K test-split subsample (numeric answer).
2. **Logic** — BIG-Bench-Hard subsample (multiple choice).
3. **Procedural** — seeded arithmetic generator (`harness/task_generator.py`),
   contamination-free by construction; numeric answer.

Final N per task set is frozen into `prereg/config.py` **after** the pilot
(P2) — see `prereg/config.py::TASK_SETS`, currently `n: null` for all three.

## Statistics (frozen, `prereg/config.py`)

- Per (technique, model, task): paired bootstrap CI over the per-question
  accuracy delta vs. baseline. `n_boot=10000`, `seed=42`,
  `alpha=0.05` (`stats/bootstrap.py::paired_bootstrap_ci`).
- Holm-Bonferroni correction across the 6 technique comparisons per model×task.
- Verdict rule (`stats/bootstrap.py::classify_verdict`), checked in this
  priority order:
  1. CI entirely within ±2 percentage points → **placebo** (equivalence bound)
  2. CI excludes 0, positive → **still works**
  3. CI excludes 0, negative → **actively hurts**
  4. Otherwise → **inconclusive**
- Temperature 0 semantics via `thinking: disabled` where applicable; one
  sample per question per arm; seeds and question order committed.

## Budget

Hard cap: **€25**, enforced by `harness/cost_meter.py` (`BudgetExceededError`
halts the runner before any call that would exceed the cap — never after).
All full-grid runs use the Anthropic Message Batches API (50% off standard
per-token rates). Pricing constants and their expiry are documented in
`harness/cost_meter.py`.

## Freeze status

**Not yet frozen.** `prereg/freeze.py::write_freeze()` has not been run. Task
set Ns are still `null` pending the pilot (P2). The freeze will be committed,
hashed, and gated (`check_freeze()`) before the full run (P3) — see
`prereg/freeze.py` module docstring for the mechanism.
