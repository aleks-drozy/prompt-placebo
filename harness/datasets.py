"""Real benchmark-sourced task sets: GSM8K (math) and BBH (logic).

Rows are fetched from Hugging Face's public datasets-server REST API
(plain HTTP JSON, no ``datasets`` package dependency):

    GET https://datasets-server.huggingface.co/rows
        ?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}

A fixed-order pool of up to 500 parsed Questions per dataset is cached to
data/tasksets/{math,logic}_pool.json (fetched once, walking the dataset in
its natural order via offset 0, 100, 200, ... -- no randomness at fetch
time). The public ``fetch_*_questions`` functions then sample deterministically
from that cached pool with ``random.Random(seed).sample(...)``.

Dataset choices (traceability):

- Math ("task set 1"): dataset="openai/gsm8k", config="main", split="test"
  (1319 rows total). MIT licensed, from OpenAI. Confirmed by a live fetch:
  each row has "question" (the word problem) and "answer" (multi-line
  reasoning ending in a "#### <number>" line).

- Logic ("task set 2"): dataset="lukaemon/bbh", config="date_understanding",
  split="test" (250 rows total). This is BIG-Bench-Hard (Apache-2.0, from
  Google/BIG-bench), mirrored on Hugging Face under lukaemon/bbh. Confirmed
  by a live fetch to be exactly the expected shape -- no fallback config was
  needed: each row has "input" (the question followed by a literal
  "Options:\n(A) ...\n(B) ..." block) and "target" (the gold letter,
  parenthesized, e.g. "(B)").
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Callable

import requests

from harness.schema import AnswerFormat, Question, TaskType

_API_URL = "https://datasets-server.huggingface.co/rows"
_PAGE_SIZE = 100
_POOL_CAP = 500
_REQUEST_TIMEOUT_S = 30

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "tasksets"
MATH_POOL_PATH = _DATA_DIR / "math_pool.json"
LOGIC_POOL_PATH = _DATA_DIR / "logic_pool.json"

_GSM8K_DATASET = "openai/gsm8k"
_GSM8K_CONFIG = "main"
_GSM8K_SPLIT = "test"

_BBH_DATASET = "lukaemon/bbh"
_BBH_CONFIG = "date_understanding"
_BBH_SPLIT = "test"

# GSM8K answers end in a line like "#### 42" (or "#### 1,234"); the final
# answer is everything after the last "####" marker.
_GSM8K_FINAL_ANSWER_RE = re.compile(r"####\s*(.+?)\s*$")
# BBH "Options:" lines look like "(A) some text" -- one per line.
_BBH_OPTION_RE = re.compile(r"^\(([A-Z])\)\s*", re.MULTILINE)
# BBH "target" looks like "(C)".
_BBH_TARGET_RE = re.compile(r"\(([A-Z])\)")


def _fetch_pool_rows(dataset: str, config: str, split: str, cap: int) -> list[dict[str, Any]]:
    """Fetch up to `cap` raw rows from the datasets-server API.

    Walks the dataset in a fixed order (offset 0, 100, 200, ...) so the pool
    is reproducible across runs. Stops early if the dataset has fewer than
    `cap` rows.
    """
    rows: list[dict[str, Any]] = []
    offset = 0
    while offset < cap:
        length = min(_PAGE_SIZE, cap - offset)
        response = requests.get(
            _API_URL,
            params={
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            },
            timeout=_REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        page_rows = [entry["row"] for entry in response.json().get("rows", [])]
        rows.extend(page_rows)
        if len(page_rows) < length:
            break  # dataset exhausted before hitting the cap
        offset += length
    return rows


def parse_gsm8k_row(row: dict[str, Any], index: int) -> Question:
    """Parse one raw GSM8K API row into a Question.

    Pure function (no network access) so it is directly unit-testable.
    """
    match = _GSM8K_FINAL_ANSWER_RE.search(row["answer"])
    if match is None:
        raise ValueError(f"gsm8k row {index}: no '#### <answer>' marker found")
    gold_answer = match.group(1).replace(",", "").strip()
    return Question(
        id=f"math-{index:04d}",
        task_type=TaskType.MATH,
        domain="mathematics",
        prompt=row["question"],
        answer_format=AnswerFormat.NUMERIC,
        gold_answer=gold_answer,
    )


def parse_bbh_row(row: dict[str, Any], index: int) -> Question:
    """Parse one raw BBH (date_understanding) API row into a Question.

    Pure function (no network access) so it is directly unit-testable.
    """
    prompt = row["input"]
    choices = tuple(_BBH_OPTION_RE.findall(prompt))
    if not choices:
        raise ValueError(f"bbh row {index}: no lettered options found in input")
    target_match = _BBH_TARGET_RE.search(row["target"])
    if target_match is None:
        raise ValueError(f"bbh row {index}: no gold letter found in target {row['target']!r}")
    return Question(
        id=f"logic-{index:04d}",
        task_type=TaskType.LOGIC,
        domain="logical reasoning",
        prompt=prompt,
        answer_format=AnswerFormat.MULTIPLE_CHOICE,
        gold_answer=target_match.group(1),
        choices=choices,
    )


def _question_to_dict(question: Question) -> dict[str, Any]:
    return {
        "id": question.id,
        "task_type": question.task_type.value,
        "domain": question.domain,
        "prompt": question.prompt,
        "answer_format": question.answer_format.value,
        "gold_answer": question.gold_answer,
        "choices": list(question.choices),
    }


def _question_from_dict(data: dict[str, Any]) -> Question:
    """Reconstruct a Question from a cached JSON dict via the real constructor.

    Goes through Question(...) itself (not e.g. a raw namespace/dict cast)
    so __post_init__ validation still runs, and task_type/answer_format come
    back as real enums rather than raw strings.
    """
    return Question(
        id=data["id"],
        task_type=TaskType(data["task_type"]),
        domain=data["domain"],
        prompt=data["prompt"],
        answer_format=AnswerFormat(data["answer_format"]),
        gold_answer=data["gold_answer"],
        choices=tuple(data.get("choices", ())),
    )


def _write_pool(path: Path, questions: list[Question]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_question_to_dict(question) for question in questions]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_pool(path: Path) -> list[Question]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [_question_from_dict(item) for item in data]


def _fetch_and_cache_math_pool(path: Path) -> None:
    rows = _fetch_pool_rows(_GSM8K_DATASET, _GSM8K_CONFIG, _GSM8K_SPLIT, _POOL_CAP)
    questions = [parse_gsm8k_row(row, index) for index, row in enumerate(rows)]
    _write_pool(path, questions)


def _fetch_and_cache_logic_pool(path: Path) -> None:
    rows = _fetch_pool_rows(_BBH_DATASET, _BBH_CONFIG, _BBH_SPLIT, _POOL_CAP)
    questions = [parse_bbh_row(row, index) for index, row in enumerate(rows)]
    _write_pool(path, questions)


def _sample_from_pool(
    path: Path,
    fetch_and_cache: Callable[[Path], None],
    n: int,
    seed: int,
) -> tuple[Question, ...]:
    if not path.exists():
        fetch_and_cache(path)
    pool = _load_pool(path)
    if n > len(pool):
        raise ValueError(f"requested n={n} exceeds pool size {len(pool)} ({path})")
    return tuple(random.Random(seed).sample(pool, n))


def fetch_math_questions(n: int, seed: int) -> tuple[Question, ...]:
    """Deterministically sample n math (GSM8K) Questions.

    Loads the cached math pool (fetching + caching it first via the network
    if the cache file doesn't exist yet), then samples n questions from the
    pool using random.Random(seed).sample(...). Raises ValueError if n
    exceeds the pool size.
    """
    return _sample_from_pool(MATH_POOL_PATH, _fetch_and_cache_math_pool, n, seed)


def fetch_logic_questions(n: int, seed: int) -> tuple[Question, ...]:
    """Deterministically sample n logic (BBH) Questions.

    Same pattern as fetch_math_questions, for the logic pool.
    """
    return _sample_from_pool(LOGIC_POOL_PATH, _fetch_and_cache_logic_pool, n, seed)
