"""Shared parsing and scoring primitives for the chess evaluation."""

from __future__ import annotations

import json
import math
import re
from typing import Any


LICHESS_CENTIPAWN_MULTIPLIER = 0.00368208
PAWNS_TO_CENTIPAWNS = 100.0
REASONING_FIELD = "reasoning"
RESPONSE_FIELD = "evaluation_pawns"

_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_NUMBER_RE = re.compile(rf"(?<![\w.])(?P<number>{_NUMBER_PATTERN})(?![\w.])")
_FENCE_RE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```[ \t]*\Z",
    re.IGNORECASE | re.DOTALL,
)
_FIELD_NAME_RE = re.compile(rf"\b{re.escape(RESPONSE_FIELD)}\b", re.IGNORECASE)
_FIELD_VALUE_RE = re.compile(
    rf"\b{re.escape(RESPONSE_FIELD)}\b[\"'`]?\s*"
    rf"(?::|=|\bis\b)\s*[\"'`]?\s*"
    rf"(?P<number>{_NUMBER_PATTERN})(?![\w.])",
    re.IGNORECASE,
)
_WRONG_UNIT_RE = re.compile(r"\s*(?:%|cp\b|centipawns?\b)", re.IGNORECASE)
_DESCRIPTIVE_VALUE_RE = re.compile(
    rf"\b(?:pawn\s+evaluation|evaluation|eval)\b\s*"
    rf"(?::|=|\bis\b)\s*(?P<number>{_NUMBER_PATTERN})(?![\w.])",
    re.IGNORECASE,
)


def win_probability(evaluation_pawns: float) -> float:
    """Convert a White-relative pawn evaluation with the Lichess-style sigmoid.

    This is the Lichess sigmoid

    ``1 / (1 + exp(-0.00368208 * centipawns))``.

    The branch form is numerically stable for very large finite evaluations.
    """

    if isinstance(evaluation_pawns, bool):
        raise TypeError("pawn evaluation must be a number, not a boolean")
    try:
        pawns = float(evaluation_pawns)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("pawn evaluation must be numeric") from error
    if not math.isfinite(pawns):
        raise ValueError("pawn evaluation must be finite")

    log_odds = LICHESS_CENTIPAWN_MULTIPLIER * PAWNS_TO_CENTIPAWNS * pawns
    if log_odds >= 0.0:
        return 1.0 / (1.0 + math.exp(-log_odds))
    exp_log_odds = math.exp(log_odds)
    return exp_log_odds / (1.0 + exp_log_odds)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _finite_evaluation(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("pawn evaluation must be numeric")

    if isinstance(value, str):
        number = value.strip().replace("\N{MINUS SIGN}", "-")
        if re.fullmatch(_NUMBER_PATTERN, number) is None:
            raise ValueError("pawn evaluation must be numeric")
    elif isinstance(value, (int, float)):
        number = value
    else:
        raise ValueError("pawn evaluation must be numeric")

    try:
        evaluation = float(number)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("pawn evaluation is not numeric") from error
    if not math.isfinite(evaluation):
        raise ValueError("pawn evaluation must be finite")
    return evaluation


def _json_evaluation(payload: Any) -> float:
    if isinstance(payload, dict):
        if RESPONSE_FIELD not in payload:
            raise ValueError(f"JSON response has no {RESPONSE_FIELD!r} field")
        return _finite_evaluation(payload[RESPONSE_FIELD])
    return _finite_evaluation(payload)


def _matched_evaluation(text: str, match: re.Match[str]) -> float:
    number_start, number_end = match.span("number")
    if text[:number_start].rstrip().endswith("#"):
        raise ValueError("mate notation is not a pawn evaluation")
    if _WRONG_UNIT_RE.match(text[number_end:]) is not None:
        raise ValueError("response number is not expressed in pawns")
    return _finite_evaluation(match.group("number"))


def parse_model_evaluation(completion: str) -> float:
    """Extract one unambiguous, finite pawn evaluation from model text."""

    if not isinstance(completion, str):
        raise TypeError("model completion must be text")

    text = completion.strip().replace("\N{MINUS SIGN}", "-")
    if not text:
        raise ValueError("response is empty")

    fence_match = _FENCE_RE.fullmatch(text)
    if fence_match is not None:
        text = fence_match.group("body").strip()
        if not text:
            raise ValueError("response code fence is empty")
    elif text.startswith("```") or text.endswith("```"):
        raise ValueError("response contains a malformed code fence")

    structured_response = fence_match is not None or text.startswith(("{", "["))
    try:
        payload = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except json.JSONDecodeError as error:
        if structured_response:
            raise ValueError(f"response contains malformed JSON: {error.msg}") from error
    else:
        return _json_evaluation(payload)

    field_names = _FIELD_NAME_RE.findall(text)
    if len(field_names) > 1:
        raise ValueError(f"response contains multiple {RESPONSE_FIELD!r} fields")
    if field_names:
        field_match = _FIELD_VALUE_RE.search(text)
        if field_match is None:
            raise ValueError(f"{RESPONSE_FIELD!r} is not followed by a number")
        return _matched_evaluation(text, field_match)

    descriptive_matches = list(_DESCRIPTIVE_VALUE_RE.finditer(text))
    if len(descriptive_matches) > 1:
        raise ValueError("response contains multiple labeled pawn evaluations")
    if descriptive_matches:
        return _matched_evaluation(text, descriptive_matches[0])

    number_matches = list(_NUMBER_RE.finditer(text))
    if not number_matches:
        raise ValueError("response contains no pawn-evaluation number")
    if len(number_matches) > 1:
        raise ValueError("response contains multiple unlabeled numbers")
    return _matched_evaluation(text, number_matches[0])


def probability_similarity(predicted_pawns: float, target_pawns: float) -> float:
    """Return ``1 - absolute error`` between converted win probabilities."""

    predicted_probability = win_probability(predicted_pawns)
    target_probability = win_probability(target_pawns)
    return 1.0 - abs(predicted_probability - target_probability)
