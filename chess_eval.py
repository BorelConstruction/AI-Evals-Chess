"""Inspect AI evaluation for White-relative chess position assessment."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import chess
from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, json_dataset
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate, system_message

from chess_scoring import (
    RESPONSE_FIELD,
    parse_model_evaluation,
    win_probability,
)


DATASET_PATH = Path(__file__).resolve().parent / "data" / "chess_positions.jsonl"
BENCHMARK_SIZE = 50
POSITIONS_PER_SIDE = BENCHMARK_SIZE // 2
EXPECTED_RECORD_FIELDS = {
    "schema_version",
    "id",
    "fen",
    "target_pawns",
    "target_win_probability",
    "mate_in",
    "line_uci",
    "side_to_move",
    "source",
    "engine",
}
EXPECTED_ENGINE_FIELDS = {
    "name",
    "author",
    "binary_sha256",
    "limit",
    "threads",
    "hash_mb",
    "pov",
    "mate_score_cp",
}
EXPECTED_SOURCE_FIELDS = {
    "provider",
    "ratings",
    "speeds",
    "until",
    "cutoff_fraction",
    "moves_per_query",
    "root_games",
    "terminal_games",
    "seed",
}

SYSTEM_MESSAGE = f"""You evaluate chess positions numerically.

Return the position's pawn evaluation from White's perspective:
- positive means White is better;
- negative means Black is better;
- zero means equal.

Return exactly one JSON object and no markdown or prose:
{{"{RESPONSE_FIELD}": <finite JSON number>}}
"""


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _integer(value: Any, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _validate_line(record: dict[str, Any], board: chess.Board) -> None:
    moves = record["line_uci"]
    if not isinstance(moves, list) or not moves or not all(
        isinstance(move, str) for move in moves
    ):
        raise TypeError("line_uci must be a non-empty list of strings")

    replay = chess.Board()
    for move_text in moves:
        try:
            move = chess.Move.from_uci(move_text)
        except ValueError as error:
            raise ValueError(f"invalid UCI move in {record['id']}: {move_text}") from error
        if move not in replay.legal_moves:
            raise ValueError(f"illegal UCI move in {record['id']}: {move_text}")
        replay.push(move)
    if replay.fen() != board.fen():
        raise ValueError(f"line_uci does not reproduce FEN for {record['id']}")


def _validate_provenance(record: dict[str, Any]) -> None:
    engine = record["engine"]
    if not isinstance(engine, dict) or set(engine) != EXPECTED_ENGINE_FIELDS:
        raise ValueError(f"invalid engine metadata for {record['id']}")
    if engine["name"] != "Stockfish 17":
        raise ValueError(f"unexpected engine for {record['id']}: {engine['name']!r}")
    if not isinstance(engine["author"], str) or not engine["author"]:
        raise TypeError("engine author must be non-empty text")
    binary_hash = engine["binary_sha256"]
    if (
        not isinstance(binary_hash, str)
        or len(binary_hash) != 64
        or any(character not in "0123456789abcdef" for character in binary_hash)
    ):
        raise ValueError("engine binary_sha256 must be a lowercase SHA-256 digest")
    if engine["limit"] != {"type": "nodes", "value": 5_000_000}:
        raise ValueError("engine limit must be 5,000,000 nodes")
    if engine["pov"] != "white":
        raise ValueError("engine POV must be white")
    if _integer(engine["threads"], "engine threads", minimum=1) != 1:
        raise ValueError("engine threads must be 1")
    if _integer(engine["hash_mb"], "engine hash_mb", minimum=1) != 256:
        raise ValueError("engine hash_mb must be 256")
    if _integer(engine["mate_score_cp"], "engine mate_score_cp", minimum=1) != 100_000:
        raise ValueError("engine mate_score_cp must be 100000")

    source = record["source"]
    if not isinstance(source, dict) or set(source) != EXPECTED_SOURCE_FIELDS:
        raise ValueError(f"invalid source metadata for {record['id']}")
    if source["provider"] != "lichess-opening-explorer":
        raise ValueError("unexpected benchmark source provider")
    if source["ratings"] != ["1800", "2000"]:
        raise ValueError("benchmark ratings must be 1800 and 2000")
    if source["speeds"] != ["blitz", "rapid"]:
        raise ValueError("benchmark speeds must be blitz and rapid")
    if source["until"] != "2026-09":
        raise ValueError("benchmark explorer end month must be 2026-09")
    if _integer(source["moves_per_query"], "moves_per_query", minimum=1) != 50:
        raise ValueError("moves_per_query must be 50")
    if _integer(source["seed"], "source seed") != 20261012:
        raise ValueError("source seed must be 20261012")
    root_games = _integer(source["root_games"], "root_games", minimum=1)
    terminal_games = _integer(source["terminal_games"], "terminal_games", minimum=0)
    if terminal_games >= source["cutoff_fraction"] * root_games:
        raise ValueError(f"sampling cutoff was not reached for {record['id']}")


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Validate a benchmark row and convert it to an Inspect sample."""

    if set(record) != EXPECTED_RECORD_FIELDS:
        missing = sorted(EXPECTED_RECORD_FIELDS - set(record))
        unexpected = sorted(set(record) - EXPECTED_RECORD_FIELDS)
        raise ValueError(
            f"invalid benchmark schema; missing={missing}, unexpected={unexpected}"
        )
    if record["schema_version"] != 1:
        raise ValueError(f"unsupported schema version: {record['schema_version']!r}")
    if not isinstance(record["id"], str) or not record["id"]:
        raise TypeError("benchmark id must be non-empty text")
    if not isinstance(record["fen"], str):
        raise TypeError("fen must be text")

    board = chess.Board(record["fen"])
    if not board.is_valid():
        raise ValueError(f"invalid benchmark FEN: {record['fen']}")
    actual_side = "white" if board.turn == chess.WHITE else "black"
    if record["side_to_move"] != actual_side:
        raise ValueError(
            f"side_to_move does not match FEN for {record['id']}: "
            f"{record['side_to_move']!r}"
        )

    target_pawns = _finite_number(record["target_pawns"], "target_pawns")
    stored_probability = _finite_number(
        record["target_win_probability"], "target_win_probability"
    )
    calculated_probability = win_probability(target_pawns)
    if not math.isclose(
        stored_probability,
        calculated_probability,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            f"stored probability does not match target_pawns for {record['id']}"
        )
    mate_in = record["mate_in"]
    if mate_in is not None and (isinstance(mate_in, bool) or not isinstance(mate_in, int)):
        raise TypeError("mate_in must be an integer or null")
    _validate_line(record, board)
    _validate_provenance(record)

    prompt = (
        f"FEN: {record['fen']}\n"
        "Evaluate this position in pawns from White's perspective. "
        f"Return only {{\"{RESPONSE_FIELD}\": <number>}}."
    )
    return Sample(
        id=record["id"],
        input=prompt,
        target=str(target_pawns),
        metadata={
            "fen": record["fen"],
            "side_to_move": actual_side,
            "target_win_probability": stored_probability,
            "mate_in": record["mate_in"],
            "source": record["source"],
            "engine": record["engine"],
        },
    )


def benchmark_dataset() -> Dataset:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"benchmark dataset not found: {DATASET_PATH}; "
            "run generate_benchmark.py first"
        )
    dataset = json_dataset(
        str(DATASET_PATH),
        sample_fields=record_to_sample,
        name="chess_positions",
    )
    if len(dataset) != BENCHMARK_SIZE:
        raise ValueError(
            f"benchmark must contain {BENCHMARK_SIZE} samples, got {len(dataset)}"
        )

    ids = [sample.id for sample in dataset]
    if len(set(ids)) != BENCHMARK_SIZE:
        raise ValueError("benchmark sample ids must be unique")
    positions = [
        " ".join(chess.Board(str(sample.metadata["fen"])).fen().split()[:4])
        for sample in dataset
    ]
    if len(set(positions)) != BENCHMARK_SIZE:
        raise ValueError("benchmark positions must be unique")
    sides = [sample.metadata["side_to_move"] for sample in dataset]
    if sides.count("white") != POSITIONS_PER_SIDE or sides.count("black") != POSITIONS_PER_SIDE:
        raise ValueError("benchmark must contain 25 positions for each side to move")
    return dataset


@scorer(metrics=[mean(), stderr()])
def winning_probability_similarity() -> Scorer:
    """Score ``1 - |P(model) - P(Stockfish)|`` after Lichess conversion."""

    async def score(state: TaskState, target: Target) -> Score:
        target_pawns = _finite_number(float(target.text), "dataset target")
        target_probability = win_probability(target_pawns)
        completion = state.output.completion

        try:
            predicted_pawns = parse_model_evaluation(completion)
        except (TypeError, ValueError) as error:
            return Score(
                value=0.0,
                answer=completion,
                explanation=f"Invalid response format: {error}",
                reason="invalid_response_format",
                metadata={"parse_error": str(error)},
            )

        predicted_probability = win_probability(predicted_pawns)
        absolute_error = abs(predicted_probability - target_probability)
        similarity = 1.0 - absolute_error
        return Score(
            value=similarity,
            answer=completion,
            explanation=(
                f"Model {predicted_pawns:+.2f} pawns -> "
                f"{predicted_probability:.6f}; Stockfish "
                f"{target_pawns:+.2f} pawns -> {target_probability:.6f}; "
                f"absolute probability error {absolute_error:.6f}."
            ),
            metadata={
                "predicted_pawns": predicted_pawns,
                "target_pawns": target_pawns,
                "predicted_win_probability": predicted_probability,
                "target_win_probability": target_probability,
                "absolute_probability_error": absolute_error,
            },
        )

    return score


@task
def chess_eval() -> Task:
    return Task(
        dataset=benchmark_dataset(),
        solver=[
            system_message(SYSTEM_MESSAGE),
            generate(temperature=0.0, tool_calls="none"),
        ],
        scorer=winning_probability_similarity(),
        metadata={
            "evaluation_pov": "white",
            "probability_formula": (
                "1 / (1 + exp(-0.00368208 * centipawns))"
            ),
            "score_formula": "1 - abs(P_model - P_stockfish)",
        },
    )
