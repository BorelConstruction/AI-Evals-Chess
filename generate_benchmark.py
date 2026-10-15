"""Generate the fixed chess-position benchmark.

Generation is intentionally separate from evaluation. The resulting JSONL can
be evaluated without network access, a Lichess token, or a Stockfish binary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import berserk
import chess
import chess.engine
import requests

from chess_scoring import win_probability
from sampler import (
    DEFAULT_CUTOFF_FRACTION,
    DEFAULT_RATINGS,
    DEFAULT_SPEEDS,
    ExplorerCache,
    normalized_fen,
    sample_line,
    total_games,
)


SCHEMA_VERSION = 1
DEFAULT_SEED = 20261012
DEFAULT_UNTIL = "2026-09"
DEFAULT_EXPLORER_MOVES = 50
DEFAULT_POSITION_COUNT = 50
DEFAULT_NODES = 5_000_000
DEFAULT_THREADS = 1
DEFAULT_HASH_MB = 256
DEFAULT_TARGET_EVAL_THRESHOLD = 1.0
MATE_SCORE_CP = 100_000
EXPLORER_URL = "https://explorer.lichess.org"
NETWORK_TIMEOUT_SECONDS = (10.0, 30.0)


@dataclass(frozen=True)
class Candidate:
    fen: str
    line_uci: tuple[str, ...]
    side_to_move: str
    root_games: int
    terminal_games: int


@dataclass(frozen=True)
class EngineEvaluation:
    pawns: float
    mate_in: int | None


CandidateEvaluator = Callable[[Candidate], EngineEvaluation]


class TimeoutTokenSession(berserk.TokenSession):
    """Authenticated Requests session with a default timeout."""

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", NETWORK_TIMEOUT_SECONDS)
        return super().request(method, url, **kwargs)


def create_lichess_client() -> tuple[berserk.Client, requests.Session]:
    """Create a Lichess client using an environment-only token."""

    token = os.environ.get("LICHESS_API_TOKEN")
    if not token:
        raise RuntimeError("LICHESS_API_TOKEN must be set to generate the benchmark")
    session = TimeoutTokenSession(token)
    return berserk.Client(session=session, explorer_url=EXPLORER_URL), session


def _cache_total(
    cache: ExplorerCache,
    fen: str,
    ratings: tuple[str, ...],
    speeds: tuple[str, ...],
    until: str | None,
    moves: int,
) -> int:
    key = (normalized_fen(chess.Board(fen)), ratings, speeds, until, moves)
    return total_games(cache[key])


def _validated_target_eval_threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("target_eval_threshold must be a number")
    threshold = float(value)
    if not math.isfinite(threshold):
        raise ValueError("target_eval_threshold must be finite")
    if threshold < 0.0:
        raise ValueError("target_eval_threshold cannot be negative")
    return threshold


def collect_candidates(
    client: berserk.Client,
    *,
    evaluate_candidate: CandidateEvaluator,
    seed: int,
    count: int,
    ratings: tuple[str, ...],
    speeds: tuple[str, ...],
    fraction: float,
    until: str | None,
    explorer_moves: int,
    request_delay: float,
    max_candidates: int,
    target_eval_threshold: float = DEFAULT_TARGET_EVAL_THRESHOLD,
) -> tuple[list[Candidate], list[EngineEvaluation]]:
    """Collect qualifying unique positions with an equal side-to-move split.

    Each unique position for a side with quota remaining is evaluated once.
    Only positions whose absolute evaluation strictly exceeds the threshold are
    returned, paired with the evaluation used for that decision.
    """

    if count <= 0 or count % 2:
        raise ValueError("position count must be a positive even number")
    if max_candidates < count:
        raise ValueError("max_candidates cannot be smaller than count")
    threshold = _validated_target_eval_threshold(target_eval_threshold)

    rng = random.Random(seed)
    cache: ExplorerCache = {}
    quota = count // 2
    side_counts = {"white": 0, "black": 0}
    seen: set[str] = set()
    candidates: list[Candidate] = []
    evaluations: list[EngineEvaluation] = []

    for attempt in range(1, max_candidates + 1):
        line, fen = sample_line(
            client,
            ratings=ratings,
            speeds=speeds,
            fraction=fraction,
            rng=rng,
            until=until,
            moves=explorer_moves,
            cache=cache,
            request_delay=request_delay,
        )
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError(f"sampler produced an invalid position: {fen}")
        if board.is_game_over():
            continue

        side = "white" if board.turn == chess.WHITE else "black"
        position_key = normalized_fen(board)
        if side_counts[side] >= quota or position_key in seen:
            continue
        seen.add(position_key)

        candidate = Candidate(
            fen=fen,
            line_uci=tuple(line),
            side_to_move=side,
            root_games=_cache_total(
                cache,
                chess.STARTING_FEN,
                ratings,
                speeds,
                until,
                explorer_moves,
            ),
            terminal_games=_cache_total(
                cache,
                fen,
                ratings,
                speeds,
                until,
                explorer_moves,
            ),
        )
        evaluation = evaluate_candidate(candidate)
        if not isinstance(evaluation, EngineEvaluation):
            raise TypeError("evaluate_candidate must return an EngineEvaluation")
        if isinstance(evaluation.pawns, bool) or not isinstance(
            evaluation.pawns, (int, float)
        ):
            raise TypeError("engine evaluation must be numeric")
        if not math.isfinite(evaluation.pawns):
            raise ValueError("engine evaluation must be finite")
        if abs(evaluation.pawns) <= threshold:
            print(
                f"rejected candidate at {evaluation.pawns:+.2f} pawns "
                f"(threshold {threshold:g}; attempt {attempt})",
                flush=True,
            )
            continue

        candidates.append(candidate)
        evaluations.append(evaluation)
        side_counts[side] += 1
        print(
            f"accepted {len(candidates):02d}/{count} at "
            f"{evaluation.pawns:+.2f} pawns "
            f"(white {side_counts['white']}/{quota}, "
            f"black {side_counts['black']}/{quota}; attempt {attempt})",
            flush=True,
        )
        if len(candidates) == count:
            return candidates, evaluations

    raise RuntimeError(
        f"collected only {len(candidates)} qualifying unique balanced positions "
        f"after {max_candidates} sampling attempts"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _configured_stockfish(
    *,
    stockfish_path: Path,
    nodes: int,
    threads: int,
    hash_mb: int,
) -> Iterator[tuple[chess.engine.SimpleEngine, dict[str, Any]]]:
    with chess.engine.SimpleEngine.popen_uci(
        str(stockfish_path), timeout=60.0
    ) as engine:
        engine_name = engine.id.get("name")
        if engine_name != "Stockfish 17":
            raise RuntimeError(
                "benchmark generation requires Stockfish 17; "
                f"engine identified itself as {engine_name!r}"
            )
        engine.configure({"Threads": threads, "Hash": hash_mb})
        engine_metadata = {
            "name": engine_name,
            "author": engine.id.get("author", "unknown"),
            "binary_sha256": sha256_file(stockfish_path),
            "limit": {"type": "nodes", "value": nodes},
            "threads": threads,
            "hash_mb": hash_mb,
            "pov": "white",
            "mate_score_cp": MATE_SCORE_CP,
        }
        yield engine, engine_metadata


def _evaluate_candidate(
    engine: chess.engine.SimpleEngine,
    candidate: Candidate,
    *,
    nodes: int,
) -> EngineEvaluation:
    if "Clear Hash" in engine.options:
        engine.configure({"Clear Hash": None})
    info = engine.analyse(
        chess.Board(candidate.fen),
        chess.engine.Limit(nodes=nodes),
        info=chess.engine.INFO_SCORE,
    )
    score = info.get("score")
    if not isinstance(score, chess.engine.PovScore):
        raise RuntimeError(
            f"Stockfish returned no score for position: {candidate.fen}"
        )
    white_score = score.white()
    centipawns = white_score.score(mate_score=MATE_SCORE_CP)
    if centipawns is None:
        raise RuntimeError(
            f"Stockfish score could not be converted for position: {candidate.fen}"
        )
    pawns = centipawns / 100.0
    if not math.isfinite(pawns):
        raise RuntimeError(
            f"Stockfish returned a non-finite score for position: {candidate.fen}"
        )
    return EngineEvaluation(pawns=pawns, mate_in=white_score.mate())


def evaluate_candidates(
    candidates: list[Candidate],
    *,
    stockfish_path: Path,
    nodes: int,
    threads: int,
    hash_mb: int,
) -> tuple[list[EngineEvaluation], dict[str, Any]]:
    """Evaluate every candidate once from White's point of view."""

    evaluations: list[EngineEvaluation] = []
    with _configured_stockfish(
        stockfish_path=stockfish_path,
        nodes=nodes,
        threads=threads,
        hash_mb=hash_mb,
    ) as (engine, engine_metadata):
        for index, candidate in enumerate(candidates, start=1):
            evaluation = _evaluate_candidate(engine, candidate, nodes=nodes)
            evaluations.append(evaluation)
            print(
                f"evaluated {index:02d}/{len(candidates)}: "
                f"{evaluation.pawns:+.2f}",
                flush=True,
            )

    return evaluations, engine_metadata


def build_rows(
    candidates: list[Candidate],
    evaluations: list[EngineEvaluation],
    *,
    engine_metadata: dict[str, Any],
    seed: int,
    ratings: tuple[str, ...],
    speeds: tuple[str, ...],
    fraction: float,
    until: str | None,
    explorer_moves: int,
) -> list[dict[str, Any]]:
    if len(candidates) != len(evaluations):
        raise ValueError("candidate and evaluation counts do not match")

    rows: list[dict[str, Any]] = []
    for index, (candidate, evaluation) in enumerate(
        zip(candidates, evaluations, strict=True), start=1
    ):
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "id": f"position-{index:03d}",
                "fen": candidate.fen,
                "target_pawns": evaluation.pawns,
                "target_win_probability": win_probability(evaluation.pawns),
                "mate_in": evaluation.mate_in,
                "line_uci": list(candidate.line_uci),
                "side_to_move": candidate.side_to_move,
                "source": {
                    "provider": "lichess-opening-explorer",
                    "ratings": list(ratings),
                    "speeds": list(speeds),
                    "until": until,
                    "cutoff_fraction": fraction,
                    "moves_per_query": explorer_moves,
                    "root_games": candidate.root_games,
                    "terminal_games": candidate.terminal_games,
                    "seed": seed,
                },
                "engine": engine_metadata,
            }
        )
    return rows


def write_jsonl_atomic(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def resolve_stockfish_path(value: str | None) -> Path:
    configured = value or os.environ.get("STOCKFISH_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    discovered = shutil.which("stockfish")
    if discovered:
        return Path(discovered).resolve()
    raise FileNotFoundError(
        "Stockfish was not found; pass --stockfish or set STOCKFISH_PATH"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "data" / "chess_positions.jsonl",
    )
    parser.add_argument("--stockfish", help="path to the Stockfish executable")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--until", default=DEFAULT_UNTIL)
    parser.add_argument("--nodes", type=int, default=DEFAULT_NODES)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--max-candidates", type=int, default=2_000)
    parser.add_argument(
        "--target-eval-threshold",
        type=float,
        default=DEFAULT_TARGET_EVAL_THRESHOLD,
        help="accept positions only when abs(Stockfish evaluation) exceeds this",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output dataset",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    target_eval_threshold = _validated_target_eval_threshold(
        args.target_eval_threshold
    )
    output_path = args.output.resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {output_path}; pass --overwrite to replace it"
        )

    stockfish_path = resolve_stockfish_path(args.stockfish)
    ratings = tuple(DEFAULT_RATINGS)
    speeds = tuple(DEFAULT_SPEEDS)
    client, session = create_lichess_client()
    try:
        with _configured_stockfish(
            stockfish_path=stockfish_path,
            nodes=args.nodes,
            threads=DEFAULT_THREADS,
            hash_mb=DEFAULT_HASH_MB,
        ) as (engine, engine_metadata):

            def evaluate_candidate(candidate: Candidate) -> EngineEvaluation:
                return _evaluate_candidate(engine, candidate, nodes=args.nodes)

            candidates, evaluations = collect_candidates(
                client,
                evaluate_candidate=evaluate_candidate,
                seed=args.seed,
                count=DEFAULT_POSITION_COUNT,
                ratings=ratings,
                speeds=speeds,
                fraction=DEFAULT_CUTOFF_FRACTION,
                until=args.until,
                explorer_moves=DEFAULT_EXPLORER_MOVES,
                request_delay=args.request_delay,
                max_candidates=args.max_candidates,
                target_eval_threshold=target_eval_threshold,
            )
    finally:
        session.close()
    rows = build_rows(
        candidates,
        evaluations,
        engine_metadata=engine_metadata,
        seed=args.seed,
        ratings=ratings,
        speeds=speeds,
        fraction=DEFAULT_CUTOFF_FRACTION,
        until=args.until,
        explorer_moves=DEFAULT_EXPLORER_MOVES,
    )
    write_jsonl_atomic(rows, output_path)
    print(f"wrote {len(rows)} benchmark rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
