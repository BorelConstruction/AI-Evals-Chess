"""Sample legal chess positions from the Lichess opening explorer.

The sampler performs a weighted random walk from the initial position. It
returns the first position whose filtered game count is below a fixed fraction
of the filtered game count at the initial position.
"""

from __future__ import annotations

import math
import random
import time
import warnings
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, Protocol

import berserk
import chess


DEFAULT_RATINGS = ("1800", "2000")
DEFAULT_SPEEDS = ("blitz", "rapid")
DEFAULT_CUTOFF_FRACTION = 0.0001
RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})

ExplorerCacheKey = tuple[str, tuple[str, ...], tuple[str, ...], str | None, int]
ExplorerCache = MutableMapping[ExplorerCacheKey, dict[str, Any]]


class RandomChoices(Protocol):
    """The portion of ``random.Random`` used by ``sample_line``."""

    def choices(
        self,
        population: Sequence[dict[str, Any]],
        weights: Sequence[int],
        *,
        k: int = 1,
    ) -> list[dict[str, Any]]: ...


def total_games(game_data: Mapping[str, Any]) -> int:
    """Return the number of games represented by an explorer result."""

    counts: list[int] = []
    for result in ("white", "draws", "black"):
        count = game_data[result]
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"{result!r} game count must be an integer")
        if count < 0:
            raise ValueError(f"{result!r} game count cannot be negative")
        counts.append(count)
    return sum(counts)


def normalized_fen(board: chess.Board) -> str:
    """Return the four position-defining FEN fields."""

    return " ".join(board.fen().split()[:4])


def _validate_explorer_result(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise TypeError("Lichess explorer response must be a JSON object")
    total_games(data)

    moves = data.get("moves")
    if not isinstance(moves, list):
        raise TypeError("Lichess explorer response must contain a moves list")
    for move in moves:
        if not isinstance(move, dict):
            raise TypeError("Each Lichess explorer move must be an object")
        if not isinstance(move.get("san"), str):
            raise TypeError("Each Lichess explorer move must contain SAN text")
        if not isinstance(move.get("uci"), str):
            raise TypeError("Each Lichess explorer move must contain UCI text")
        total_games(move)
    return data


def _retry_delay(error: berserk.exceptions.ResponseError, attempt: int) -> float:
    retry_after = error.response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            parsed = float(retry_after)
        except ValueError:
            pass
        else:
            if math.isfinite(parsed) and parsed >= 0.0:
                return parsed
    if error.status_code == 429:
        return 60.0
    return min(2.0**attempt, 10.0)


def _explorer(
    client: berserk.Client,
    board: chess.Board,
    ratings: tuple[str, ...],
    speeds: tuple[str, ...],
    *,
    until: str | None,
    moves: int,
    cache: ExplorerCache,
    max_attempts: int,
    request_delay: float,
) -> dict[str, Any]:
    key = (normalized_fen(board), ratings, speeds, until, moves)
    cached = cache.get(key)
    if cached is not None:
        return cached

    if request_delay:
        time.sleep(request_delay)

    for attempt in range(max_attempts):
        try:
            data = client.opening_explorer.get_lichess_games(
                position=board.fen(),
                ratings=list(ratings),
                speeds=list(speeds),
                until=until,
                moves=moves,
                top_games=0,
                recent_games=0,
                history=False,
            )
            validated = _validate_explorer_result(data)
            cache[key] = validated
            return validated
        except berserk.exceptions.ResponseError as error:
            if error.status_code not in RETRYABLE_HTTP_STATUSES:
                raise
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    "Lichess explorer failed after "
                    f"{max_attempts} attempts (HTTP {error.status_code})"
                ) from error
            delay = _retry_delay(error, attempt)
            warnings.warn(
                f"Lichess explorer returned HTTP {error.status_code}; "
                f"retrying in {delay:g} seconds",
                RuntimeWarning,
                stacklevel=2,
            )
            time.sleep(delay)
        except berserk.exceptions.ApiError as error:
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"Lichess explorer failed after {max_attempts} attempts"
                ) from error
            delay = min(2.0**attempt, 10.0)
            warnings.warn(
                f"Lichess explorer request failed; retrying in {delay:g} seconds",
                RuntimeWarning,
                stacklevel=2,
            )
            time.sleep(delay)

    raise AssertionError("unreachable")


def sample_line(
    client: berserk.Client,
    ratings: Sequence[str | int] = DEFAULT_RATINGS,
    speeds: Sequence[str] = DEFAULT_SPEEDS,
    fraction: float = DEFAULT_CUTOFF_FRACTION,
    rng: RandomChoices = random,
    *,
    until: str | None = None,
    moves: int = 50,
    cache: ExplorerCache | None = None,
    max_attempts: int = 5,
    request_delay: float = 0.1,
    max_plies: int = 100,
) -> tuple[list[str], str]:
    """Return a sampled UCI line and its terminal FEN.

    Moves are selected in proportion to their game counts. The threshold is
    always based on the root position, so the returned FEN is normally the
    first position along the walk represented by fewer than ``fraction`` of
    all root games for the requested filters.
    """

    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be finite and strictly between 0 and 1")

    normalized_ratings = tuple(str(rating) for rating in ratings)
    normalized_speeds = tuple(str(speed) for speed in speeds)
    if not normalized_ratings:
        raise ValueError("at least one rating bucket is required")
    if not normalized_speeds:
        raise ValueError("at least one speed is required")

    response_cache: ExplorerCache = cache if cache is not None else {}
    board = chess.Board()
    data = _explorer(
        client,
        board,
        normalized_ratings,
        normalized_speeds,
        until=until,
        moves=moves,
        cache=response_cache,
        max_attempts=max_attempts,
        request_delay=request_delay,
    )
    root_games = total_games(data)
    if root_games == 0:
        raise ValueError("Lichess explorer returned no games for these filters")
    threshold = fraction * root_games

    line: list[str] = []
    while total_games(data) >= threshold:
        if not data["moves"]:
            raise RuntimeError(
                "Lichess explorer returned no continuations before the "
                "sampling cutoff was reached"
            )
        if len(line) >= max_plies:
            raise RuntimeError(f"sampled line exceeded {max_plies} plies")

        # Stable sorting prevents API response ordering from changing a seeded run.
        candidates = sorted(data["moves"], key=lambda move: move["uci"])
        weights = [total_games(move) for move in candidates]
        if sum(weights) == 0:
            raise ValueError("Lichess explorer returned moves with zero total weight")

        choice = rng.choices(candidates, weights, k=1)[0]
        move = board.parse_san(choice["san"])
        accepted_ucis = {move.uci(), board.uci(move, chess960=True)}
        if choice["uci"] not in accepted_ucis:
            raise ValueError(
                "Lichess explorer SAN/UCI mismatch: "
                f"{choice['san']!r} != {choice['uci']!r}"
            )
        board.push(move)
        line.append(move.uci())
        data = _explorer(
            client,
            board,
            normalized_ratings,
            normalized_speeds,
            until=until,
            moves=moves,
            cache=response_cache,
            max_attempts=max_attempts,
            request_delay=request_delay,
        )

    return line, board.fen()
