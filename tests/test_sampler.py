from __future__ import annotations

import math
import unittest
from typing import Any

import chess

from sampler import ExplorerCache, normalized_fen, sample_line, total_games


def explorer_result(
    total: int, moves: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {"white": total, "draws": 0, "black": 0, "moves": moves or []}


def explorer_move(san: str, uci: str, total: int) -> dict[str, Any]:
    return {
        "san": san,
        "uci": uci,
        "white": total,
        "draws": 0,
        "black": 0,
    }


class FakeOpeningExplorer:
    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self.responses = responses
        self.calls = 0

    def get_lichess_games(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        key = normalized_fen(chess.Board(kwargs["position"]))
        return self.responses[key]


class FakeClient:
    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self.opening_explorer = FakeOpeningExplorer(responses)


class FirstChoiceRandom:
    def __init__(self) -> None:
        self.weights: list[int] | None = None

    def choices(
        self,
        population: list[dict[str, Any]],
        weights: list[int],
        *,
        k: int = 1,
    ) -> list[dict[str, Any]]:
        self.weights = weights
        return [population[0]]


def board_after(*sans: str) -> chess.Board:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


class TotalGamesTests(unittest.TestCase):
    def test_sums_results(self) -> None:
        self.assertEqual(total_games({"white": 2, "draws": 3, "black": 5}), 10)

    def test_bad_counts_fail(self) -> None:
        with self.assertRaises(KeyError):
            total_games({"white": 1, "draws": 2})
        with self.assertRaises(TypeError):
            total_games({"white": True, "draws": 2, "black": 3})
        with self.assertRaises(ValueError):
            total_games({"white": -1, "draws": 2, "black": 3})


class SampleLineTests(unittest.TestCase):
    def test_stops_at_first_position_below_threshold(self) -> None:
        root = chess.Board()
        after_e4 = board_after("e4")
        client = FakeClient(
            {
                normalized_fen(root): explorer_result(
                    100, [explorer_move("e4", "e2e4", 60)]
                ),
                normalized_fen(after_e4): explorer_result(49),
            }
        )
        line, fen = sample_line(
            client,
            fraction=0.5,
            rng=FirstChoiceRandom(),
            request_delay=0,
        )
        self.assertEqual(line, ["e2e4"])
        self.assertEqual(fen, after_e4.fen())

    def test_exact_threshold_continues(self) -> None:
        root = chess.Board()
        after_e4 = board_after("e4")
        after_e4_e5 = board_after("e4", "e5")
        client = FakeClient(
            {
                normalized_fen(root): explorer_result(
                    100, [explorer_move("e4", "e2e4", 60)]
                ),
                normalized_fen(after_e4): explorer_result(
                    50, [explorer_move("e5", "e7e5", 50)]
                ),
                normalized_fen(after_e4_e5): explorer_result(49),
            }
        )
        line, fen = sample_line(
            client,
            fraction=0.5,
            rng=FirstChoiceRandom(),
            request_delay=0,
        )
        self.assertEqual(line, ["e2e4", "e7e5"])
        self.assertEqual(fen, after_e4_e5.fen())

    def test_move_weights_use_all_results(self) -> None:
        root = chess.Board()
        after_d4 = board_after("d4")
        rng = FirstChoiceRandom()
        client = FakeClient(
            {
                normalized_fen(root): {
                    "white": 100,
                    "draws": 0,
                    "black": 0,
                    "moves": [
                        {
                            "san": "d4",
                            "uci": "d2d4",
                            "white": 2,
                            "draws": 3,
                            "black": 5,
                        }
                    ],
                },
                normalized_fen(after_d4): explorer_result(0),
            }
        )
        sample_line(client, rng=rng, request_delay=0)
        self.assertEqual(rng.weights, [10])

    def test_external_cache_avoids_duplicate_requests(self) -> None:
        root = chess.Board()
        after_e4 = board_after("e4")
        client = FakeClient(
            {
                normalized_fen(root): explorer_result(
                    100, [explorer_move("e4", "e2e4", 100)]
                ),
                normalized_fen(after_e4): explorer_result(0),
            }
        )
        cache: ExplorerCache = {}
        first = sample_line(client, cache=cache, request_delay=0)
        second = sample_line(client, cache=cache, request_delay=0)
        self.assertEqual(first, second)
        self.assertEqual(client.opening_explorer.calls, 2)

    def test_no_moves_above_cutoff_fails(self) -> None:
        root = chess.Board()
        client = FakeClient({normalized_fen(root): explorer_result(100)})
        with self.assertRaises(RuntimeError):
            sample_line(client, request_delay=0)

    def test_accepts_lichess_chess960_castling_uci(self) -> None:
        sans = ["e4", "e5", "Nf3", "Nc6", "Be2", "Nf6", "O-O"]
        lichess_ucis = ["e2e4", "e7e5", "g1f3", "b8c6", "f1e2", "g8f6", "e1h1"]
        responses: dict[str, dict[str, Any]] = {}
        board = chess.Board()
        for san, uci in zip(sans, lichess_ucis, strict=True):
            responses[normalized_fen(board)] = explorer_result(
                100, [explorer_move(san, uci, 100)]
            )
            board.push_san(san)
        responses[normalized_fen(board)] = explorer_result(0)
        client = FakeClient(responses)

        line, fen = sample_line(
            client,
            rng=FirstChoiceRandom(),
            request_delay=0,
            max_plies=10,
        )

        self.assertEqual(line[-1], "e1g1")
        self.assertEqual(fen, board.fen())

    def test_zero_root_games_fails(self) -> None:
        root = chess.Board()
        client = FakeClient({normalized_fen(root): explorer_result(0)})
        with self.assertRaises(ValueError):
            sample_line(client, request_delay=0)

    def test_bad_fraction_fails_before_network(self) -> None:
        client = FakeClient({})
        for fraction in (0.0, -0.1, 1.0, 1.1, math.nan):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                sample_line(client, fraction=fraction, request_delay=0)
        self.assertEqual(client.opening_explorer.calls, 0)


if __name__ == "__main__":
    unittest.main()
