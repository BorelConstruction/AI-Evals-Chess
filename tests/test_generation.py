from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess
import chess.engine

from generate_benchmark import (
    Candidate,
    build_rows,
    create_lichess_client,
    evaluate_candidates,
    write_jsonl_atomic,
)


class FakeEngine:
    def __init__(self, name: str, score: chess.engine.PovScore) -> None:
        self.id = {"name": name, "author": "test"}
        self.options: dict[str, object] = {}
        self.score = score
        self.configurations: list[dict[str, object]] = []
        self.closed = False

    def __enter__(self) -> "FakeEngine":
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True

    def configure(self, options: dict[str, object]) -> None:
        self.configurations.append(options)

    def analyse(
        self,
        board: chess.Board,
        limit: chess.engine.Limit,
        *,
        info: chess.engine.Info,
    ) -> dict[str, object]:
        return {"score": self.score}


class ClientTests(unittest.TestCase):
    def test_token_is_required(self) -> None:
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            create_lichess_client()

    def test_token_session_can_be_closed(self) -> None:
        with patch.dict(os.environ, {"LICHESS_API_TOKEN": "test-token"}, clear=True):
            client, session = create_lichess_client()
        self.assertIs(client.opening_explorer._r.session, session)
        session.close()


class EngineEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate = Candidate(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            line_uci=(),
            side_to_move="black",
            root_games=100,
            terminal_games=0,
        )

    def test_score_is_converted_to_white_pov(self) -> None:
        fake_engine = FakeEngine(
            "Stockfish 17",
            chess.engine.PovScore(chess.engine.Cp(30), chess.BLACK),
        )
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "stockfish.exe"
            binary.write_bytes(b"fake engine")
            with patch.object(
                chess.engine.SimpleEngine,
                "popen_uci",
                return_value=fake_engine,
            ):
                evaluations, metadata = evaluate_candidates(
                    [self.candidate],
                    stockfish_path=binary,
                    nodes=123,
                    threads=1,
                    hash_mb=16,
                )

        self.assertEqual(evaluations[0].pawns, -0.3)
        self.assertIsNone(evaluations[0].mate_in)
        self.assertEqual(metadata["pov"], "white")
        self.assertEqual(metadata["limit"], {"type": "nodes", "value": 123})
        self.assertTrue(fake_engine.closed)

    def test_non_stockfish_17_engine_fails(self) -> None:
        fake_engine = FakeEngine(
            "Stockfish 16",
            chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE),
        )
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "stockfish.exe"
            binary.write_bytes(b"fake engine")
            with patch.object(
                chess.engine.SimpleEngine,
                "popen_uci",
                return_value=fake_engine,
            ), self.assertRaises(RuntimeError):
                evaluate_candidates(
                    [self.candidate],
                    stockfish_path=binary,
                    nodes=123,
                    threads=1,
                    hash_mb=16,
                )
        self.assertTrue(fake_engine.closed)


class OutputTests(unittest.TestCase):
    def test_mismatched_candidate_and_evaluation_counts_fail(self) -> None:
        with self.assertRaises(ValueError):
            build_rows(
                [],
                [object()],  # type: ignore[list-item]
                engine_metadata={},
                seed=1,
                ratings=("1800", "2000"),
                speeds=("blitz", "rapid"),
                fraction=0.01,
                until="2026-09",
                explorer_moves=50,
            )

    def test_atomic_jsonl_writer(self) -> None:
        rows = [{"id": "one"}, {"id": "two"}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "benchmark.jsonl"
            write_jsonl_atomic(rows, output)
            parsed = [json.loads(line) for line in output.read_text().splitlines()]
            leftovers = list(output.parent.glob("*.tmp"))
        self.assertEqual(parsed, rows)
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
