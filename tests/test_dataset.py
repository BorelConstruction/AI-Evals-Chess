from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import chess

from chess_eval import DATASET_PATH, benchmark_dataset
from chess_scoring import win_probability
from sampler import normalized_fen


class BenchmarkDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = [
            json.loads(line)
            for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_has_exactly_fifty_unique_positions(self) -> None:
        self.assertEqual(len(self.rows), 50)
        positions = {
            normalized_fen(chess.Board(row["fen"])) for row in self.rows
        }
        self.assertEqual(len(positions), 50)

    def test_has_balanced_side_to_move(self) -> None:
        sides = [row["side_to_move"] for row in self.rows]
        self.assertEqual(sides.count("white"), 25)
        self.assertEqual(sides.count("black"), 25)

    def test_ids_and_positions_are_valid(self) -> None:
        expected_ids = [f"position-{index:03d}" for index in range(1, 51)]
        self.assertEqual([row["id"] for row in self.rows], expected_ids)
        for row in self.rows:
            with self.subTest(id=row["id"]):
                board = chess.Board(row["fen"])
                self.assertTrue(board.is_valid())
                self.assertFalse(board.is_game_over())
                self.assertEqual(
                    row["side_to_move"],
                    "white" if board.turn == chess.WHITE else "black",
                )

    def test_targets_and_metadata_are_consistent(self) -> None:
        for row in self.rows:
            with self.subTest(id=row["id"]):
                target = float(row["target_pawns"])
                self.assertTrue(math.isfinite(target))
                self.assertAlmostEqual(
                    row["target_win_probability"], win_probability(target)
                )
                self.assertEqual(row["source"]["ratings"], ["1800", "2000"])
                self.assertEqual(row["source"]["speeds"], ["blitz", "rapid"])
                self.assertNotIn("url", row["source"])
                self.assertEqual(row["engine"]["name"], "Stockfish 17")
                self.assertEqual(row["engine"]["limit"], {"type": "nodes", "value": 5_000_000})
                self.assertEqual(row["engine"]["threads"], 1)
                self.assertEqual(row["engine"]["pov"], "white")

    def test_inspect_loader_has_fifty_samples(self) -> None:
        dataset = benchmark_dataset()
        self.assertEqual(len(dataset), 50)
        for sample in dataset:
            self.assertIn(str(sample.metadata["fen"]), str(sample.input))

    def test_token_is_not_stored_in_project_artifacts(self) -> None:
        project_dir = Path(__file__).resolve().parents[1]
        suffixes = {".py", ".json", ".jsonl", ".md", ".txt"}
        for path in project_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                text = path.read_text(encoding="utf-8", errors="ignore")
                with self.subTest(path=path):
                    self.assertNotIn("li" + "p_", text)


if __name__ == "__main__":
    unittest.main()
