from __future__ import annotations

import inspect
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess
import chess.engine

from generate_benchmark import (
    Candidate,
    DEFAULT_CUTOFF_FRACTION,
    DEFAULT_TARGET_EVAL_THRESHOLD,
    EngineEvaluation,
    build_rows,
    collect_candidates,
    create_lichess_client,
    evaluate_candidates,
    parse_args,
    write_jsonl_atomic,
)
from sampler import DEFAULT_CUTOFF_FRACTION as SAMPLER_CUTOFF_FRACTION


def sampled_position(*moves: str) -> tuple[list[str], str]:
    board = chess.Board()
    for move_text in moves:
        board.push_uci(move_text)
    return list(moves), board.fen()


class FakeEngine:
    def __init__(self, name: str, score: chess.engine.PovScore) -> None:
        self.id = {"name": name, "author": "test"}
        self.options: dict[str, object] = {}
        self.score = score
        self.configurations: list[dict[str, object]] = []
        self.analysed_fens: list[str] = []
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
        self.analysed_fens.append(board.fen())
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


class CandidateCollectionTests(unittest.TestCase):
    def collect_from(
        self,
        samples: list[tuple[list[str], str]],
        evaluations_by_fen: dict[str, float],
        *,
        target_eval_threshold: float = DEFAULT_TARGET_EVAL_THRESHOLD,
        max_candidates: int | None = None,
    ) -> tuple[
        list[Candidate],
        list[EngineEvaluation],
        list[str],
        int,
    ]:
        evaluated_fens: list[str] = []

        def evaluate_candidate(candidate: Candidate) -> EngineEvaluation:
            evaluated_fens.append(candidate.fen)
            return EngineEvaluation(
                pawns=evaluations_by_fen[candidate.fen],
                mate_in=None,
            )

        with patch(
            "generate_benchmark.sample_line", side_effect=samples
        ) as sample_line_mock, patch(
            "generate_benchmark._cache_total", return_value=1
        ), patch("builtins.print"):
            candidates, evaluations = collect_candidates(
                object(),  # type: ignore[arg-type]
                evaluate_candidate=evaluate_candidate,
                seed=1,
                count=2,
                ratings=("1800", "2000"),
                speeds=("blitz", "rapid"),
                fraction=DEFAULT_CUTOFF_FRACTION,
                until="2026-09",
                explorer_moves=50,
                request_delay=0,
                max_candidates=(
                    len(samples) if max_candidates is None else max_candidates
                ),
                target_eval_threshold=target_eval_threshold,
            )

        return candidates, evaluations, evaluated_fens, sample_line_mock.call_count

    def test_default_threshold_is_one_pawn(self) -> None:
        default = inspect.signature(collect_candidates).parameters[
            "target_eval_threshold"
        ].default

        self.assertEqual(default, 1.0)

    def test_strict_absolute_threshold_refills_until_balanced(self) -> None:
        white_boundary = sampled_position("e2e4", "e7e5")
        black_boundary = sampled_position("e2e4")
        white_accepted = sampled_position("d2d4", "d7d5")
        black_accepted = sampled_position("d2d4")
        samples = [
            white_boundary,
            black_boundary,
            white_accepted,
            black_accepted,
        ]
        evaluation_values = {
            white_boundary[1]: 1.0,
            black_boundary[1]: -1.0,
            white_accepted[1]: -1.01,
            black_accepted[1]: 1.01,
        }

        candidates, evaluations, evaluated_fens, sample_count = self.collect_from(
            samples, evaluation_values
        )

        self.assertEqual(
            [candidate.side_to_move for candidate in candidates],
            ["white", "black"],
        )
        self.assertEqual(
            [evaluation.pawns for evaluation in evaluations],
            [-1.01, 1.01],
        )
        self.assertEqual(evaluated_fens, [sample[1] for sample in samples])
        self.assertEqual(sample_count, 4)

    def test_zero_threshold_rejects_only_zero(self) -> None:
        zero = sampled_position("e2e4", "e7e5")
        positive = sampled_position("d2d4", "d7d5")
        negative = sampled_position("e2e4")
        samples = [zero, positive, negative]
        evaluation_values = {
            zero[1]: 0.0,
            positive[1]: 0.01,
            negative[1]: -0.01,
        }

        _, evaluations, _, sample_count = self.collect_from(
            samples,
            evaluation_values,
            target_eval_threshold=0.0,
        )

        self.assertEqual(
            [evaluation.pawns for evaluation in evaluations],
            [0.01, -0.01],
        )
        self.assertEqual(sample_count, 3)

    def test_rejected_duplicates_and_full_side_are_not_reevaluated(self) -> None:
        rejected = sampled_position("e2e4", "e7e5")
        white_accepted = sampled_position("d2d4", "d7d5")
        white_after_quota = sampled_position("c2c4", "e7e5")
        black_accepted = sampled_position("e2e4")
        samples = [
            rejected,
            rejected,
            white_accepted,
            white_after_quota,
            black_accepted,
        ]
        evaluation_values = {
            rejected[1]: 0.5,
            white_accepted[1]: 1.5,
            white_after_quota[1]: 2.0,
            black_accepted[1]: -1.5,
        }

        candidates, _, evaluated_fens, sample_count = self.collect_from(
            samples, evaluation_values
        )

        self.assertEqual(
            evaluated_fens,
            [rejected[1], white_accepted[1], black_accepted[1]],
        )
        self.assertEqual(
            [candidate.fen for candidate in candidates],
            [white_accepted[1], black_accepted[1]],
        )
        self.assertEqual(sample_count, 5)

    def test_attempt_limit_includes_threshold_rejections(self) -> None:
        white = sampled_position("e2e4", "e7e5")
        black = sampled_position("e2e4")
        evaluated_fens: list[str] = []

        def evaluate_candidate(candidate: Candidate) -> EngineEvaluation:
            evaluated_fens.append(candidate.fen)
            return EngineEvaluation(pawns=0.5, mate_in=None)

        with patch(
            "generate_benchmark.sample_line", side_effect=[white, black]
        ) as sample_line_mock, patch(
            "generate_benchmark._cache_total", return_value=1
        ), patch("builtins.print"), self.assertRaisesRegex(
            RuntimeError, "after 2 sampling attempts"
        ):
            collect_candidates(
                object(),  # type: ignore[arg-type]
                evaluate_candidate=evaluate_candidate,
                seed=1,
                count=2,
                ratings=("1800", "2000"),
                speeds=("blitz", "rapid"),
                fraction=DEFAULT_CUTOFF_FRACTION,
                until="2026-09",
                explorer_moves=50,
                request_delay=0,
                max_candidates=2,
            )

        self.assertEqual(sample_line_mock.call_count, 2)
        self.assertEqual(evaluated_fens, [white[1], black[1]])

    def test_attempt_limit_includes_game_over_positions(self) -> None:
        checkmate = sampled_position("f2f3", "e7e5", "g2g4", "d8h4")
        accepted = sampled_position("e2e4", "e7e5")
        evaluated_fens: list[str] = []

        def evaluate_candidate(candidate: Candidate) -> EngineEvaluation:
            evaluated_fens.append(candidate.fen)
            return EngineEvaluation(pawns=2.0, mate_in=None)

        with patch(
            "generate_benchmark.sample_line", side_effect=[checkmate, accepted]
        ) as sample_line_mock, patch(
            "generate_benchmark._cache_total", return_value=1
        ), patch("builtins.print"), self.assertRaisesRegex(
            RuntimeError, "after 2 sampling attempts"
        ):
            collect_candidates(
                object(),  # type: ignore[arg-type]
                evaluate_candidate=evaluate_candidate,
                seed=1,
                count=2,
                ratings=("1800", "2000"),
                speeds=("blitz", "rapid"),
                fraction=DEFAULT_CUTOFF_FRACTION,
                until="2026-09",
                explorer_moves=50,
                request_delay=0,
                max_candidates=2,
            )

        self.assertEqual(sample_line_mock.call_count, 2)
        self.assertEqual(evaluated_fens, [accepted[1]])

    def test_invalid_thresholds_fail_before_sampling(self) -> None:
        invalid_values = (
            (True, TypeError),
            ("1", TypeError),
            (-0.01, ValueError),
            (math.nan, ValueError),
            (math.inf, ValueError),
            (-math.inf, ValueError),
        )
        for value, error_type in invalid_values:
            with self.subTest(value=value), patch(
                "generate_benchmark.sample_line"
            ) as sample_line_mock, self.assertRaises(error_type):
                collect_candidates(
                    object(),  # type: ignore[arg-type]
                    evaluate_candidate=lambda candidate: EngineEvaluation(2.0, None),
                    seed=1,
                    count=2,
                    ratings=("1800", "2000"),
                    speeds=("blitz", "rapid"),
                    fraction=DEFAULT_CUTOFF_FRACTION,
                    until="2026-09",
                    explorer_moves=50,
                    request_delay=0,
                    max_candidates=2,
                    target_eval_threshold=value,  # type: ignore[arg-type]
                )
            sample_line_mock.assert_not_called()


class ArgumentTests(unittest.TestCase):
    def test_target_eval_threshold_cli_default_and_override(self) -> None:
        self.assertEqual(
            parse_args([]).target_eval_threshold,
            DEFAULT_TARGET_EVAL_THRESHOLD,
        )
        self.assertEqual(
            parse_args(["--target-eval-threshold", "2.5"]).target_eval_threshold,
            2.5,
        )


class OutputTests(unittest.TestCase):
    def test_generator_uses_sampler_cutoff_configuration(self) -> None:
        self.assertEqual(DEFAULT_CUTOFF_FRACTION, SAMPLER_CUTOFF_FRACTION)

    def test_rows_record_fraction_without_redundant_url(self) -> None:
        candidate = Candidate(
            fen=chess.STARTING_FEN,
            line_uci=(),
            side_to_move="white",
            root_games=1_000,
            terminal_games=0,
        )
        rows = build_rows(
            [candidate],
            [EngineEvaluation(pawns=0.0, mate_in=None)],
            engine_metadata={},
            seed=1,
            ratings=("1800", "2000"),
            speeds=("blitz", "rapid"),
            fraction=DEFAULT_CUTOFF_FRACTION,
            until="2026-09",
            explorer_moves=50,
        )

        source = rows[0]["source"]
        self.assertEqual(source["cutoff_fraction"], DEFAULT_CUTOFF_FRACTION)
        self.assertNotIn("url", source)

    def test_mismatched_candidate_and_evaluation_counts_fail(self) -> None:
        with self.assertRaises(ValueError):
            build_rows(
                [],
                [object()],  # type: ignore[list-item]
                engine_metadata={},
                seed=1,
                ratings=("1800", "2000"),
                speeds=("blitz", "rapid"),
                fraction=DEFAULT_CUTOFF_FRACTION,
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
