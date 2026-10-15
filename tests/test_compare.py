from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import chess

import compare as compare_module
from chess_scoring import REASONING_FIELD, RESPONSE_FIELD, win_probability
from compare import (
    SCORER_NAME,
    _json_for_html,
    _mean_sem,
    build_report_data,
    extract_rationale,
    render_report,
)


def make_sample(
    sample_id: str,
    *,
    fen: str = chess.STARTING_FEN,
    target_pawns: float = 0.0,
    completion: str | None = None,
    content: list[object] | None = None,
) -> SimpleNamespace:
    board = chess.Board(fen)
    if completion is None:
        completion = json.dumps({RESPONSE_FIELD: target_pawns})
    predicted_probability = win_probability(target_pawns)
    score = SimpleNamespace(
        value=1.0,
        metadata={
            "predicted_pawns": target_pawns,
            "predicted_win_probability": predicted_probability,
            "absolute_probability_error": 0.0,
        },
        reason=None,
        explanation="Exact match.",
    )
    usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        reasoning_tokens=2,
    )
    output = SimpleNamespace(
        completion=completion,
        message=SimpleNamespace(content=[] if content is None else content),
        usage=usage,
    )
    metadata = {
        "fen": fen,
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "target_win_probability": win_probability(target_pawns),
        "mate_in": None,
        "line_uci": [],
        "engine": {
            "name": "Stockfish 17",
            "limit": {"type": "nodes", "value": 5_000_000},
            "threads": 1,
            "hash_mb": 256,
            "pov": "white",
        },
        "source": {
            "ratings": ["1800", "2000"],
            "speeds": ["blitz", "rapid"],
            "until": "2026-09",
            "cutoff_fraction": 0.001,
            "root_games": 1_000,
            "terminal_games": 0,
            "url": "https://explorer.lichess.org",
        },
    }
    return SimpleNamespace(
        id=sample_id,
        epoch=1,
        target=str(target_pawns),
        metadata=metadata,
        scores={SCORER_NAME: score},
        output=output,
        error=None,
        total_time=0.25,
    )


def make_log(
    model: str,
    samples: list[SimpleNamespace],
    *,
    run_id: str = "test-run",
) -> SimpleNamespace:
    return SimpleNamespace(
        eval=SimpleNamespace(
            model=model,
            run_id=run_id,
            eval_set_id="test-eval-set",
            created="2026-10-13T12:00:00+00:00",
        ),
        samples=samples,
    )


class HtmlSerializationTests(unittest.TestCase):
    def test_json_for_html_escapes_script_breakout_and_js_separators(self) -> None:
        data = {
            "text": "</script><script>alert('x')</script>&\N{LINE SEPARATOR}"
            "next\N{PARAGRAPH SEPARATOR}line"
        }

        serialized = _json_for_html(data)

        self.assertNotIn("</script", serialized.lower())
        self.assertNotIn("&", serialized)
        self.assertNotIn("\N{LINE SEPARATOR}", serialized)
        self.assertNotIn("\N{PARAGRAPH SEPARATOR}", serialized)
        self.assertIn(r"\u003c/script\u003e", serialized)
        self.assertIn(r"\u0026", serialized)
        self.assertIn(r"\u2028", serialized)
        self.assertIn(r"\u2029", serialized)
        self.assertEqual(json.loads(serialized), data)

    def test_render_report_injects_data_into_one_template_placeholder(self) -> None:
        data = {"model": "example/model", "score": 0.75}
        with tempfile.TemporaryDirectory() as directory:
            template_path = Path(directory) / "template.html"
            template_path.write_text(
                "<html><script>const report = __REPORT_DATA__;</script></html>",
                encoding="utf-8",
            )

            rendered = render_report(data, template_path)

        self.assertNotIn("__REPORT_DATA__", rendered)
        self.assertIn(
            '<script>const report = {"model":"example/model","score":0.75};</script>',
            rendered,
        )

    def test_render_report_rejects_ambiguous_template_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template_path = Path(directory) / "template.html"
            template_path.write_text(
                "__REPORT_DATA__ and __REPORT_DATA__", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "exactly one"):
                render_report({}, template_path)


class RationaleExtractionTests(unittest.TestCase):
    def test_extracts_structured_user_visible_reasoning(self) -> None:
        completion = json.dumps(
            {
                REASONING_FIELD: "  White controls the open file.  ",
                RESPONSE_FIELD: 0.4,
            }
        )
        sample = make_sample("position-001", completion=completion)

        rationale, status = extract_rationale(sample)

        self.assertEqual(rationale, "White controls the open file.")
        self.assertEqual(status, "available")

    def test_extracts_visible_prose_before_trailing_evaluation_json(self) -> None:
        prose = "White has more space and the safer king in this position."
        completion = f'{prose}\n\n{{"{RESPONSE_FIELD}": 0.35}}'
        sample = make_sample("position-001", completion=completion)

        rationale, status = extract_rationale(sample)

        self.assertEqual(rationale, prose)
        self.assertEqual(status, "available")

    def test_redacted_provider_reasoning_never_returns_ciphertext(self) -> None:
        ciphertext = "opaque-provider-ciphertext-must-not-leak"
        reasoning_block = SimpleNamespace(
            type="reasoning",
            reasoning=ciphertext,
            summary="",
            redacted=True,
        )
        sample = make_sample("position-001", content=[reasoning_block])

        rationale, status = extract_rationale(sample)

        self.assertIsNone(rationale)
        self.assertEqual(status, "provider_redacted")
        self.assertNotIn(ciphertext, rationale or "")


class StatisticsTests(unittest.TestCase):
    def test_mean_sem_handles_empty_singleton_and_multiple_values(self) -> None:
        self.assertEqual(_mean_sem([]), (None, None))
        self.assertEqual(_mean_sem([2.5]), (2.5, None))

        mean, sem = _mean_sem([1.0, 3.0])

        self.assertEqual(mean, 2.0)
        self.assertAlmostEqual(sem, 1.0)


class ReportDataValidationTests(unittest.TestCase):
    def test_successful_report_keeps_details_without_leaking_stale_data(self) -> None:
        rationale = "White has more central space and the safer king."
        ciphertext = "opaque-provider-ciphertext-must-not-leak"
        visible_sample = make_sample(
            "position-001",
            completion=json.dumps(
                {REASONING_FIELD: rationale, RESPONSE_FIELD: 0.0}
            ),
        )
        redacted_sample = make_sample(
            "position-001",
            content=[
                SimpleNamespace(
                    type="reasoning",
                    reasoning=ciphertext,
                    summary="",
                    redacted=True,
                )
            ],
        )

        data = build_report_data(
            [
                make_log("provider/model-a", [visible_sample]),
                make_log("provider/model-b", [redacted_sample]),
            ]
        )
        rendered = render_report(data)

        position = data["positions"][0]
        self.assertNotIn("url", position["source"])
        self.assertEqual(
            position["models"]["provider/model-b"]["reasoning_status"],
            "provider_redacted",
        )
        for expected in (chess.STARTING_FEN, "Stockfish 17", rationale):
            self.assertIn(expected, rendered)
        self.assertNotIn("https://explorer.lichess.org", rendered)
        self.assertNotIn(ciphertext, rendered)

    def test_resumed_eval_set_can_combine_exact_logs_from_multiple_runs(self) -> None:
        data = build_report_data(
            [
                make_log(
                    "provider/model-a",
                    [make_sample("position-001")],
                    run_id="earlier-run",
                ),
                make_log(
                    "provider/model-b",
                    [make_sample("position-001")],
                    run_id="new-run",
                ),
            ]
        )

        self.assertEqual(data["meta"]["run_id"], "combined")
        self.assertEqual(data["meta"]["run_ids"], ["earlier-run", "new-run"])
        self.assertIn(
            '"run_ids":["earlier-run","new-run"]', render_report(data)
        )

    def test_rejects_inconsistent_fen_across_models(self) -> None:
        after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        logs = [
            make_log("provider/model-a", [make_sample("position-001")]),
            make_log(
                "provider/model-b",
                [make_sample("position-001", fen=after_e4)],
            ),
        ]

        with self.assertRaisesRegex(ValueError, "inconsistent fen"):
            build_report_data(logs)

    def test_rejects_engine_metadata_the_browser_cannot_render(self) -> None:
        sample = make_sample("position-001")
        sample.metadata["engine"].pop("limit")

        with self.assertRaisesRegex(TypeError, "engine limit"):
            build_report_data([make_log("provider/model-a", [sample])])

    def test_rejects_source_metadata_the_browser_cannot_render(self) -> None:
        sample = make_sample("position-001")
        sample.metadata["source"].pop("ratings")

        with self.assertRaisesRegex(TypeError, "source ratings"):
            build_report_data([make_log("provider/model-a", [sample])])

    def test_rejects_inconsistent_target_across_models(self) -> None:
        logs = [
            make_log(
                "provider/model-a",
                [make_sample("position-001", target_pawns=0.0)],
            ),
            make_log(
                "provider/model-b",
                [make_sample("position-001", target_pawns=0.25)],
            ),
        ]

        with self.assertRaisesRegex(ValueError, "inconsistent target_pawns"):
            build_report_data(logs)

    def test_rejects_different_sample_sets(self) -> None:
        logs = [
            make_log(
                "provider/model-a",
                [make_sample("position-001"), make_sample("position-002")],
            ),
            make_log("provider/model-b", [make_sample("position-001")]),
        ]

        with self.assertRaisesRegex(ValueError, "sample set mismatch"):
            build_report_data(logs)


class ImportSafetyTests(unittest.TestCase):
    def test_import_does_not_write_files_to_working_directory(self) -> None:
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            temporary_directory = Path(directory)
            try:
                os.chdir(temporary_directory)
                importlib.reload(compare_module)
            finally:
                os.chdir(original_directory)

            self.assertEqual(list(temporary_directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
