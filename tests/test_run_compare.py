from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import run_compare


def make_log(model: str, run_id: str, status: str = "success") -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        eval=SimpleNamespace(model=model, run_id=run_id),
        samples=[],
        location=f"{model}-{run_id}.eval",
    )


def configured_logs(run_id: str = "new-run") -> list[SimpleNamespace]:
    return [make_log(model, run_id) for model in run_compare.MODELS]


class RunCompareTests(unittest.TestCase):
    def test_main_builds_only_the_run_returned_by_eval_set(self) -> None:
        logs = configured_logs()
        report = {"report": "data"}

        with (
            patch.object(
                run_compare, "eval_set", return_value=(True, logs)
            ) as eval_set,
            patch.object(
                run_compare, "build_report_data", return_value=report
            ) as build_report_data,
            patch.object(run_compare, "write_report") as write_report,
        ):
            run_compare.main()

        self.assertIs(eval_set.call_args.args[0], run_compare.chess_eval)
        self.assertIs(eval_set.call_args.kwargs["log_dir_allow_dirty"], True)
        build_report_data.assert_called_once_with(logs)
        write_report.assert_called_once_with(
            report,
            run_compare.PROJECT_DIR / "summary.html",
            run_compare.PROJECT_DIR / "per_position.csv",
        )

    def test_failed_eval_set_cannot_fall_back_to_an_old_report(self) -> None:
        with (
            patch.object(
                run_compare,
                "eval_set",
                return_value=(False, configured_logs("failed-run")),
            ),
            patch.object(run_compare, "build_report_data") as build_report_data,
            patch.object(run_compare, "write_report") as write_report,
        ):
            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                run_compare.main()

        build_report_data.assert_not_called()
        write_report.assert_not_called()

    def test_logs_from_multiple_runs_are_kept_for_resumed_eval_sets(self) -> None:
        logs = configured_logs()
        logs[-1].eval.run_id = "another-run"

        completed = run_compare._completed_logs(True, logs)  # type: ignore[arg-type]

        self.assertEqual(completed, logs)

    def test_header_only_reused_logs_are_loaded_from_their_exact_location(self) -> None:
        logs = configured_logs()
        header = logs[0]
        header.samples = None
        full_log = make_log(header.eval.model, header.eval.run_id)

        with patch.object(
            run_compare, "read_eval_log", return_value=full_log
        ) as read_eval_log:
            completed = run_compare._completed_logs(True, logs)  # type: ignore[arg-type]

        read_eval_log.assert_called_once_with(header.location)
        self.assertIs(completed[0], full_log)


if __name__ == "__main__":
    unittest.main()
