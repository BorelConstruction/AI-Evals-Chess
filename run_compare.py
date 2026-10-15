"""Run the configured model set and build its interactive comparison report."""

from pathlib import Path

from inspect_ai import eval_set
from inspect_ai.log import EvalLog, read_eval_log

from compare import build_report_data, write_report


PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "logs" / "compare-v1"
MODELS = (
    "anthropic/claude-haiku-4-5-20251001",
    "anthropic/claude-sonnet-5-5",
    "anthropic/claude-opus-5-5",
)


def _completed_logs(success: bool, logs: list[EvalLog]) -> list[EvalLog]:
    if not success:
        raise RuntimeError("comparison eval set did not complete successfully")
    if not logs:
        raise RuntimeError("comparison eval set returned no logs")

    returned_models = [log.eval.model for log in logs]
    if not all(isinstance(model, str) and model for model in returned_models):
        raise RuntimeError("comparison eval set returned an invalid model id")

    unsuccessful = [log.eval.model for log in logs if log.status != "success"]
    if unsuccessful:
        raise RuntimeError(
            "comparison eval set returned unsuccessful logs for: "
            + ", ".join(unsuccessful)
        )

    if len(set(returned_models)) != len(returned_models):
        raise RuntimeError("comparison eval set returned duplicate model logs")
    if set(returned_models) != set(MODELS):
        missing = sorted(set(MODELS) - set(returned_models))
        extra = sorted(set(returned_models) - set(MODELS))
        raise RuntimeError(
            f"comparison model set mismatch: missing={missing}, extra={extra}"
        )

    completed_logs: list[EvalLog] = []
    for log in logs:
        run_id = log.eval.run_id
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("comparison logs have no run id")
        if log.samples is not None:
            completed_logs.append(log)
            continue
        if not isinstance(log.location, str) or not log.location:
            raise RuntimeError(f"comparison log for {log.eval.model} has no location")
        completed_logs.append(read_eval_log(log.location))
    return completed_logs


def main() -> None:
    success, logs = eval_set(
        str(PROJECT_DIR / "chess_eval.py"),
        model=list(MODELS),
        log_dir=str(LOG_DIR),
        log_dir_allow_dirty=True,
    )
    report = build_report_data(_completed_logs(success, logs))
    output_path = PROJECT_DIR / "summary.html"
    csv_path = PROJECT_DIR / "per_position.csv"
    write_report(report, output_path, csv_path)
    print(f"Interactive comparison: {output_path}")


if __name__ == "__main__":
    main()
