"""Build a self-contained interactive comparison report from Inspect logs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import statistics
import tempfile
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import chess
from inspect_ai.log import EvalLog, EvalSample, list_eval_logs, read_eval_log

from chess_scoring import REASONING_FIELD, RESPONSE_FIELD, win_probability


SCORER_NAME = "winning_probability_similarity"
TEMPLATE_PATH = Path(__file__).resolve().with_name("comparison_template.html")
_FENCED_JSON_RE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<body>.*)\r?\n```[ \t]*\Z",
    re.IGNORECASE | re.DOTALL,
)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{field} must be a number") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _optional_finite_number(value: Any, field: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, field)


def _optional_nonnegative_integer(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} cannot be negative")
    return value


def _required_integer(value: Any, field: str, *, minimum: int = 0) -> int:
    integer = _optional_nonnegative_integer(value, field)
    if integer is None:
        raise TypeError(f"{field} must be an integer")
    if integer < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return integer


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be non-empty text")
    return value


def _required_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise TypeError(f"{field} must be a non-empty list of text values")
    return list(value)


def _mean_sem(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, None
    return mean, statistics.stdev(values) / math.sqrt(len(values))


def _json_response(completion: str) -> tuple[dict[str, Any] | None, str]:
    """Return the evaluation JSON object and any visible text around it."""

    text = completion.strip()
    fence_match = _FENCED_JSON_RE.fullmatch(text)
    if fence_match is not None:
        text = fence_match.group("body").strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and RESPONSE_FIELD in payload:
        return payload, ""

    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            candidate, length = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, dict) or RESPONSE_FIELD not in candidate:
            continue
        visible_text = f"{text[:start]}\n{text[start + length:]}".strip()
        return candidate, visible_text
    return None, text


def extract_rationale(sample: EvalSample) -> tuple[str | None, str]:
    """Extract only user-visible rationale; never expose redacted provider data."""

    output = sample.output
    completion = output.completion if output is not None else ""
    payload, surrounding_text = _json_response(completion or "")
    if payload is not None:
        structured = payload.get(REASONING_FIELD)
        if isinstance(structured, str) and structured.strip():
            return structured.strip(), "available"

    visible_provider_reasoning: list[str] = []
    provider_redacted = False
    message = output.message if output is not None else None
    content = message.content if message is not None else None
    if isinstance(content, list):
        for block in content:
            if getattr(block, "type", None) != "reasoning":
                continue
            summary = getattr(block, "summary", None)
            if isinstance(summary, str) and summary.strip():
                visible_provider_reasoning.append(summary.strip())
            if getattr(block, "redacted", False):
                provider_redacted = True
                continue
            reasoning = getattr(block, "reasoning", None)
            if isinstance(reasoning, str) and reasoning.strip():
                visible_provider_reasoning.append(reasoning.strip())

    if visible_provider_reasoning:
        return "\n\n".join(visible_provider_reasoning), "available"
    if surrounding_text and ("\n" in surrounding_text or len(surrounding_text.split()) > 6):
        return surrounding_text, "available"
    if provider_redacted:
        return None, "provider_redacted"
    return None, "not_returned"


def _usage_fields(sample: EvalSample) -> dict[str, int | None]:
    usage = sample.output.usage if sample.output is not None else None
    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "reasoning_tokens": None,
        }
    return {
        field: _optional_nonnegative_integer(getattr(usage, field, None), field)
        for field in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "reasoning_tokens",
        )
    }


def _sample_error_text(sample: EvalSample) -> str | None:
    if sample.error is None:
        return None
    message = getattr(sample.error, "message", None)
    if isinstance(message, str) and message:
        return message
    return str(sample.error)


def _model_result(sample: EvalSample, model_id: str) -> dict[str, Any]:
    scores = sample.scores or {}
    score = scores.get(SCORER_NAME)
    sample_error = _sample_error_text(sample)
    if score is None:
        detail = f": {sample_error}" if sample_error else ""
        raise ValueError(
            f"{model_id}/{sample.id} has no {SCORER_NAME!r} score{detail}"
        )

    score_metadata: dict[str, Any] = {}
    reason = sample_error
    score_value = _finite_number(score.value, f"{model_id}/{sample.id} score")
    if score.metadata is not None:
        if not isinstance(score.metadata, dict):
            raise TypeError(f"{model_id}/{sample.id} score metadata must be an object")
        score_metadata = score.metadata
    reason = score.reason or reason
    explanation = score.explanation
    if reason is not None and not isinstance(reason, str):
        raise TypeError(f"{model_id}/{sample.id} score reason must be text or null")
    if explanation is not None and not isinstance(explanation, str):
        raise TypeError(f"{model_id}/{sample.id} score explanation must be text or null")

    predicted_pawns = _optional_finite_number(
        score_metadata.get("predicted_pawns"),
        f"{model_id}/{sample.id} predicted_pawns",
    )
    predicted_probability = _optional_finite_number(
        score_metadata.get("predicted_win_probability"),
        f"{model_id}/{sample.id} predicted_win_probability",
    )
    if predicted_pawns is not None and predicted_probability is None:
        predicted_probability = win_probability(predicted_pawns)

    probability_error = _optional_finite_number(
        score_metadata.get("absolute_probability_error"),
        f"{model_id}/{sample.id} absolute_probability_error",
    )
    target_pawns = _finite_number(sample.target, f"{sample.id} target")
    if predicted_probability is not None and probability_error is None:
        probability_error = abs(
            predicted_probability - win_probability(target_pawns)
        )
    pawn_error = (
        abs(predicted_pawns - target_pawns) if predicted_pawns is not None else None
    )
    rationale, reasoning_status = extract_rationale(sample)
    completion = sample.output.completion if sample.output is not None else ""
    usage = _usage_fields(sample)

    result: dict[str, Any] = {
        "predicted_pawns": predicted_pawns,
        "predicted_probability": predicted_probability,
        "score": score_value,
        "probability_error": probability_error,
        "pawn_error": pawn_error,
        "completion": completion or "",
        "rationale": rationale,
        "reasoning_status": reasoning_status,
        "total_time": _optional_finite_number(
            sample.total_time, f"{model_id}/{sample.id} total_time"
        ),
        "valid": predicted_pawns is not None and reason is None,
        "reason": reason,
        "explanation": explanation,
    }
    result.update(usage)
    return result


def _metadata_object(sample: EvalSample) -> dict[str, Any]:
    metadata = sample.metadata
    if not isinstance(metadata, dict):
        raise TypeError(f"{sample.id} metadata must be an object")
    return metadata


def _whitelist_engine(metadata: dict[str, Any], sample_id: str) -> dict[str, Any]:
    engine = metadata.get("engine")
    if not isinstance(engine, dict):
        raise TypeError(f"{sample_id} has no engine metadata")

    limit = engine.get("limit")
    if not isinstance(limit, dict):
        raise TypeError(f"{sample_id} engine limit must be an object")
    limit_value = _finite_number(
        limit.get("value"), f"{sample_id} engine limit value"
    )
    if limit_value <= 0:
        raise ValueError(f"{sample_id} engine limit value must be positive")

    return {
        "name": _required_string(engine.get("name"), f"{sample_id} engine name"),
        "limit": {
            "type": _required_string(
                limit.get("type"), f"{sample_id} engine limit type"
            ),
            "value": limit_value,
        },
        "threads": _required_integer(
            engine.get("threads"), f"{sample_id} engine threads", minimum=1
        ),
        "hash_mb": _required_integer(
            engine.get("hash_mb"), f"{sample_id} engine hash_mb", minimum=1
        ),
        "pov": _required_string(engine.get("pov"), f"{sample_id} engine pov"),
    }


def _whitelist_source(metadata: dict[str, Any], sample_id: str) -> dict[str, Any]:
    source = metadata.get("source")
    if not isinstance(source, dict):
        raise TypeError(f"{sample_id} has no source metadata")

    until = source.get("until")
    if until is not None and not isinstance(until, str):
        raise TypeError(f"{sample_id} source until must be text or null")
    cutoff_fraction = _finite_number(
        source.get("cutoff_fraction"), f"{sample_id} source cutoff_fraction"
    )
    if not 0.0 < cutoff_fraction < 1.0:
        raise ValueError(
            f"{sample_id} source cutoff_fraction must be strictly between 0 and 1"
        )

    return {
        "ratings": _required_string_list(
            source.get("ratings"), f"{sample_id} source ratings"
        ),
        "speeds": _required_string_list(
            source.get("speeds"), f"{sample_id} source speeds"
        ),
        "until": until,
        "cutoff_fraction": cutoff_fraction,
        "root_games": _required_integer(
            source.get("root_games"), f"{sample_id} source root_games", minimum=1
        ),
        "terminal_games": _required_integer(
            source.get("terminal_games"),
            f"{sample_id} source terminal_games",
        ),
    }


def _position_data(sample: EvalSample) -> dict[str, Any]:
    if not isinstance(sample.id, str) or not sample.id:
        raise TypeError("sample id must be non-empty text")
    if sample.epoch != 1:
        raise ValueError(f"comparison reports currently require epoch 1, got {sample.epoch}")

    metadata = _metadata_object(sample)
    fen = metadata.get("fen")
    if not isinstance(fen, str):
        raise TypeError(f"{sample.id} FEN must be text")
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError(f"{sample.id} has an invalid FEN")
    side_to_move = "white" if board.turn == chess.WHITE else "black"
    if metadata.get("side_to_move") != side_to_move:
        raise ValueError(f"{sample.id} side_to_move does not match its FEN")

    target_pawns = _finite_number(sample.target, f"{sample.id} target")
    target_probability = _finite_number(
        metadata.get("target_win_probability"),
        f"{sample.id} target_win_probability",
    )
    expected_probability = win_probability(target_pawns)
    if not math.isclose(target_probability, expected_probability, abs_tol=1e-12):
        raise ValueError(f"{sample.id} target probability does not match its target")

    mate_in = metadata.get("mate_in")
    if mate_in is not None and (isinstance(mate_in, bool) or not isinstance(mate_in, int)):
        raise TypeError(f"{sample.id} mate_in must be an integer or null")
    line_uci = metadata.get("line_uci", [])
    if not isinstance(line_uci, list) or not all(isinstance(move, str) for move in line_uci):
        raise TypeError(f"{sample.id} line_uci must be a list of strings")

    return {
        "id": sample.id,
        "fen": fen,
        "side_to_move": side_to_move,
        "target_pawns": target_pawns,
        "target_probability": target_probability,
        "mate_in": mate_in,
        "line_uci": list(line_uci),
        "engine": _whitelist_engine(metadata, sample.id),
        "source": _whitelist_source(metadata, sample.id),
    }


def _assert_same_position(existing: dict[str, Any], candidate: dict[str, Any]) -> None:
    fields = (
        "fen",
        "side_to_move",
        "target_pawns",
        "target_probability",
        "mate_in",
        "line_uci",
        "engine",
        "source",
    )
    for field in fields:
        if existing[field] != candidate[field]:
            raise ValueError(
                f"inconsistent {field} across model logs for {existing['id']}"
            )


def _select_logs(log_dir: Path, run_id: str | None) -> list[EvalLog]:
    if not log_dir.is_dir():
        raise FileNotFoundError(f"comparison log directory not found: {log_dir}")
    infos = list_eval_logs(str(log_dir))
    if not infos:
        raise FileNotFoundError(f"no Inspect eval logs found in {log_dir}")

    groups: dict[str, list[tuple[Any, EvalLog]]] = {}
    for info in infos:
        header = read_eval_log(info, header_only=True)
        header_run_id = header.eval.run_id
        if not isinstance(header_run_id, str) or not header_run_id:
            raise ValueError(f"eval log has no run id: {info}")
        groups.setdefault(header_run_id, []).append((info, header))

    selected_run_id = run_id
    if selected_run_id is None:
        selected_run_id = max(
            groups,
            key=lambda candidate: max(
                str(header.eval.created) for _, header in groups[candidate]
            ),
        )
    if selected_run_id not in groups:
        available = ", ".join(sorted(groups))
        raise ValueError(
            f"run id {selected_run_id!r} was not found; available run ids: {available}"
        )

    selected = groups[selected_run_id]
    failed = [str(info) for info, header in selected if header.status != "success"]
    if failed:
        raise RuntimeError(
            f"selected comparison run contains non-successful logs: {', '.join(failed)}"
        )

    logs = [read_eval_log(info) for info, _ in selected]
    model_ids = [log.eval.model for log in logs]
    duplicates = sorted({model for model in model_ids if model_ids.count(model) > 1})
    if duplicates:
        raise ValueError(
            "selected run contains duplicate model logs: " + ", ".join(duplicates)
        )
    return sorted(logs, key=lambda log: log.eval.model)


def _model_summary(
    model_id: str, positions: list[dict[str, Any]]
) -> dict[str, Any]:
    results = [position["models"][model_id] for position in positions]
    scores = [result["score"] for result in results if result["score"] is not None]
    probability_errors = [
        result["probability_error"]
        for result in results
        if result["probability_error"] is not None
    ]
    pawn_errors = [
        result["pawn_error"] for result in results if result["pawn_error"] is not None
    ]
    times = [
        result["total_time"] for result in results if result["total_time"] is not None
    ]
    mean_score, sem_score = _mean_sem(scores)
    mean_probability_error, _ = _mean_sem(probability_errors)
    mean_pawn_error, _ = _mean_sem(pawn_errors)
    average_time, _ = _mean_sem(times)
    token_counts = [
        result["total_tokens"]
        for result in results
        if result["total_tokens"] is not None
    ]
    return {
        "id": model_id,
        "label": model_id.split("/", 1)[-1],
        "mean_score": mean_score,
        "sem_score": sem_score,
        "mean_probability_error": mean_probability_error,
        "mean_pawn_error": mean_pawn_error,
        "invalid_count": sum(not result["valid"] for result in results),
        "rationale_count": sum(
            result["reasoning_status"] == "available" for result in results
        ),
        "redacted_count": sum(
            result["reasoning_status"] == "provider_redacted" for result in results
        ),
        "average_time": average_time,
        "total_tokens": sum(token_counts) if token_counts else None,
    }


def _pairwise_rows(
    model_ids: list[str], positions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_a, model_b in combinations(model_ids, 2):
        deltas = []
        for position in positions:
            score_a = position["models"][model_a]["score"]
            score_b = position["models"][model_b]["score"]
            if score_a is not None and score_b is not None:
                deltas.append(score_a - score_b)
        mean_delta, sem_delta = _mean_sem(deltas)
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "mean_delta": mean_delta,
                "sem_delta": sem_delta,
                "count": len(deltas),
            }
        )
    return rows


def build_report_data(logs: list[EvalLog]) -> dict[str, Any]:
    if not logs:
        raise ValueError("at least one eval log is required")

    positions_by_id: dict[str, dict[str, Any]] = {}
    ids_by_model: dict[str, set[str]] = {}
    for log in sorted(logs, key=lambda item: item.eval.model):
        model_id = log.eval.model
        if not isinstance(model_id, str) or not model_id:
            raise TypeError("log model id must be non-empty text")
        if log.samples is None:
            raise ValueError(f"log for {model_id} contains no samples")
        model_ids: set[str] = set()
        for sample in log.samples:
            if sample.id in model_ids:
                raise ValueError(f"duplicate sample {sample.id} in log for {model_id}")
            model_ids.add(sample.id)
            candidate = _position_data(sample)
            position = positions_by_id.get(sample.id)
            if position is None:
                position = candidate
                position["models"] = {}
                positions_by_id[sample.id] = position
            else:
                _assert_same_position(position, candidate)
            position["models"][model_id] = _model_result(sample, model_id)
        ids_by_model[model_id] = model_ids

    expected_ids = next(iter(ids_by_model.values()))
    for model_id, sample_ids in ids_by_model.items():
        if sample_ids != expected_ids:
            missing = sorted(expected_ids - sample_ids)
            extra = sorted(sample_ids - expected_ids)
            raise ValueError(
                f"sample set mismatch for {model_id}: missing={missing}, extra={extra}"
            )

    model_ids = sorted(ids_by_model)
    positions = [positions_by_id[sample_id] for sample_id in sorted(expected_ids)]
    for position in positions:
        predicted = [
            result["predicted_pawns"]
            for result in position["models"].values()
            if result["predicted_pawns"] is not None
        ]
        scores = [
            result["score"]
            for result in position["models"].values()
            if result["score"] is not None
        ]
        position["disagreement"] = max(predicted) - min(predicted) if predicted else None
        position["worst_score"] = min(scores) if scores else None

    run_ids = {log.eval.run_id for log in logs}
    eval_set_ids = {log.eval.eval_set_id for log in logs}
    if (
        not run_ids
        or not all(isinstance(run_id, str) and run_id for run_id in run_ids)
        or len(eval_set_ids) != 1
    ):
        raise ValueError("all comparison logs must belong to one eval set")
    eval_set_id = next(iter(eval_set_ids))
    if not isinstance(eval_set_id, str) or not eval_set_id:
        raise ValueError("comparison logs have no eval set id")
    sorted_run_ids = sorted(run_ids)
    created = min(str(log.eval.created) for log in logs)
    return {
        "meta": {
            "run_id": sorted_run_ids[0] if len(sorted_run_ids) == 1 else "combined",
            "run_ids": sorted_run_ids,
            "eval_set_id": eval_set_id,
            "created": created,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "position_count": len(positions),
            "model_count": len(model_ids),
        },
        "models": [_model_summary(model_id, positions) for model_id in model_ids],
        "pairwise": _pairwise_rows(model_ids, positions),
        "positions": positions,
    }


def load_report_data(log_dir: Path, run_id: str | None = None) -> dict[str, Any]:
    return build_report_data(_select_logs(log_dir, run_id))


def _json_for_html(data: dict[str, Any]) -> str:
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return (
        serialized.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\N{LINE SEPARATOR}", "\\u2028")
        .replace("\N{PARAGRAPH SEPARATOR}", "\\u2029")
    )


def render_report(data: dict[str, Any], template_path: Path = TEMPLATE_PATH) -> str:
    template = template_path.read_text(encoding="utf-8")
    placeholder = "__REPORT_DATA__"
    if template.count(placeholder) != 1:
        raise ValueError(f"report template must contain exactly one {placeholder} placeholder")
    return template.replace(placeholder, _json_for_html(data))


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _csv_text(data: dict[str, Any]) -> str:
    model_ids = [model["id"] for model in data["models"]]
    base_fields = [
        "id",
        "fen",
        "side_to_move",
        "stockfish_pawns",
        "stockfish_win_probability",
    ]
    model_fields = [
        f"{model_id} {suffix}"
        for model_id in model_ids
        for suffix in ("pawns", "score", "probability_error", "reasoning_status")
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=base_fields + model_fields)
    writer.writeheader()
    for position in data["positions"]:
        row: dict[str, Any] = {
            "id": position["id"],
            "fen": position["fen"],
            "side_to_move": position["side_to_move"],
            "stockfish_pawns": position["target_pawns"],
            "stockfish_win_probability": position["target_probability"],
        }
        for model_id in model_ids:
            result = position["models"][model_id]
            row[f"{model_id} pawns"] = result["predicted_pawns"]
            row[f"{model_id} score"] = result["score"]
            row[f"{model_id} probability_error"] = result["probability_error"]
            row[f"{model_id} reasoning_status"] = result["reasoning_status"]
        writer.writerow(row)
    return output.getvalue()


def write_report(
    data: dict[str, Any], output_path: Path, csv_path: Path | None = None
) -> None:
    _write_text_atomic(output_path, render_report(data))
    if csv_path is not None:
        _write_text_atomic(csv_path, _csv_text(data))


def _print_summary(data: dict[str, Any]) -> None:
    print("\nModel summary")
    for model in data["models"]:
        mean = model["mean_score"]
        sem = model["sem_score"]
        mean_text = f"{mean:.4f}" if mean is not None else "n/a"
        sem_text = f"{sem:.4f}" if sem is not None else "n/a"
        print(
            f"  {model['id']}: {mean_text} +/- {sem_text} "
            f"({model['rationale_count']}/{data['meta']['position_count']} rationales)"
        )
    if data["pairwise"]:
        print("\nPaired score differences")
        for row in data["pairwise"]:
            mean = row["mean_delta"]
            sem = row["sem_delta"]
            mean_text = f"{mean:+.4f}" if mean is not None else "n/a"
            sem_text = f"{sem:.4f}" if sem is not None else "n/a"
            print(
                f"  {row['model_a']} - {row['model_b']}: "
                f"{mean_text} +/- {sem_text} (n={row['count']})"
            )


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir", type=Path, default=project_dir / "logs" / "compare-v1"
    )
    parser.add_argument("--run-id", help="specific Inspect run id to report")
    parser.add_argument(
        "--output", type=Path, default=project_dir / "summary.html"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=project_dir / "per_position.csv",
        help="write a detailed CSV alongside the HTML (use --no-csv to disable)",
    )
    parser.add_argument("--no-csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_report_data(args.log_dir.resolve(), args.run_id)
    output_path = args.output.resolve()
    csv_path = None if args.no_csv else args.csv.resolve()
    write_report(data, output_path, csv_path)
    _print_summary(data)
    print(f"\nInteractive report: {output_path}")
    if csv_path is not None:
        print(f"Detailed CSV: {csv_path}")


if __name__ == "__main__":
    main()
