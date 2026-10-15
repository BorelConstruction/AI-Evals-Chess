# Chess position evaluation benchmark

This project contains an [Inspect AI](https://inspect.aisi.org.uk/) evaluation
that asks a model for a numeric chess-position evaluation and compares it with
a fixed Stockfish reference set.

## Benchmark contract

- The checked-in dataset contains 50 unique legal positions: 25 with White to
  move and 25 with Black to move.
- Positions are sampled by a game-count-weighted random walk through the
  Lichess opening explorer. Every explorer query combines rating buckets 1800
  and 2000 and game speeds `blitz` and `rapid`.
- A walk stops at the first position represented by fewer than the configured
  fraction of filtered games at the initial position. Set
  `DEFAULT_CUTOFF_FRACTION` in `sampler.py`; the sampler and generator share
  that value, and every generated row records it. The generator uses seed
  `20261012` and pins explorer data through September 2026.
- Reference scores use Stockfish 17, 5,000,000 nodes per position, one thread,
  a 256 MiB hash, and a cleared hash between positions. Scores are always from
  White's point of view. Forced mates are represented near +/-1000 pawns, with
  the sentinel adjusted by mate distance.
- The generator records the engine binary SHA-256 and all sampling/engine
  parameters in every JSONL record.

The model is asked to return one JSON object:

```json
{
  "reasoning": "White has more central space and the more active pieces.",
  "evaluation_pawns": 0.35
}
```

`reasoning` is a concise, user-visible rationale that comparison reports can
show alongside the numeric answer. The scorer remains backward-compatible
with earlier runs where that field is absent. It tolerates common presentation
differences: raw or fenced JSON, a bare numeric value, a numeric string, or
prose containing one unambiguous evaluation number. An explicit
`evaluation_pawns` value takes precedence over other numeric metadata.
Malformed JSON, missing or non-finite evaluations, duplicate labels,
ambiguous unlabeled numbers, percentages, centipawns, and mate notation are
rejected rather than silently interpreted as pawn values. Rejected responses
receive a score of zero and an explicit `invalid_response_format` reason.
Positive means White is better; negative means Black is better.

## Metric

Both the model evaluation and the Stockfish evaluation are converted using
the agreed Lichess sigmoid formula:

```text
P(White wins) = 1 / (1 + exp(-0.00368208 * centipawns))
centipawns     = 100 * evaluation_pawns
```

The formula is applied directly to every finite evaluation, without Lichess's
separate +/-10-pawn display clamp. This makes extreme scores approach 0% or
100% continuously and makes the treatment of the +/-1000-pawn mate sentinel
explicit.

The per-position score is:

```text
score = 1 - abs(P_model - P_stockfish)
```

An exact match scores 1.0, a ten-percentage-point miss scores 0.9, and the
worst possible miss scores 0.0. Inspect reports the mean and standard error;
each sample also records the two probabilities and raw absolute probability
error in score metadata.

## Install and run

Python 3.10 is the tested runtime.

```powershell
py -3.10 -m pip install -r requirements.txt
inspect eval chess_eval.py --model openai/gpt-4o-mini
```

Use any Inspect-supported model identifier and configure that provider's API
credentials separately. A local smoke run that intentionally exercises the
invalid-format path is:

```powershell
inspect eval chess_eval.py --model mockllm/mock --limit 1
```

Run the test suite with:

```powershell
py -3.10 -m unittest discover -s tests -v
```

## Compare models

Edit the model list in `run_compare.py`, then run the comparison set:

```powershell
py -3.10 run_compare.py
```

When the run finishes, `summary.html` is a self-contained interactive report.
It includes the position board and FEN, Stockfish reference, model evaluations,
probability errors, scores, token use, timing, visible rationale, raw responses,
filters, and keyboard navigation. Open it locally with:

```powershell
Invoke-Item .\summary.html
```

To rebuild the report from logs without rerunning any models:

```powershell
py -3.10 compare.py --log-dir logs/compare-v1
```

If a provider redacts its internal reasoning and the response contains no
visible rationale, the report marks it unavailable rather than displaying the
provider's opaque payload. New runs explicitly request a concise visible
`reasoning` field.

## Regenerate the dataset

Normal eval runs load `data/chess_positions.jsonl` and make no network or
engine calls. Regeneration is an explicit offline maintenance operation:

```powershell
$env:LICHESS_API_TOKEN = Read-Host "Lichess API token"
$env:STOCKFISH_PATH = "C:\path\to\stockfish.exe"
py -3.10 generate_benchmark.py --overwrite
Remove-Item Env:LICHESS_API_TOKEN
Remove-Item Env:STOCKFISH_PATH
```

The token is read only from `LICHESS_API_TOKEN`; there is deliberately no token
command-line argument, and credentials are never written to the dataset.
Generation fails before replacing the existing JSONL if sampling or Stockfish
analysis fails.

The seed, filters, and end month make sampling stable for a fixed explorer
snapshot. Lichess may still revise historical aggregates, which is why the
generated JSONL itself—not a future live regeneration—is the authoritative
benchmark.
