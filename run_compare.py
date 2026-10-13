from inspect_ai import eval_set

eval_set(
    "chess_eval.py",
    model=[
        "anthropic/claude-haiku-4-5-20251001",
        "anthropic/claude-sonnet-5-5",
        "anthropic/claude-opus-5-5",
    ],
    log_dir="logs/compare-v1",
)