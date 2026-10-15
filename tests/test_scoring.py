from __future__ import annotations

import asyncio
import math
import unittest
from types import SimpleNamespace

from inspect_ai.scorer import Target

from chess_eval import winning_probability_similarity
from chess_scoring import (
    parse_model_evaluation,
    probability_similarity,
    win_probability,
)


class WinProbabilityTests(unittest.TestCase):
    def test_known_values(self) -> None:
        expected = {
            0.0: 0.5,
            1.0: 0.5910258971916128,
            -1.0: 0.4089741028083871,
            3.0: 0.7511255009383484,
            10.0: 0.9754474363414323,
        }
        for pawns, probability in expected.items():
            with self.subTest(pawns=pawns):
                self.assertAlmostEqual(win_probability(pawns), probability)

    def test_symmetry_and_monotonicity(self) -> None:
        values = [0.0, 0.25, 1.0, 3.0, 10.0]
        probabilities = [win_probability(value) for value in values]
        self.assertEqual(probabilities, sorted(probabilities))
        for value in values:
            self.assertAlmostEqual(
                win_probability(-value), 1.0 - win_probability(value)
            )

    def test_huge_finite_values_are_stable(self) -> None:
        self.assertEqual(win_probability(1e300), 1.0)
        self.assertEqual(win_probability(-1e300), 0.0)

    def test_non_finite_values_fail(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                win_probability(value)
        with self.assertRaises(TypeError):
            win_probability(True)


class ModelResponseTests(unittest.TestCase):
    def test_accepts_json(self) -> None:
        self.assertEqual(parse_model_evaluation('{"evaluation_pawns": 0.35}'), 0.35)
        self.assertEqual(parse_model_evaluation(' {"evaluation_pawns": -2} \n'), -2.0)
        self.assertEqual(
            parse_model_evaluation(
                '{"evaluation_pawns": 0.3, "confidence": 0.8}'
            ),
            0.3,
        )

    def test_accepts_structured_reasoning_json(self) -> None:
        completion = (
            '{"reasoning": "White controls the center and has more space.", '
            '"evaluation_pawns": 0.35}'
        )

        self.assertEqual(parse_model_evaluation(completion), 0.35)

    def test_accepts_common_model_formatting(self) -> None:
        accepted = {
            '```json\n{"evaluation_pawns": 0.35}\n```': 0.35,
            '```\n{"evaluation_pawns": 0.25}\n```': 0.25,
            '{"evaluation_pawns": "-0.75"}': -0.75,
            "evaluation_pawns = +.5": 0.5,
            "The pawn evaluation is −1.25 pawns.": -1.25,
            "I estimate 0.2 pawns for White.": 0.2,
            "0": 0.0,
        }
        for completion, expected in accepted.items():
            with self.subTest(completion=completion):
                self.assertEqual(parse_model_evaluation(completion), expected)

    def test_invalid_responses_fail(self) -> None:
        invalid = [
            "",
            "The position is equal.",
            "```json\n{\"evaluation_pawns\": 0.35}",
            '{"evaluation_pawns": 0.35',
            '{"estimate": 0.35}',
            '{"evaluation_pawns": true}',
            '{"evaluation_pawns": null}',
            '{"evaluation_pawns": NaN}',
            '{"evaluation_pawns": Infinity}',
            '{"evaluation_pawns": 1e999}',
            '{"evaluation_pawns": ' + ("9" * 400) + "}",
            '{"evaluation_pawns": 0.3, "evaluation_pawns": 0.4}',
            "The estimate is between 0.3 and 0.5 pawns.",
            '{"evaluation_pawns": null, "confidence": 0.9}',
            "Evaluation: 55%",
            "Evaluation: 35 cp",
            "Mate is #3",
        ]
        for completion in invalid:
            with self.subTest(completion=completion), self.assertRaises(ValueError):
                parse_model_evaluation(completion)

    def test_probability_similarity(self) -> None:
        self.assertEqual(probability_similarity(0.5, 0.5), 1.0)
        self.assertAlmostEqual(
            probability_similarity(1.0, -1.0), 0.8179482056167742
        )


class InspectScorerTests(unittest.TestCase):
    def test_valid_response(self) -> None:
        scorer = winning_probability_similarity()
        state = SimpleNamespace(
            output=SimpleNamespace(completion='{"evaluation_pawns": 1.0}')
        )
        result = asyncio.run(scorer(state, Target("-1.0")))
        self.assertAlmostEqual(float(result.value), 0.8179482056167742)
        self.assertEqual(result.metadata["predicted_pawns"], 1.0)
        self.assertEqual(result.metadata["target_pawns"], -1.0)

    def test_fenced_json_response(self) -> None:
        scorer = winning_probability_similarity()
        state = SimpleNamespace(
            output=SimpleNamespace(
                completion='```json\n{"evaluation_pawns": 1.0}\n```'
            )
        )
        result = asyncio.run(scorer(state, Target("-1.0")))
        self.assertAlmostEqual(float(result.value), 0.8179482056167742)
        self.assertEqual(result.metadata["predicted_pawns"], 1.0)

    def test_structured_reasoning_response_scores_normally(self) -> None:
        scorer = winning_probability_similarity()
        completion = (
            '{"reasoning": "White controls the center and has more space.", '
            '"evaluation_pawns": 1.0}'
        )
        state = SimpleNamespace(output=SimpleNamespace(completion=completion))

        result = asyncio.run(scorer(state, Target("-1.0")))

        self.assertAlmostEqual(float(result.value), 0.8179482056167742)
        self.assertEqual(result.answer, completion)
        self.assertEqual(result.metadata["predicted_pawns"], 1.0)

    def test_invalid_model_response_scores_zero(self) -> None:
        scorer = winning_probability_similarity()
        state = SimpleNamespace(
            output=SimpleNamespace(completion="The position is equal.")
        )
        result = asyncio.run(scorer(state, Target("0.0")))
        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.reason, "invalid_response_format")
        self.assertIn("parse_error", result.metadata)

    def test_bad_dataset_target_raises(self) -> None:
        scorer = winning_probability_similarity()
        state = SimpleNamespace(
            output=SimpleNamespace(completion='{"evaluation_pawns": 0.0}')
        )
        with self.assertRaises(ValueError):
            asyncio.run(scorer(state, Target("not-a-number")))


if __name__ == "__main__":
    unittest.main()
