"""Numerical reference parity, outcomes, uncertainty and collection integrity."""

from __future__ import annotations

import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rating import _fit_bayes_bt, _profile_deweight, aggregate, estimate
from rating_arena import collect

if TYPE_CHECKING:
    import subprocess
    from collections.abc import Sequence
    from typing import Any


GOLDEN = {
    "names": ["a", "b", "c", "d"],
    "pairs": [
        ["a", "b", 650, 300, 50],
        ["a", "c", 45, 45, 10],
        ["a", "d", 10, 70, 20],
        ["b", "c", 30, 60, 10],
        ["b", "d", 50, 40, 10],
        ["c", "d", 75, 20, 5],
    ],
    "weights": {
        "a": 1.0698359491104061,
        "b": 0.954541520150984,
        "c": 0.9457505676111574,
        "d": 1.0298719631274522,
    },
    "mu": {
        "a": 7.992410400932152,
        "b": -84.11349157240811,
        "c": 77.07629564125901,
        "d": -0.9552144697830505,
    },
    "sigma": {
        "a": 11.413362604367348,
        "b": 11.797965928360927,
        "c": 15.92438257156821,
        "d": 15.233544418082857,
    },
    "covariance": [
        [130.26484593877103, 27.710209507349646, -83.62690075168211, -74.34815469442401],
        [27.710209507349646, 139.19200004676532, -89.5742741223803, -77.32793543173466],
        [-83.62690075168211, -89.5742741223803, 253.58596028566535, -80.38478541157383],
        [-74.34815469442401, -77.32793543173466, -80.38478541158838, 232.0608755377034],
    ],
}


class RatingTests(unittest.TestCase):
    """Validate rating tests."""

    def records(self) -> list[dict[str, Any]]:
        """Create deterministic win, draw, and loss records."""
        return [dict(a="a", b="b", seed=i, result=i % 3) for i in range(60)]

    def test_original_estimator_golden(self) -> None:
        """Check original estimator golden."""
        fixture = GOLDEN
        records = []
        for a, b, wa, wb, draws in fixture["pairs"]:
            records.extend(
                dict(a=a, b=b, seed=i, result=result)
                for i, result in enumerate([0] * wa + [1] * wb + [2] * draws)
            )
        stats = aggregate(records)
        names = fixture["names"]
        w = _profile_deweight(names, stats, 0.12, 16.0, 3.0)
        mu, sigma, cov = _fit_bayes_bt(names, stats, 600.0, w, 500)
        for actual, expected in (
            (w, fixture["weights"]),
            (mu, fixture["mu"]),
            (sigma, fixture["sigma"]),
        ):
            np.testing.assert_allclose(
                [actual[n] for n in names], [expected[n] for n in names], atol=1e-8, rtol=1e-10
            )
        np.testing.assert_allclose(cov, fixture["covariance"], atol=1e-8, rtol=1e-10)

    def test_errors_draws_and_duplicates(self) -> None:
        """Check errors draws and duplicates."""
        rows = [
            dict(a="a", b="b", seed=0, result=0, agent_errors=1, error_seat=0),
            dict(a="b", b="a", seed=1, result=2),
            dict(a="a", b="b", seed=2, result=0, engine_errors=1),
            dict(a="a", b="b", seed=3, result=0, agent_errors=1),
        ]
        s = aggregate(rows)[("a", "b")]
        self.assertEqual(
            (s["games"], s["fit_games"], s["wins"]["b"], s["draws"], s["none"]), (4, 2, 1, 1, 2)
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate(rows + rows[:1])

    def test_uncertainty_permutation_and_connectivity(self) -> None:
        """Check uncertainty permutation and connectivity."""
        r = estimate(["a", "b"], self.records(), raw_bt=True, max_sigma=1)
        self.assertFalse(r["precision_met"])
        self.assertAlmostEqual(sum(x["mu"] for x in r["ratings"]), 2000)
        cov = np.asarray(r["covariance"])
        np.testing.assert_allclose(cov.sum(0), 0, atol=1e-8)
        self.assertAlmostEqual(
            r["matchups"][0]["difference_sigma"], np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
        )
        reversed_r = estimate(["b", "a"], self.records(), raw_bt=True)
        np.testing.assert_allclose(
            [x["mu"] for x in r["ratings"]], [x["mu"] for x in reversed_r["ratings"]]
        )
        with self.assertRaisesRegex(ValueError, "Disconnected"):
            estimate(["a", "b", "c"], self.records())

    def test_resume_seats_and_changed_binary(self) -> None:
        """Check resume seats and changed binary."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            binary = p / "arena"
            binary.write_text("fixture")
            entries = {n: dict(agent=n, deck="unused") for n in ("a", "b", "c")}
            calls = []

            def run(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
                """Emulate arena output for resume and match scheduling tests."""
                calls.append(command)
                n = int(command[command.index("--games") + 1])

                class Result:
                    """Validate result."""

                    stderr = ""
                    returncode = 0

                result = Result()
                result.stdout = "\n".join(
                    json.dumps(
                        dict(game=i, result=i % 3, steps=10, agent_errors=0, engine_errors=0)
                    )
                    for i in range(n)
                )
                return result

            with patch("rating_arena.subprocess.run", side_effect=run):
                collect(binary, p / "out", entries, 2, workers=1)
                self.assertEqual(len(calls), 6)
                rows = [json.loads(s) for s in (p / "out/games.jsonl").read_text().splitlines()]
                for a, b in itertools.combinations(entries, 2):
                    self.assertEqual(
                        [r["seed"] for r in rows if (r["a"], r["b"]) == (a, b)],
                        [r["seed"] for r in rows if (r["a"], r["b"]) == (b, a)],
                    )
                collect(binary, p / "out", entries, 2, workers=1, resume=True)
                self.assertEqual(len(calls), 6)
                collect(binary, p / "out", entries, 4, workers=1, resume=True)
                self.assertEqual(len(calls), 12)
                binary.write_text("changed")
                with self.assertRaisesRegex(ValueError, "changed"):
                    collect(binary, p / "out", entries, 4, resume=True)


if __name__ == "__main__":
    unittest.main()
