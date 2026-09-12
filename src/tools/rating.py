"""Bayesian Bradley–Terry estimates from cumulative W/D/L.

This is an internal scale, not Kaggle leaderboard points.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import ArrayLike, NDArray


ELO_SCALE = 400.0 / math.log(10.0)
DEFAULT_PAIR_CAP = 500


def _sigmoid(x: float) -> float:
    """Evaluate a sigmoid without overflow for extreme rating differences."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _pair_key(a: str, b: str) -> tuple[str, str]:
    """Return the canonical ordering of a pair of agent labels."""
    return (a, b) if a <= b else (b, a)


def _empty_stat() -> dict[str, Any]:
    """Create zeroed win, draw, loss, and error counters for one matchup."""
    return {
        "games": 0,
        "fit_games": 0,
        "wins": defaultdict(int),
        "draws": 0,
        "none": 0,
        "errors": defaultdict(int),
        "steps": 0,
    }


def _solve_linear(matrix: ArrayLike, rhs: ArrayLike) -> list[float]:
    """Solve the Newton linear system using NumPy."""
    return np.linalg.solve(np.asarray(matrix, float), np.asarray(rhs, float)).tolist()


def _inverse(matrix: ArrayLike) -> list[list[float]]:
    """Invert a precision matrix for posterior covariance."""
    return np.linalg.inv(np.asarray(matrix, float)).tolist()


def _center_covariance(cov: ArrayLike) -> list[list[float]]:
    """Project covariance onto the subspace of centered ratings."""
    c = np.asarray(cov, float)
    return (c - c.mean(0, keepdims=True) - c.mean(1, keepdims=True) + c.mean()).tolist()


def _profile_arrays(
    names: list[str],
    stats: dict[tuple[str, str], dict[str, Any]],
    prior_games: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Precompute shrunk matchup scores/reliabilities for profile comparisons."""
    idx = {name: i for i, name in enumerate(names)}
    n = len(names)
    scores = np.full((n, n), 0.5, dtype=np.float64)
    reliabilities = np.zeros((n, n), dtype=np.float64)
    for (a, b), stat in stats.items():
        if a == b or a not in idx or b not in idx:
            continue
        fit_games = stat["fit_games"]
        if fit_games <= 0:
            continue
        i = idx[a]
        j = idx[b]
        denominator = fit_games + prior_games
        reliability = fit_games / denominator
        scores[i, j] = (stat["wins"][a] + 0.5 * stat["draws"] + 0.5 * prior_games) / denominator
        scores[j, i] = (stat["wins"][b] + 0.5 * stat["draws"] + 0.5 * prior_games) / denominator
        reliabilities[i, j] = reliability
        reliabilities[j, i] = reliability
    return scores, reliabilities


def _profile_distance_arrays(
    scores: np.ndarray,
    reliabilities: np.ndarray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute every profile distance/shared count with NumPy reductions."""
    n = scores.shape[0]
    distances = np.ones((n, n), dtype=np.float64)
    shared = np.zeros((n, n), dtype=np.int64)
    diagonal = np.arange(n)
    for i in range(n):
        weights = np.minimum(reliabilities, reliabilities[i][None, :])
        weights[:, i] = 0.0
        weights[diagonal, diagonal] = 0.0
        deltas = scores - scores[i][None, :]
        weight_sums = weights.sum(axis=1)
        square_sums = (weights * deltas * deltas).sum(axis=1)
        np.sqrt(
            np.divide(square_sums, weight_sums, out=np.ones(n), where=weight_sums > 0),
            out=distances[i],
        )
        shared[i] = np.count_nonzero(weights > 0, axis=1)
        distances[i, i] = 0.0
        shared[i, i] = n - 1
    return distances, shared


def _profile_deweight(
    names: list[str],
    stats: dict[tuple[str, str], dict[str, Any]],
    bandwidth: float,
    prior_games: float,
    cap: float,
) -> dict[str, float]:
    """Reduce the influence of dense clusters of similar opponent profiles."""
    if bandwidth <= 0:
        return {name: 1.0 for name in names}
    scores, reliabilities = _profile_arrays(names, stats, prior_games)
    distances, _ = _profile_distance_arrays(scores, reliabilities)
    weights: dict[str, float] = {}
    denominator = 2.0 * bandwidth * bandwidth
    densities = np.exp(-(distances * distances) / denominator).sum(axis=1)
    for i, name in enumerate(names):
        weights[name] = 1.0 / max(float(densities[i]), 1e-9)
    mean_weight = sum(weights.values()) / len(weights) if weights else 1.0
    normalized = {name: weights[name] / mean_weight for name in names}
    if cap > 0:
        normalized = {name: min(cap, max(1.0 / cap, weight)) for name, weight in normalized.items()}
        mean_weight = sum(normalized.values()) / len(normalized) if normalized else 1.0
        normalized = {name: weight / mean_weight for name, weight in normalized.items()}
    return normalized


def _pair_fit_weight(a: str, b: str, agent_weights: dict[str, float] | None) -> float:
    """Combine the two agent weights for one matchup."""
    if not agent_weights:
        return 1.0
    return math.sqrt(agent_weights.get(a, 1.0) * agent_weights.get(b, 1.0))


def _effective_fit_games(fit_games: int, pair_cap: int) -> float:
    """Cap the effective evidence contributed by one matchup."""
    if pair_cap <= 0:
        return float(fit_games)
    return float(min(fit_games, pair_cap))


def _fit_bayes_bt(
    names: list[str],
    stats: dict[tuple[str, str], dict[str, Any]],
    prior_sigma: float,
    agent_weights: dict[str, float] | None = None,
    pair_cap: int = DEFAULT_PAIR_CAP,
) -> tuple[dict[str, float], dict[str, float], list[list[float]]]:
    """Fit centered Bayesian Bradley-Terry ratings and their Laplace covariance."""
    idx = {name: i for i, name in enumerate(names)}
    n = len(names)
    ratings = [0.0] * n
    prior_precision = 1.0 / (prior_sigma * prior_sigma)

    for _ in range(80):
        grad = [-prior_precision * r for r in ratings]
        info = [[0.0] * n for _ in range(n)]
        for (a, b), s in stats.items():
            if a == b or a not in idx or b not in idx:
                continue
            fit_games = s["fit_games"]
            if fit_games <= 0:
                continue
            pair_weight = _pair_fit_weight(a, b, agent_weights)
            effective_fit_games = _effective_fit_games(fit_games, pair_cap)
            effective_scale = effective_fit_games / fit_games
            i = idx[a]
            j = idx[b]
            fit_games_w = effective_fit_games * pair_weight
            score_a = (s["wins"][a] + 0.5 * s["draws"]) * effective_scale * pair_weight
            p = _sigmoid((ratings[i] - ratings[j]) / ELO_SCALE)
            g = score_a - fit_games_w * p
            w = fit_games_w * p * (1.0 - p)
            grad[i] += g / ELO_SCALE
            grad[j] -= g / ELO_SCALE
            v = w / (ELO_SCALE * ELO_SCALE)
            info[i][i] += v
            info[j][j] += v
            info[i][j] -= v
            info[j][i] -= v

        for i in range(n):
            info[i][i] += prior_precision

        step = _solve_linear(info, grad)
        max_step = max(abs(x) for x in step) if step else 0.0
        for i in range(n):
            ratings[i] += step[i]
        mean_rating = sum(ratings) / n
        ratings = [r - mean_rating for r in ratings]
        if max_step < 1e-5:
            break
    else:
        raise ValueError("Bradley-Terry fit did not converge")

    info = [[0.0] * n for _ in range(n)]
    for (a, b), s in stats.items():
        if a == b or a not in idx or b not in idx:
            continue
        fit_games = s["fit_games"]
        if fit_games <= 0:
            continue
        pair_weight = _pair_fit_weight(a, b, agent_weights)
        effective_fit_games = _effective_fit_games(fit_games, pair_cap)
        i = idx[a]
        j = idx[b]
        p = _sigmoid((ratings[i] - ratings[j]) / ELO_SCALE)
        v = effective_fit_games * pair_weight * p * (1.0 - p) / (ELO_SCALE * ELO_SCALE)
        info[i][i] += v
        info[j][j] += v
        info[i][j] -= v
        info[j][i] -= v
    for i in range(n):
        info[i][i] += prior_precision
    cov = _center_covariance(_inverse(info))

    rating_by_name = {name: ratings[idx[name]] for name in names}
    se_by_name = {name: math.sqrt(max(0.0, cov[idx[name]][idx[name]])) for name in names}
    return rating_by_name, se_by_name, cov


def aggregate(records: Sequence[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Keep every outcome; only attributed agent errors become forfeits."""
    stats = defaultdict(_empty_stat)
    seen = set()
    for r in records:
        if type(r["seed"]) is not int or not 0 <= r["seed"] < 2**32:
            raise ValueError("Invalid game seed")
        for key in ("steps", "agent_errors", "engine_errors"):
            if type(r.get(key, 0)) is not int or r.get(key, 0) < 0:
                raise ValueError("Invalid game counts")
        if type(r.get("result", -1)) is not int or r.get("result", -1) not in (-1, 0, 1, 2):
            raise ValueError("Invalid game outcome")
        if type(r.get("error_seat", -1)) is not int or r.get("error_seat", -1) not in (-1, 0, 1):
            raise ValueError("Invalid error seat")
        key = (r["a"], r["b"], r["seed"])
        if key in seen:
            raise ValueError("Duplicate game identity")
        seen.add(key)
        if r["a"] == r["b"]:
            continue
        s = stats[_pair_key(r["a"], r["b"])]
        s["games"] += 1
        s["steps"] += r.get("steps", 0)
        result = r.get("result", -1)
        error_seat = r.get("error_seat", -1)
        if r.get("engine_errors", 0) or r.get("unresolved", False):
            result = -1
        elif r.get("agent_errors", 0):
            if error_seat in (0, 1):
                s["errors"][(r["a"], r["b"])[error_seat]] += 1
                result = 1 - error_seat
            else:
                result = -1
        if result in (0, 1):
            s["wins"][(r["a"], r["b"])[result]] += 1
            s["fit_games"] += 1
        elif result == 2:
            s["draws"] += 1
            s["fit_games"] += 1
        else:
            s["none"] += 1
    return stats


def estimate(
    names: Sequence[str],
    records: Sequence[dict[str, Any]],
    prior_sigma: float = 600.0,
    pair_cap: int = 500,
    raw_bt: bool = False,
    max_sigma: float | None = None,
) -> dict[str, Any]:
    """Estimate ratings and uncertainty for a connected comparison pool."""
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("At least two unique agents required")
    if not math.isfinite(prior_sigma) or prior_sigma <= 0 or pair_cap < 0:
        raise ValueError("Invalid prior or pair cap")
    if max_sigma is not None and (not math.isfinite(max_sigma) or max_sigma <= 0):
        raise ValueError("Invalid uncertainty target")
    if any(r["a"] not in names or r["b"] not in names for r in records):
        raise ValueError("Unknown recorded agent")
    stats = aggregate(records)
    neighbors = {n: set() for n in names}
    for (a, b), s in stats.items():
        if s["fit_games"]:
            neighbors[a].add(b)
            neighbors[b].add(a)
    reached = {names[0]}
    pending = [names[0]]
    while pending:
        for n in neighbors[pending.pop()] - reached:
            reached.add(n)
            pending.append(n)
    if reached != set(names):
        raise ValueError(
            "Disconnected comparison pool: collect resolved cross-group games before ranking"
        )
    weights = (
        {n: 1.0 for n in names} if raw_bt else _profile_deweight(names, stats, 0.12, 16.0, 3.0)
    )
    cap = 0 if raw_bt else pair_cap
    mu, sigma, cov = _fit_bayes_bt(names, stats, prior_sigma, weights, cap)
    ratings = []
    for n in names:
        games = sum(s["fit_games"] for pair, s in stats.items() if n in pair)
        ratings.append(
            dict(
                agent=n,
                mu=1000 + mu[n],
                sigma=sigma[n],
                lcb=1000 + mu[n] - 2 * sigma[n],
                interval95=[1000 + mu[n] - 1.96 * sigma[n], 1000 + mu[n] + 1.96 * sigma[n]],
                games=games,
                weight=weights[n],
            )
        )
    pairs = []
    for (a, b), s in sorted(stats.items()):
        ia, ib = names.index(a), names.index(b)
        variance = cov[ia][ia] + cov[ib][ib] - 2 * cov[ia][ib]
        pairs.append(
            dict(
                a=a,
                b=b,
                **s,
                model_win_probability=_sigmoid((mu[a] - mu[b]) / ELO_SCALE),
                score_rate=(s["wins"][a] + 0.5 * s["draws"]) / s["fit_games"]
                if s["fit_games"]
                else None,
                difference_sigma=math.sqrt(max(0.0, variance)),
            )
        )
    return dict(
        format="ptcg-internal-rating-v1",
        scale="Bayesian Bradley-Terry, field mean 1000; not Kaggle LB rating",
        prior_sigma=prior_sigma,
        pair_cap=cap,
        profile_bandwidth=0 if raw_bt else 0.12,
        profile_prior_games=16.0,
        uncertainty="centered Laplace covariance; conditional on the model and comparison pool",
        max_sigma=max_sigma,
        precision_met=None if max_sigma is None else max(sigma.values()) <= max_sigma,
        ratings=sorted(ratings, key=lambda r: (-r["mu"], r["agent"])),
        covariance_order=names,
        covariance=cov,
        matchups=pairs,
    )
