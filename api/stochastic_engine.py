from __future__ import annotations

import math
import statistics
import time
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter(
    prefix="/stochastic",
    tags=["stochastic"],
)

ENGINE_NAME = "TravisLabs Python/NumPy stochastic engine"

ModelName = Literal[
    "brownian",
    "gbm",
    "ou",
    "vasicek",
    "cir",
    "heston",
]

MAX_PATHS = 500_000
MAX_STATE_UPDATES = 300_000_000


class StochasticSimulationRequest(BaseModel):
    model: ModelName

    x0: float = 0.0
    s0: float = 100.0
    v0: float = 0.04

    mu: float = 0.05
    sigma: float = 0.20

    kappa: float = 2.0
    theta: float = 0.05
    xi: float = 0.30
    rho: float = -0.70

    maturity: float = Field(default=1.0, gt=0.0, le=30.0)
    steps: int = Field(default=252, ge=1, le=5000)
    paths: int = Field(default=10_000, ge=100, le=MAX_PATHS)
    seed: int = 42

    sample_paths: int = Field(default=20, ge=0, le=100)
    terminal_preview: int = Field(default=2000, ge=0, le=5000)
    batch_size: int = Field(default=50_000, ge=1000, le=100_000)


class StochasticBenchmarkRequest(BaseModel):
    model: ModelName = "ou"

    x0: float = 0.0
    s0: float = 100.0
    v0: float = 0.04

    mu: float = 0.05
    sigma: float = 0.20

    kappa: float = 2.0
    theta: float = 0.05
    xi: float = 0.30
    rho: float = -0.70

    maturity: float = Field(default=1.0, gt=0.0, le=30.0)
    steps: int = Field(default=252, ge=1, le=2000)

    path_counts: list[int] = Field(
        default_factory=lambda: [10_000, 50_000, 100_000, 250_000, 500_000],
        min_length=1,
        max_length=8,
    )

    batch_size: int = Field(default=50_000, ge=1000, le=100_000)
    seed: int = 42
    repeats: int = Field(default=1, ge=1, le=3)


def _validate_common(
    *,
    model: str,
    sigma: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> None:
    if sigma < 0.0:
        raise HTTPException(
            status_code=400,
            detail="sigma must be non-negative",
        )

    if model in {"ou", "vasicek", "cir", "heston"} and kappa <= 0.0:
        raise HTTPException(
            status_code=400,
            detail="kappa must be greater than zero for mean-reverting models",
        )

    if model in {"cir", "heston"} and theta < 0.0:
        raise HTTPException(
            status_code=400,
            detail="theta must be non-negative for CIR/Heston",
        )

    if model == "heston":
        if xi < 0.0:
            raise HTTPException(
                status_code=400,
                detail="xi must be non-negative for Heston",
            )

        if not -1.0 <= rho <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="rho must be between -1 and 1",
            )


def _validate_workload(paths: int, steps: int) -> None:
    state_updates = paths * steps

    if state_updates > MAX_STATE_UPDATES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Requested workload is {state_updates:,} state updates. "
                f"Maximum per request is {MAX_STATE_UPDATES:,}. "
                "Reduce paths or steps."
            ),
        )


def _theoretical_moments_at_time(
    *,
    model: str,
    t: float,
    x0: float,
    s0: float,
    v0: float,
    mu: float,
    sigma: float,
    kappa: float,
    theta: float,
) -> tuple[float | None, float | None]:
    if model == "brownian":
        mean = x0
        variance = sigma * sigma * t
        return mean, variance

    if model == "gbm":
        mean = s0 * math.exp(mu * t)
        variance = (
            s0
            * s0
            * math.exp(2.0 * mu * t)
            * (math.exp(sigma * sigma * t) - 1.0)
        )
        return mean, variance

    if model in {"ou", "vasicek"}:
        decay = math.exp(-kappa * t)
        mean = theta + (x0 - theta) * decay
        variance = (
            sigma
            * sigma
            / (2.0 * kappa)
            * (1.0 - math.exp(-2.0 * kappa * t))
        )
        return mean, variance

    if model == "cir":
        decay = math.exp(-kappa * t)
        mean = theta + (x0 - theta) * decay
        variance = (
            x0
            * sigma
            * sigma
            * decay
            * (1.0 - decay)
            / kappa
            + theta
            * sigma
            * sigma
            * (1.0 - decay) ** 2
            / (2.0 * kappa)
        )
        return mean, variance

    if model == "heston":
        return None, None

    return None, None


def _theoretical_paths(
    *,
    model: str,
    times: np.ndarray,
    x0: float,
    s0: float,
    v0: float,
    mu: float,
    sigma: float,
    kappa: float,
    theta: float,
) -> tuple[list[float] | None, list[float] | None]:
    if model == "heston":
        return None, None

    means: list[float] = []
    variances: list[float] = []

    for t in times:
        mean, variance = _theoretical_moments_at_time(
            model=model,
            t=float(t),
            x0=x0,
            s0=s0,
            v0=v0,
            mu=mu,
            sigma=sigma,
            kappa=kappa,
            theta=theta,
        )

        means.append(float(mean))
        variances.append(float(variance))

    return means, variances


def _sample_variance_from_sums(
    total: float,
    total_sq: float,
    n: int,
) -> float:
    if n <= 1:
        return 0.0

    numerator = total_sq - (total * total / n)
    variance = numerator / (n - 1)

    return float(max(variance, 0.0))


def _simulate_batched(
    *,
    model: str,
    x0: float,
    s0: float,
    v0: float,
    mu: float,
    sigma: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    maturity: float,
    steps: int,
    paths: int,
    seed: int,
    batch_size: int,
    sample_paths: int,
    terminal_preview: int,
    track_time_series: bool,
) -> dict:
    dt = maturity / steps
    sqrt_dt = math.sqrt(dt)

    times = np.linspace(
        0.0,
        maturity,
        steps + 1,
        dtype=np.float64,
    )

    if track_time_series:
        sum_path = np.zeros(steps + 1, dtype=np.float64)
        sumsq_path = np.zeros(steps + 1, dtype=np.float64)
    else:
        sum_path = None
        sumsq_path = None

    if model == "heston" and track_time_series:
        variance_sum_path = np.zeros(steps + 1, dtype=np.float64)
    else:
        variance_sum_path = None

    effective_sample_paths = min(
        sample_paths,
        paths,
        batch_size,
    )

    if effective_sample_paths > 0:
        sampled = np.empty(
            (effective_sample_paths, steps + 1),
            dtype=np.float64,
        )
    else:
        sampled = None

    preview_values: list[float] = []

    terminal_sum = 0.0
    terminal_sumsq = 0.0
    terminal_variance_state_sum = 0.0

    rng = np.random.default_rng(seed)

    processed = 0
    batch_number = 0

    start = time.perf_counter()

    while processed < paths:
        n = min(
            batch_size,
            paths - processed,
        )

        if model in {"gbm", "heston"}:
            x = np.full(n, s0, dtype=np.float64)
        else:
            x = np.full(n, x0, dtype=np.float64)

        if model == "heston":
            v = np.full(n, max(v0, 0.0), dtype=np.float64)
        else:
            v = None

        if track_time_series:
            sum_path[0] += float(np.sum(x))
            sumsq_path[0] += float(np.dot(x, x))

            if model == "heston":
                variance_sum_path[0] += float(np.sum(v))

        if batch_number == 0 and sampled is not None:
            sampled[:, 0] = x[:effective_sample_paths]

        for step in range(1, steps + 1):
            if model == "brownian":
                z = rng.standard_normal(n)
                x = x + sigma * sqrt_dt * z

            elif model == "gbm":
                z = rng.standard_normal(n)
                x = x * np.exp(
                    (mu - 0.5 * sigma * sigma) * dt
                    + sigma * sqrt_dt * z
                )

            elif model in {"ou", "vasicek"}:
                z = rng.standard_normal(n)
                x = (
                    x
                    + kappa * (theta - x) * dt
                    + sigma * sqrt_dt * z
                )

            elif model == "cir":
                z = rng.standard_normal(n)
                x_positive = np.maximum(x, 0.0)

                x = (
                    x
                    + kappa * (theta - x_positive) * dt
                    + sigma * np.sqrt(x_positive) * sqrt_dt * z
                )

                x = np.maximum(x, 0.0)

            elif model == "heston":
                z_spot = rng.standard_normal(n)
                z_independent = rng.standard_normal(n)

                z_variance = (
                    rho * z_spot
                    + math.sqrt(max(1.0 - rho * rho, 0.0))
                    * z_independent
                )

                v_old = np.maximum(v, 0.0)
                sqrt_v = np.sqrt(v_old)

                x = x * np.exp(
                    (mu - 0.5 * v_old) * dt
                    + sqrt_v * sqrt_dt * z_spot
                )

                v = (
                    v
                    + kappa * (theta - v_old) * dt
                    + xi * sqrt_v * sqrt_dt * z_variance
                )

                v = np.maximum(v, 0.0)

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported model: {model}",
                )

            if track_time_series:
                sum_path[step] += float(np.sum(x))
                sumsq_path[step] += float(np.dot(x, x))

                if model == "heston":
                    variance_sum_path[step] += float(np.sum(v))

            if batch_number == 0 and sampled is not None:
                sampled[:, step] = x[:effective_sample_paths]

        terminal_sum += float(np.sum(x))
        terminal_sumsq += float(np.dot(x, x))

        if model == "heston":
            terminal_variance_state_sum += float(np.sum(v))

        remaining_preview = terminal_preview - len(preview_values)

        if remaining_preview > 0:
            take = min(remaining_preview, n)
            preview_values.extend(
                x[:take].astype(float).tolist()
            )

        processed += n
        batch_number += 1

    runtime_ms = (
        time.perf_counter() - start
    ) * 1000.0

    terminal_mean = terminal_sum / paths

    terminal_variance = _sample_variance_from_sums(
        terminal_sum,
        terminal_sumsq,
        paths,
    )

    result: dict = {
        "runtime_ms": float(runtime_ms),
        "terminal_mean": float(terminal_mean),
        "terminal_variance": float(terminal_variance),
        "terminal_std_dev": float(math.sqrt(terminal_variance)),
        "terminal_preview": preview_values,
        "sample_paths": (
            sampled.astype(float).tolist()
            if sampled is not None
            else []
        ),
        "batches": batch_number,
        "effective_sample_paths": effective_sample_paths,
    }

    if model == "heston":
        result["terminal_variance_state_mean"] = float(
            terminal_variance_state_sum / paths
        )

    if track_time_series:
        mean_path = sum_path / paths

        if paths > 1:
            variance_path = (
                sumsq_path
                - (sum_path * sum_path / paths)
            ) / (paths - 1)

            variance_path = np.maximum(
                variance_path,
                0.0,
            )
        else:
            variance_path = np.zeros_like(mean_path)

        result["times"] = times.astype(float).tolist()
        result["mean_path"] = mean_path.astype(float).tolist()
        result["variance_path"] = variance_path.astype(float).tolist()

        if model == "heston":
            result["mean_variance_path"] = (
                variance_sum_path / paths
            ).astype(float).tolist()

    return result


@router.post("/simulate")
def simulate_stochastic(
    request: StochasticSimulationRequest,
):
    _validate_common(
        model=request.model,
        sigma=request.sigma,
        kappa=request.kappa,
        theta=request.theta,
        xi=request.xi,
        rho=request.rho,
    )

    _validate_workload(
        request.paths,
        request.steps,
    )

    simulation = _simulate_batched(
        model=request.model,
        x0=request.x0,
        s0=request.s0,
        v0=request.v0,
        mu=request.mu,
        sigma=request.sigma,
        kappa=request.kappa,
        theta=request.theta,
        xi=request.xi,
        rho=request.rho,
        maturity=request.maturity,
        steps=request.steps,
        paths=request.paths,
        seed=request.seed,
        batch_size=request.batch_size,
        sample_paths=request.sample_paths,
        terminal_preview=request.terminal_preview,
        track_time_series=True,
    )

    theoretical_mean, theoretical_variance = (
        _theoretical_moments_at_time(
            model=request.model,
            t=request.maturity,
            x0=request.x0,
            s0=request.s0,
            v0=request.v0,
            mu=request.mu,
            sigma=request.sigma,
            kappa=request.kappa,
            theta=request.theta,
        )
    )

    times = np.linspace(
        0.0,
        request.maturity,
        request.steps + 1,
        dtype=np.float64,
    )

    (
        theoretical_mean_path,
        theoretical_variance_path,
    ) = _theoretical_paths(
        model=request.model,
        times=times,
        x0=request.x0,
        s0=request.s0,
        v0=request.v0,
        mu=request.mu,
        sigma=request.sigma,
        kappa=request.kappa,
        theta=request.theta,
    )

    if theoretical_variance is not None:
        theoretical_std = math.sqrt(
            max(theoretical_variance, 0.0)
        )
    else:
        theoretical_std = None

    empirical_mean = simulation["terminal_mean"]
    empirical_variance = simulation["terminal_variance"]

    if theoretical_mean is not None:
        mean_error = abs(
            empirical_mean - theoretical_mean
        )
    else:
        mean_error = None

    if theoretical_variance is not None:
        variance_error = abs(
            empirical_variance - theoretical_variance
        )
    else:
        variance_error = None

    half_life = None

    if request.model in {
        "ou",
        "vasicek",
        "cir",
        "heston",
    }:
        half_life = math.log(2.0) / request.kappa

    cir_feller_left = None
    cir_feller_right = None
    cir_feller_satisfied = None

    if request.model == "cir":
        cir_feller_left = (
            2.0
            * request.kappa
            * request.theta
        )

        cir_feller_right = (
            request.sigma
            * request.sigma
        )

        cir_feller_satisfied = (
            cir_feller_left
            >= cir_feller_right
        )

    heston_feller_left = None
    heston_feller_right = None
    heston_feller_satisfied = None
    theoretical_expected_variance = None

    if request.model == "heston":
        heston_feller_left = (
            2.0
            * request.kappa
            * request.theta
        )

        heston_feller_right = (
            request.xi
            * request.xi
        )

        heston_feller_satisfied = (
            heston_feller_left
            >= heston_feller_right
        )

        theoretical_expected_variance = (
            request.theta
            + (
                request.v0
                - request.theta
            )
            * math.exp(
                -request.kappa
                * request.maturity
            )
        )

    state_updates = (
        request.paths
        * request.steps
    )

    runtime_seconds = (
        simulation["runtime_ms"]
        / 1000.0
    )

    if runtime_seconds > 0.0:
        updates_per_second = (
            state_updates
            / runtime_seconds
        )

        paths_per_second = (
            request.paths
            / runtime_seconds
        )
    else:
        updates_per_second = None
        paths_per_second = None

    return {
        "engine": ENGINE_NAME,
        "model": request.model,

        "inputs": {
            "x0": request.x0,
            "s0": request.s0,
            "v0": request.v0,
            "mu": request.mu,
            "sigma": request.sigma,
            "kappa": request.kappa,
            "theta": request.theta,
            "xi": request.xi,
            "rho": request.rho,
            "maturity": request.maturity,
            "steps": request.steps,
            "paths": request.paths,
            "seed": request.seed,
            "dt": (
                request.maturity
                / request.steps
            ),
            "batch_size": request.batch_size,
        },

        "empirical": {
            "mean": simulation[
                "terminal_mean"
            ],
            "variance": simulation[
                "terminal_variance"
            ],
            "std_dev": simulation[
                "terminal_std_dev"
            ],
        },

        "theoretical": {
            "mean": theoretical_mean,
            "variance": (
                theoretical_variance
            ),
            "std_dev": theoretical_std,
        },

        "errors": {
            "mean_absolute_error": (
                mean_error
            ),
            "variance_absolute_error": (
                variance_error
            ),
        },

        "diagnostics": {
            "mean_reversion_half_life": (
                half_life
            ),

            "feller_left_2kappa_theta": (
                cir_feller_left
            ),
            "feller_right_sigma_squared": (
                cir_feller_right
            ),
            "feller_satisfied": (
                cir_feller_satisfied
            ),

            "heston_feller_left_2kappa_theta": (
                heston_feller_left
            ),
            "heston_feller_right_xi_squared": (
                heston_feller_right
            ),
            "heston_feller_satisfied": (
                heston_feller_satisfied
            ),

            "terminal_variance_state_mean": (
                simulation.get(
                    "terminal_variance_state_mean"
                )
            ),
            "theoretical_expected_variance": (
                theoretical_expected_variance
            ),

            "runtime_ms": (
                simulation["runtime_ms"]
            ),
            "state_updates": (
                state_updates
            ),
            "batches": (
                simulation["batches"]
            ),
            "batch_size": (
                request.batch_size
            ),
            "paths_per_second": (
                paths_per_second
            ),
            "state_updates_per_second": (
                updates_per_second
            ),
        },

        "chart_data": {
            "times": simulation["times"],
            "sample_paths": (
                simulation[
                    "sample_paths"
                ]
            ),
            "terminal_preview": (
                simulation[
                    "terminal_preview"
                ]
            ),
            "mean_path": (
                simulation[
                    "mean_path"
                ]
            ),
            "variance_path": (
                simulation[
                    "variance_path"
                ]
            ),
            "theoretical_mean_path": (
                theoretical_mean_path
            ),
            "theoretical_variance_path": (
                theoretical_variance_path
            ),
            "mean_variance_path": (
                simulation.get(
                    "mean_variance_path"
                )
            ),
        },
    }


@router.post("/benchmark")
def benchmark_stochastic(
    request: StochasticBenchmarkRequest,
):
    _validate_common(
        model=request.model,
        sigma=request.sigma,
        kappa=request.kappa,
        theta=request.theta,
        xi=request.xi,
        rho=request.rho,
    )

    cleaned_counts = sorted(
        set(request.path_counts)
    )

    for count in cleaned_counts:
        if count < 100:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Every benchmark path count "
                    "must be at least 100."
                ),
            )

        if count > MAX_PATHS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Benchmark path counts cannot exceed "
                    f"{MAX_PATHS:,}."
                ),
            )

        _validate_workload(
            count,
            request.steps,
        )

    total_updates = (
        sum(cleaned_counts)
        * request.steps
        * request.repeats
    )

    max_total_benchmark_updates = (
        900_000_000
    )

    if (
        total_updates
        > max_total_benchmark_updates
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Benchmark requests are limited to "
                f"{max_total_benchmark_updates:,} total "
                "state updates across all path counts "
                "and repeats."
            ),
        )

    results = []

    for path_count in cleaned_counts:
        run_times: list[float] = []
        last_run: dict | None = None

        for repeat in range(
            request.repeats
        ):
            run = _simulate_batched(
                model=request.model,
                x0=request.x0,
                s0=request.s0,
                v0=request.v0,
                mu=request.mu,
                sigma=request.sigma,
                kappa=request.kappa,
                theta=request.theta,
                xi=request.xi,
                rho=request.rho,
                maturity=request.maturity,
                steps=request.steps,
                paths=path_count,
                seed=(
                    request.seed
                    + repeat
                ),
                batch_size=(
                    request.batch_size
                ),
                sample_paths=0,
                terminal_preview=0,
                track_time_series=False,
            )

            run_times.append(
                run["runtime_ms"]
            )

            last_run = run

        median_ms = float(
            statistics.median(
                run_times
            )
        )

        mean_ms = float(
            statistics.fmean(
                run_times
            )
        )

        runtime_seconds = (
            median_ms
            / 1000.0
        )

        state_updates = (
            path_count
            * request.steps
        )

        if runtime_seconds > 0.0:
            paths_per_second = (
                path_count
                / runtime_seconds
            )

            updates_per_second = (
                state_updates
                / runtime_seconds
            )
        else:
            paths_per_second = None
            updates_per_second = None

        random_draws_per_step = (
            2
            if request.model == "heston"
            else 1
        )

        results.append(
            {
                "paths": path_count,
                "steps": (
                    request.steps
                ),
                "state_updates": (
                    state_updates
                ),
                "approx_random_draws": (
                    state_updates
                    * random_draws_per_step
                ),
                "runs_ms": [
                    float(x)
                    for x in run_times
                ],
                "median_runtime_ms": (
                    median_ms
                ),
                "mean_runtime_ms": (
                    mean_ms
                ),
                "paths_per_second": (
                    paths_per_second
                ),
                "state_updates_per_second": (
                    updates_per_second
                ),
                "terminal_mean": (
                    last_run[
                        "terminal_mean"
                    ]
                ),
                "terminal_variance": (
                    last_run[
                        "terminal_variance"
                    ]
                ),
                "terminal_variance_state_mean": (
                    last_run.get(
                        "terminal_variance_state_mean"
                    )
                ),
            }
        )

    return {
        "engine": ENGINE_NAME,
        "benchmark": (
            "batched Monte Carlo "
            "simulation kernel"
        ),
        "model": request.model,
        "steps": request.steps,
        "maturity": (
            request.maturity
        ),
        "batch_size": (
            request.batch_size
        ),
        "repeats": (
            request.repeats
        ),
        "notes": [
            (
                "Runtime measures backend NumPy simulation only; "
                "network round-trip and chart JSON serialization "
                "are excluded."
            ),
            (
                "Large simulations are processed in batches so "
                "the engine does not retain every full path in memory."
            ),
        ],
        "results": results,
    }
# ============================================================
# CORRELATED BROWNIAN / CHOLESKY ENGINE
# ============================================================

class CorrelationSimulationRequest(BaseModel):
    correlation_matrix: list[list[float]]
    labels: list[str] | None = None

    samples: int = Field(
        default=100_000,
        ge=1_000,
        le=500_000,
    )

    batch_size: int = Field(
        default=50_000,
        ge=1_000,
        le=100_000,
    )

    seed: int = 42

    preview_rows: int = Field(
        default=250,
        ge=0,
        le=500,
    )


def _validate_correlation_matrix(
    matrix: list[list[float]],
) -> np.ndarray:
    if len(matrix) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "correlation_matrix must contain "
                "at least 2 rows."
            ),
        )

    dimension = len(matrix)

    if dimension > 50:
        raise HTTPException(
            status_code=400,
            detail=(
                "correlation_matrix dimension "
                "cannot exceed 50."
            ),
        )

    if any(
        len(row) != dimension
        for row in matrix
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "correlation_matrix must be square."
            ),
        )

    corr = np.asarray(
        matrix,
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(corr)
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "correlation_matrix contains "
                "non-finite values."
            ),
        )

    if not np.allclose(
        corr,
        corr.T,
        atol=1e-10,
        rtol=0.0,
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "correlation_matrix must be symmetric."
            ),
        )

    diagonal = np.diag(corr)

    if not np.allclose(
        diagonal,
        np.ones(dimension),
        atol=1e-10,
        rtol=0.0,
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Every diagonal correlation "
                "must equal 1."
            ),
        )

    if np.any(corr < -1.0) or np.any(corr > 1.0):
        raise HTTPException(
            status_code=400,
            detail=(
                "All correlation values must "
                "lie between -1 and 1."
            ),
        )

    try:
        np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "correlation_matrix must be "
                "positive definite for Cholesky "
                "factorization."
            ),
        ) from exc

    return corr


@router.post("/correlation")
def simulate_correlated_brownian(
    request: CorrelationSimulationRequest,
):
    corr = _validate_correlation_matrix(
        request.correlation_matrix
    )

    dimension = corr.shape[0]

    if request.labels is None:
        labels = [
            f"Factor {i + 1}"
            for i in range(dimension)
        ]
    else:
        if len(request.labels) != dimension:
            raise HTTPException(
                status_code=400,
                detail=(
                    "labels length must match "
                    "correlation_matrix dimension."
                ),
            )

        labels = request.labels

    cholesky = np.linalg.cholesky(
        corr
    )

    eigenvalues = np.linalg.eigvalsh(
        corr
    )

    condition_number = float(
        np.linalg.cond(corr)
    )

    rng = np.random.default_rng(
        request.seed
    )

    sum_vector = np.zeros(
        dimension,
        dtype=np.float64,
    )

    sum_outer = np.zeros(
        (dimension, dimension),
        dtype=np.float64,
    )

    preview: list[list[float]] = []

    processed = 0
    batches = 0

    start = time.perf_counter()

    while processed < request.samples:
        n = min(
            request.batch_size,
            request.samples - processed,
        )

        independent = rng.standard_normal(
            size=(n, dimension)
        )

        correlated = (
            independent
            @ cholesky.T
        )

        sum_vector += np.sum(
            correlated,
            axis=0,
        )

        sum_outer += (
            correlated.T
            @ correlated
        )

        remaining_preview = (
            request.preview_rows
            - len(preview)
        )

        if remaining_preview > 0:
            take = min(
                remaining_preview,
                n,
            )

            preview.extend(
                correlated[:take]
                .astype(float)
                .tolist()
            )

        processed += n
        batches += 1

    runtime_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    mean_vector = (
        sum_vector
        / request.samples
    )

    if request.samples > 1:
        covariance = (
            sum_outer
            - request.samples
            * np.outer(
                mean_vector,
                mean_vector,
            )
        ) / (
            request.samples - 1
        )
    else:
        covariance = np.zeros(
            (dimension, dimension),
            dtype=np.float64,
        )

    empirical_std = np.sqrt(
        np.maximum(
            np.diag(covariance),
            0.0,
        )
    )

    denominator = np.outer(
        empirical_std,
        empirical_std,
    )

    empirical_corr = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(
            covariance
        ),
        where=denominator > 0.0,
    )

    np.fill_diagonal(
        empirical_corr,
        1.0,
    )

    absolute_error = np.abs(
        empirical_corr - corr
    )

    off_diagonal_mask = ~np.eye(
        dimension,
        dtype=bool,
    )

    if dimension > 1:
        max_abs_error = float(
            np.max(
                absolute_error[
                    off_diagonal_mask
                ]
            )
        )

        mean_abs_error = float(
            np.mean(
                absolute_error[
                    off_diagonal_mask
                ]
            )
        )
    else:
        max_abs_error = 0.0
        mean_abs_error = 0.0

    runtime_seconds = (
        runtime_ms / 1000.0
    )

    total_random_draws = (
        request.samples
        * dimension
    )

    if runtime_seconds > 0.0:
        samples_per_second = (
            request.samples
            / runtime_seconds
        )

        random_draws_per_second = (
            total_random_draws
            / runtime_seconds
        )
    else:
        samples_per_second = None
        random_draws_per_second = None

    return {
        "engine": ENGINE_NAME,
        "method": (
            "Cholesky-correlated "
            "Gaussian factor simulation"
        ),

        "inputs": {
            "dimension": dimension,
            "labels": labels,
            "samples": request.samples,
            "batch_size": (
                request.batch_size
            ),
            "seed": request.seed,
        },

        "target_correlation": (
            corr.astype(float).tolist()
        ),

        "cholesky_factor": (
            cholesky
            .astype(float)
            .tolist()
        ),

        "empirical_correlation": (
            empirical_corr
            .astype(float)
            .tolist()
        ),

        "empirical_covariance": (
            covariance
            .astype(float)
            .tolist()
        ),

        "empirical_means": (
            mean_vector
            .astype(float)
            .tolist()
        ),

        "empirical_std_devs": (
            empirical_std
            .astype(float)
            .tolist()
        ),

        "validation": {
            "max_absolute_correlation_error": (
                max_abs_error
            ),
            "mean_absolute_correlation_error": (
                mean_abs_error
            ),
            "minimum_eigenvalue": float(
                np.min(eigenvalues)
            ),
            "maximum_eigenvalue": float(
                np.max(eigenvalues)
            ),
            "condition_number": (
                condition_number
            ),
            "positive_definite": True,
        },

        "benchmark": {
            "runtime_ms": runtime_ms,
            "batches": batches,
            "total_random_draws": (
                total_random_draws
            ),
            "samples_per_second": (
                samples_per_second
            ),
            "random_draws_per_second": (
                random_draws_per_second
            ),
        },

        "preview": preview,

        "notes": [
            (
                "The target matrix is validated as "
                "symmetric, unit-diagonal, bounded "
                "between -1 and 1, and positive "
                "definite."
            ),
            (
                "Simulation is batched and accumulates "
                "moments online; the full sample matrix "
                "is not retained in memory."
            ),
            (
                "Empirical correlation is estimated from "
                "the simulated correlated Gaussian factors."
            ),
        ],
    }

# ============================================================
# NUMERICAL CONVERGENCE ENGINE
# ============================================================

class ConvergenceRequest(BaseModel):
    s0: float = Field(default=100.0, gt=0.0)
    mu: float = 0.05
    sigma: float = Field(default=0.20, ge=0.0)
    maturity: float = Field(default=1.0, gt=0.0, le=10.0)

    step_counts: list[int] = Field(
        default_factory=lambda: [8, 16, 32, 64, 128, 256],
        min_length=3,
        max_length=10,
    )

    paths: int = Field(
        default=100_000,
        ge=5_000,
        le=500_000,
    )

    batch_size: int = Field(
        default=25_000,
        ge=1_000,
        le=100_000,
    )

    seed: int = 42


def _fit_loglog_order(
    dts: list[float],
    errors: list[float],
) -> float | None:
    valid = [
        (dt, err)
        for dt, err in zip(dts, errors)
        if dt > 0.0
        and err > 0.0
        and math.isfinite(dt)
        and math.isfinite(err)
    ]

    if len(valid) < 2:
        return None

    x = np.log(
        np.asarray(
            [item[0] for item in valid],
            dtype=np.float64,
        )
    )

    y = np.log(
        np.asarray(
            [item[1] for item in valid],
            dtype=np.float64,
        )
    )

    slope = np.polyfit(
        x,
        y,
        1,
    )[0]

    return float(slope)


@router.post("/convergence")
def stochastic_convergence(
    request: ConvergenceRequest,
):
    cleaned_steps = sorted(
        set(request.step_counts)
    )

    if len(cleaned_steps) < 3:
        raise HTTPException(
            status_code=400,
            detail=(
                "Provide at least 3 distinct "
                "step_counts for convergence fitting."
            ),
        )

    if any(
        steps < 1
        for steps in cleaned_steps
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Every step count must be >= 1."
            ),
        )

    if max(cleaned_steps) > 4096:
        raise HTTPException(
            status_code=400,
            detail=(
                "step_counts cannot exceed 4096."
            ),
        )

    total_state_updates = (
        request.paths
        * sum(cleaned_steps)
    )

    max_total_state_updates = (
        500_000_000
    )

    if (
        total_state_updates
        > max_total_state_updates
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Convergence request requires "
                f"{total_state_updates:,} state updates. "
                f"Maximum is "
                f"{max_total_state_updates:,}."
            ),
        )

    theoretical_mean = (
        request.s0
        * math.exp(
            request.mu
            * request.maturity
        )
    )

    theoretical_variance = (
        request.s0
        * request.s0
        * math.exp(
            2.0
            * request.mu
            * request.maturity
        )
        * (
            math.exp(
                request.sigma
                * request.sigma
                * request.maturity
            )
            - 1.0
        )
    )

    results = []

    for steps in cleaned_steps:
        dt = (
            request.maturity
            / steps
        )

        sqrt_dt = math.sqrt(
            dt
        )

        rng = np.random.default_rng(
            request.seed
            + steps
        )

        processed = 0
        batches = 0

        em_abs_error_sum = 0.0
        milstein_abs_error_sum = 0.0

        em_sq_error_sum = 0.0
        milstein_sq_error_sum = 0.0

        em_terminal_sum = 0.0
        milstein_terminal_sum = 0.0
        exact_terminal_sum = 0.0

        start = time.perf_counter()

        while processed < request.paths:
            n = min(
                request.batch_size,
                request.paths
                - processed,
            )

            em = np.full(
                n,
                request.s0,
                dtype=np.float64,
            )

            milstein = np.full(
                n,
                request.s0,
                dtype=np.float64,
            )

            brownian_terminal = np.zeros(
                n,
                dtype=np.float64,
            )

            for _ in range(steps):
                z = rng.standard_normal(
                    n
                )

                d_w = (
                    sqrt_dt
                    * z
                )

                brownian_terminal += d_w

                em = (
                    em
                    + request.mu
                    * em
                    * dt
                    + request.sigma
                    * em
                    * d_w
                )

                milstein = (
                    milstein
                    + request.mu
                    * milstein
                    * dt
                    + request.sigma
                    * milstein
                    * d_w
                    + 0.5
                    * request.sigma
                    * request.sigma
                    * milstein
                    * (
                        d_w * d_w
                        - dt
                    )
                )

            exact = (
                request.s0
                * np.exp(
                    (
                        request.mu
                        - 0.5
                        * request.sigma
                        * request.sigma
                    )
                    * request.maturity
                    + request.sigma
                    * brownian_terminal
                )
            )

            em_error = (
                em - exact
            )

            milstein_error = (
                milstein - exact
            )

            em_abs_error_sum += float(
                np.sum(
                    np.abs(
                        em_error
                    )
                )
            )

            milstein_abs_error_sum += float(
                np.sum(
                    np.abs(
                        milstein_error
                    )
                )
            )

            em_sq_error_sum += float(
                np.dot(
                    em_error,
                    em_error,
                )
            )

            milstein_sq_error_sum += float(
                np.dot(
                    milstein_error,
                    milstein_error,
                )
            )

            em_terminal_sum += float(
                np.sum(em)
            )

            milstein_terminal_sum += float(
                np.sum(milstein)
            )

            exact_terminal_sum += float(
                np.sum(exact)
            )

            processed += n
            batches += 1

        runtime_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        em_strong_mae = (
            em_abs_error_sum
            / request.paths
        )

        milstein_strong_mae = (
            milstein_abs_error_sum
            / request.paths
        )

        em_strong_rmse = math.sqrt(
            em_sq_error_sum
            / request.paths
        )

        milstein_strong_rmse = math.sqrt(
            milstein_sq_error_sum
            / request.paths
        )

        em_mean = (
            em_terminal_sum
            / request.paths
        )

        milstein_mean = (
            milstein_terminal_sum
            / request.paths
        )

        exact_mc_mean = (
            exact_terminal_sum
            / request.paths
        )

        em_weak_error = abs(
            em_mean
            - theoretical_mean
        )

        milstein_weak_error = abs(
            milstein_mean
            - theoretical_mean
        )

        exact_mc_mean_error = abs(
            exact_mc_mean
            - theoretical_mean
        )

        runtime_seconds = (
            runtime_ms
            / 1000.0
        )

        state_updates = (
            request.paths
            * steps
        )

        if runtime_seconds > 0.0:
            state_updates_per_second = (
                state_updates
                / runtime_seconds
            )
        else:
            state_updates_per_second = None

        results.append(
            {
                "steps": steps,
                "dt": float(dt),
                "paths": (
                    request.paths
                ),
                "batches": batches,
                "state_updates": (
                    state_updates
                ),

                "euler_maruyama": {
                    "strong_mae": float(
                        em_strong_mae
                    ),
                    "strong_rmse": float(
                        em_strong_rmse
                    ),
                    "terminal_mean": float(
                        em_mean
                    ),
                    "weak_mean_error": float(
                        em_weak_error
                    ),
                },

                "milstein": {
                    "strong_mae": float(
                        milstein_strong_mae
                    ),
                    "strong_rmse": float(
                        milstein_strong_rmse
                    ),
                    "terminal_mean": float(
                        milstein_mean
                    ),
                    "weak_mean_error": float(
                        milstein_weak_error
                    ),
                },

                "exact_gbm": {
                    "terminal_mean": float(
                        exact_mc_mean
                    ),
                    "mean_sampling_error": float(
                        exact_mc_mean_error
                    ),
                },

                "runtime_ms": float(
                    runtime_ms
                ),
                "state_updates_per_second": (
                    state_updates_per_second
                ),
            }
        )

    dts = [
        item["dt"]
        for item in results
    ]

    em_strong_errors = [
        item[
            "euler_maruyama"
        ]["strong_mae"]
        for item in results
    ]

    milstein_strong_errors = [
        item[
            "milstein"
        ]["strong_mae"]
        for item in results
    ]

    em_rmse_errors = [
        item[
            "euler_maruyama"
        ]["strong_rmse"]
        for item in results
    ]

    milstein_rmse_errors = [
        item[
            "milstein"
        ]["strong_rmse"]
        for item in results
    ]

    em_measured_order_mae = (
        _fit_loglog_order(
            dts,
            em_strong_errors,
        )
    )

    milstein_measured_order_mae = (
        _fit_loglog_order(
            dts,
            milstein_strong_errors,
        )
    )

    em_measured_order_rmse = (
        _fit_loglog_order(
            dts,
            em_rmse_errors,
        )
    )

    milstein_measured_order_rmse = (
        _fit_loglog_order(
            dts,
            milstein_rmse_errors,
        )
    )

    return {
        "engine": ENGINE_NAME,
        "experiment": (
            "GBM strong convergence: "
            "Euler-Maruyama vs Milstein "
            "against exact terminal solution"
        ),

        "model": "gbm",

        "inputs": {
            "s0": request.s0,
            "mu": request.mu,
            "sigma": (
                request.sigma
            ),
            "maturity": (
                request.maturity
            ),
            "paths": request.paths,
            "batch_size": (
                request.batch_size
            ),
            "seed": request.seed,
            "step_counts": (
                cleaned_steps
            ),
        },

        "theoretical_terminal": {
            "mean": float(
                theoretical_mean
            ),
            "variance": float(
                theoretical_variance
            ),
            "std_dev": float(
                math.sqrt(
                    max(
                        theoretical_variance,
                        0.0,
                    )
                )
            ),
        },

        "measured_orders": {
            "euler_maruyama_strong_mae": (
                em_measured_order_mae
            ),
            "milstein_strong_mae": (
                milstein_measured_order_mae
            ),
            "euler_maruyama_strong_rmse": (
                em_measured_order_rmse
            ),
            "milstein_strong_rmse": (
                milstein_measured_order_rmse
            ),
        },

        "reference_orders": {
            "euler_maruyama_strong": 0.5,
            "milstein_strong": 1.0,
        },

        "results": results,

        "notes": [
            (
                "For each discretization, Euler-Maruyama, "
                "Milstein, and the exact GBM terminal value "
                "use the same Brownian increments, enabling "
                "a coupled strong-error comparison."
            ),
            (
                "Strong convergence order is estimated by "
                "a least-squares fit of log(error) against "
                "log(dt)."
            ),
            (
                "Weak mean error includes Monte Carlo sampling "
                "noise, so it need not decrease monotonically "
                "at every resolution."
            ),
            (
                "Runtime measures backend NumPy simulation "
                "and error calculation only; network round-trip "
                "is excluded."
            ),
        ],
    }

# ============================================================
# HESTON IMPLIED-VOLATILITY SURFACE CALIBRATION
# ============================================================

from scipy.optimize import brentq, least_squares


class HestonCalibrationQuote(BaseModel):
    strike: float = Field(gt=0.0)
    maturity: float = Field(gt=0.0, le=10.0)
    implied_vol: float = Field(gt=0.001, le=3.0)
    weight: float = Field(default=1.0, gt=0.0, le=100.0)


class HestonCalibrationRequest(BaseModel):
    spot: float = Field(default=100.0, gt=0.0)
    rate: float = 0.03
    quotes: list[HestonCalibrationQuote] = Field(min_length=5, max_length=100)

    initial_v0: float = Field(default=0.04, gt=0.0001, le=1.0)
    initial_kappa: float = Field(default=2.0, gt=0.01, le=20.0)
    initial_theta: float = Field(default=0.04, gt=0.0001, le=1.0)
    initial_xi: float = Field(default=0.30, gt=0.001, le=5.0)
    initial_rho: float = Field(default=-0.70, ge=-0.999, le=0.999)

    max_nfev: int = Field(default=120, ge=10, le=500)
    quadrature_points: int = Field(default=64, ge=32, le=128)
    integration_upper: float = Field(default=100.0, ge=30.0, le=250.0)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _black_scholes_call_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> float:
    if maturity <= 0.0:
        return max(spot - strike, 0.0)

    if volatility <= 0.0:
        return max(spot - strike * math.exp(-rate * maturity), 0.0)

    sqrt_t = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate + 0.5 * volatility * volatility) * maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    return (
        spot * _normal_cdf(d1)
        - strike * math.exp(-rate * maturity) * _normal_cdf(d2)
    )


def _black_scholes_vega(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> float:
    if maturity <= 0.0 or volatility <= 0.0:
        return 0.0

    sqrt_t = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate + 0.5 * volatility * volatility) * maturity
    ) / (volatility * sqrt_t)

    return spot * _normal_pdf(d1) * sqrt_t


def _implied_vol_from_call_price(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
) -> float | None:
    discounted_strike = strike * math.exp(-rate * maturity)
    lower = max(spot - discounted_strike, 0.0)
    upper = spot

    clipped_price = min(max(price, lower), upper)

    if clipped_price <= lower + 1e-10:
        return 1e-6

    if clipped_price >= upper - 1e-10:
        return None

    def objective(volatility: float) -> float:
        return (
            _black_scholes_call_price(
                spot,
                strike,
                maturity,
                rate,
                volatility,
            )
            - clipped_price
        )

    try:
        return float(brentq(objective, 1e-6, 5.0, maxiter=100))
    except (ValueError, RuntimeError):
        return None


def _heston_return_characteristic_function(
    u: np.ndarray,
    maturity: float,
    rate: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> np.ndarray:
    i = 1j

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        c1 = kappa * theta
        c2 = -np.sqrt(
            (rho * xi * u * i - kappa) ** 2
            - xi * xi * (-u * i - u * u)
        )
        c3 = (
            kappa - rho * xi * u * i + c2
        ) / (
            kappa - rho * xi * u * i - c2
        )

        exp_c2t = np.exp(c2 * maturity)

        h1 = (
            rate * u * i * maturity
            + (c1 / (xi * xi))
            * (
                (kappa - rho * xi * u * i + c2) * maturity
                - 2.0
                * np.log(
                    (1.0 - c3 * exp_c2t)
                    / (1.0 - c3)
                )
            )
        )

        h2 = (
            (kappa - rho * xi * u * i + c2)
            / (xi * xi)
            * (
                (1.0 - exp_c2t)
                / (1.0 - c3 * exp_c2t)
            )
        )

        values = np.exp(h1 + h2 * v0)

    return values


def _heston_prices_for_quotes(
    *,
    spot: float,
    rate: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    quadrature_points: int,
    integration_upper: float,
) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    u = 0.5 * (nodes + 1.0) * integration_upper
    quadrature_weights = 0.5 * integration_upper * weights
    shifted_u = u - 0.5j
    denominator = u * u + 0.25

    prices = np.empty(len(strikes), dtype=np.float64)

    for maturity in np.unique(maturities):
        indices = np.flatnonzero(maturities == maturity)

        cf = _heston_return_characteristic_function(
            shifted_u,
            float(maturity),
            rate,
            v0,
            kappa,
            theta,
            xi,
            rho,
        )

        if not np.all(np.isfinite(cf)):
            raise FloatingPointError("Non-finite Heston characteristic function")

        for index in indices:
            strike = float(strikes[index])
            log_moneyness = math.log(spot / strike)

            integrand = (
                np.real(
                    np.exp(1j * u * log_moneyness)
                    * cf
                )
                / denominator
            )

            integral = float(
                np.dot(quadrature_weights, integrand)
            )

            call_price = (
                spot
                - math.exp(-rate * float(maturity))
                * math.sqrt(spot * strike)
                / math.pi
                * integral
            )

            discounted_strike = strike * math.exp(-rate * float(maturity))
            lower = max(spot - discounted_strike, 0.0)
            upper = spot
            prices[index] = min(max(call_price, lower), upper)

    return prices


@router.post("/calibrate/heston")
def calibrate_heston(
    request: HestonCalibrationRequest,
):
    strikes = np.asarray(
        [quote.strike for quote in request.quotes],
        dtype=np.float64,
    )
    maturities = np.asarray(
        [quote.maturity for quote in request.quotes],
        dtype=np.float64,
    )
    market_vols = np.asarray(
        [quote.implied_vol for quote in request.quotes],
        dtype=np.float64,
    )
    quote_weights = np.asarray(
        [quote.weight for quote in request.quotes],
        dtype=np.float64,
    )

    market_prices = np.asarray(
        [
            _black_scholes_call_price(
                request.spot,
                float(strike),
                float(maturity),
                request.rate,
                float(volatility),
            )
            for strike, maturity, volatility in zip(
                strikes,
                maturities,
                market_vols,
            )
        ],
        dtype=np.float64,
    )

    vegas = np.asarray(
        [
            _black_scholes_vega(
                request.spot,
                float(strike),
                float(maturity),
                request.rate,
                float(volatility),
            )
            for strike, maturity, volatility in zip(
                strikes,
                maturities,
                market_vols,
            )
        ],
        dtype=np.float64,
    )

    # Price errors divided by Black-Scholes vega are approximately
    # implied-volatility errors, which balances quotes across strikes.
    residual_scale = np.maximum(vegas, 1.0)
    sqrt_weights = np.sqrt(quote_weights)

    x0 = np.asarray(
        [
            request.initial_v0,
            request.initial_kappa,
            request.initial_theta,
            request.initial_xi,
            request.initial_rho,
        ],
        dtype=np.float64,
    )

    lower_bounds = np.asarray(
        [0.001, 0.05, 0.001, 0.02, -0.98],
        dtype=np.float64,
    )
    upper_bounds = np.asarray(
        [0.50, 15.0, 0.50, 3.0, 0.98],
        dtype=np.float64,
    )

    x0 = np.minimum(
        np.maximum(x0, lower_bounds + 1e-8),
        upper_bounds - 1e-8,
    )

    evaluations = 0

    def residuals(parameters: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        evaluations += 1

        v0, kappa, theta, xi, rho = [float(x) for x in parameters]

        try:
            model_prices = _heston_prices_for_quotes(
                spot=request.spot,
                rate=request.rate,
                strikes=strikes,
                maturities=maturities,
                v0=v0,
                kappa=kappa,
                theta=theta,
                xi=xi,
                rho=rho,
                quadrature_points=request.quadrature_points,
                integration_upper=request.integration_upper,
            )

            raw = (
                (model_prices - market_prices)
                / residual_scale
            ) * sqrt_weights

            if not np.all(np.isfinite(raw)):
                raise FloatingPointError("Non-finite calibration residual")

            return raw

        except (FloatingPointError, OverflowError, ValueError):
            return np.full(
                len(request.quotes),
                1e3,
                dtype=np.float64,
            )

    start = time.perf_counter()

    result = least_squares(
        residuals,
        x0=x0,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=request.max_nfev,
        method="trf",
        x_scale="jac",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )

    runtime_ms = (time.perf_counter() - start) * 1000.0

    fitted_v0, fitted_kappa, fitted_theta, fitted_xi, fitted_rho = [
        float(x) for x in result.x
    ]

    fitted_prices = _heston_prices_for_quotes(
        spot=request.spot,
        rate=request.rate,
        strikes=strikes,
        maturities=maturities,
        v0=fitted_v0,
        kappa=fitted_kappa,
        theta=fitted_theta,
        xi=fitted_xi,
        rho=fitted_rho,
        quadrature_points=request.quadrature_points,
        integration_upper=request.integration_upper,
    )

    fitted_vols: list[float | None] = []
    quote_results = []
    squared_vol_errors = []
    absolute_vol_errors = []
    squared_price_errors = []

    for index, quote in enumerate(request.quotes):
        fitted_vol = _implied_vol_from_call_price(
            float(fitted_prices[index]),
            request.spot,
            quote.strike,
            quote.maturity,
            request.rate,
        )

        fitted_vols.append(fitted_vol)

        price_error = float(fitted_prices[index] - market_prices[index])
        squared_price_errors.append(price_error * price_error)

        if fitted_vol is not None:
            vol_error = float(fitted_vol - quote.implied_vol)
            squared_vol_errors.append(vol_error * vol_error)
            absolute_vol_errors.append(abs(vol_error))
        else:
            vol_error = None

        quote_results.append(
            {
                "strike": quote.strike,
                "maturity": quote.maturity,
                "weight": quote.weight,
                "market_implied_vol": quote.implied_vol,
                "model_implied_vol": fitted_vol,
                "implied_vol_error": vol_error,
                "market_call_price": float(market_prices[index]),
                "model_call_price": float(fitted_prices[index]),
                "price_error": price_error,
            }
        )

    price_rmse = math.sqrt(
        sum(squared_price_errors) / len(squared_price_errors)
    )

    if squared_vol_errors:
        implied_vol_rmse = math.sqrt(
            sum(squared_vol_errors) / len(squared_vol_errors)
        )
        implied_vol_mae = sum(absolute_vol_errors) / len(absolute_vol_errors)
        max_implied_vol_abs_error = max(absolute_vol_errors)
    else:
        implied_vol_rmse = None
        implied_vol_mae = None
        max_implied_vol_abs_error = None

    feller_left = 2.0 * fitted_kappa * fitted_theta
    feller_right = fitted_xi * fitted_xi

    return {
        "engine": ENGINE_NAME,
        "method": (
            "Heston semi-closed-form pricing with Gauss-Legendre "
            "quadrature and bounded SciPy least-squares calibration"
        ),
        "objective": (
            "Vega-scaled option-price residuals, approximately "
            "implied-volatility residuals"
        ),
        "inputs": {
            "spot": request.spot,
            "rate": request.rate,
            "quote_count": len(request.quotes),
            "quadrature_points": request.quadrature_points,
            "integration_upper": request.integration_upper,
            "max_nfev": request.max_nfev,
        },
        "initial_parameters": {
            "v0": request.initial_v0,
            "kappa": request.initial_kappa,
            "theta": request.initial_theta,
            "xi": request.initial_xi,
            "rho": request.initial_rho,
        },
        "fitted_parameters": {
            "v0": fitted_v0,
            "kappa": fitted_kappa,
            "theta": fitted_theta,
            "xi": fitted_xi,
            "rho": fitted_rho,
        },
        "optimization": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "nfev": int(result.nfev),
            "njev": int(result.njev) if result.njev is not None else None,
            "residual_function_calls": evaluations,
            "runtime_ms": float(runtime_ms),
        },
        "fit_quality": {
            "price_rmse": float(price_rmse),
            "implied_vol_rmse": (
                float(implied_vol_rmse)
                if implied_vol_rmse is not None
                else None
            ),
            "implied_vol_mae": (
                float(implied_vol_mae)
                if implied_vol_mae is not None
                else None
            ),
            "max_implied_vol_absolute_error": (
                float(max_implied_vol_abs_error)
                if max_implied_vol_abs_error is not None
                else None
            ),
        },
        "diagnostics": {
            "feller_left_2kappa_theta": float(feller_left),
            "feller_right_xi_squared": float(feller_right),
            "feller_satisfied": bool(feller_left >= feller_right),
        },
        "quotes": quote_results,
        "notes": [
            (
                "Input market implied volatilities are converted to Black-Scholes "
                "call prices using the supplied continuously compounded rate."
            ),
            (
                "Calibration minimizes vega-scaled price errors so deep ITM/OTM "
                "quotes do not dominate only because of price scale."
            ),
            (
                "The current endpoint assumes zero continuous dividend yield."
            ),
            (
                "The Feller condition is reported as a diagnostic but is not "
                "imposed as a hard optimization constraint."
            ),
        ],
    }
