import math
import time
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter(
    prefix="/stochastic",
    tags=["stochastic"],
)


ModelName = Literal[
    "brownian",
    "gbm",
    "ou",
    "vasicek",
    "cir",
    "heston",
]


class StochasticSimulationRequest(BaseModel):
    model: ModelName

    x0: float = 0.0
    s0: float = 100.0
    v0: float = 0.04

    mu: float = 0.05
    sigma: float = 0.20

    kappa: float = 1.0
    theta: float = 0.05

    xi: float = 0.30
    rho: float = -0.70

    maturity: float = Field(default=1.0, gt=0)
    steps: int = Field(default=252, ge=2, le=5000)
    paths: int = Field(default=10000, ge=10, le=500000)
    seed: int = 42

    sample_paths: int = Field(
        default=20,
        ge=1,
        le=50,
    )


def _theoretical_moments(
    request: StochasticSimulationRequest,
):
    model = request.model
    T = request.maturity

    if model == "brownian":
        mean = request.x0 + request.mu * T
        variance = request.sigma**2 * T

        return mean, variance

    if model == "gbm":
        mean = request.s0 * math.exp(
            request.mu * T
        )

        variance = (
            request.s0**2
            * math.exp(
                2.0 * request.mu * T
            )
            * (
                math.exp(
                    request.sigma**2 * T
                )
                - 1.0
            )
        )

        return mean, variance

    if model in {"ou", "vasicek"}:
        decay = math.exp(
            -request.kappa * T
        )

        mean = (
            request.x0 * decay
            + request.theta
            * (1.0 - decay)
        )

        if request.kappa <= 0:
            return mean, None

        variance = (
            request.sigma**2
            / (2.0 * request.kappa)
            * (
                1.0
                - math.exp(
                    -2.0
                    * request.kappa
                    * T
                )
            )
        )

        return mean, variance

    if model == "cir":
        if request.kappa <= 0:
            return None, None

        decay = math.exp(
            -request.kappa * T
        )

        mean = (
            request.x0 * decay
            + request.theta
            * (1.0 - decay)
        )

        variance = (
            request.x0
            * request.sigma**2
            * decay
            * (1.0 - decay)
            / request.kappa
            +
            request.theta
            * request.sigma**2
            * (1.0 - decay) ** 2
            / (2.0 * request.kappa)
        )

        return mean, variance

    # Heston terminal spot moments are not
    # being claimed analytically here.
    return None, None


@router.post("/simulate")
def simulate_stochastic_process(
    request: StochasticSimulationRequest,
):
    if request.sigma < 0:
        raise HTTPException(
            status_code=400,
            detail="sigma must be non-negative",
        )

    if request.kappa < 0:
        raise HTTPException(
            status_code=400,
            detail="kappa must be non-negative",
        )

    if request.xi < 0:
        raise HTTPException(
            status_code=400,
            detail="xi must be non-negative",
        )

    if not -1.0 <= request.rho <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="rho must lie in [-1, 1]",
        )

    start = time.perf_counter()

    rng = np.random.default_rng(
        request.seed
    )

    dt = (
        request.maturity
        / request.steps
    )

    sqrt_dt = math.sqrt(dt)

    model = request.model
    paths = request.paths
    steps = request.steps

    if model in {"gbm", "heston"}:
        initial_value = request.s0
    else:
        initial_value = request.x0

    x = np.full(
        paths,
        initial_value,
        dtype=np.float64,
    )

    if model == "heston":
        variance_state = np.full(
            paths,
            request.v0,
            dtype=np.float64,
        )
    else:
        variance_state = None

    sample_count = min(
        request.sample_paths,
        paths,
    )

    sampled_paths = np.empty(
        (
            sample_count,
            steps + 1,
        ),
        dtype=np.float64,
    )

    sampled_paths[:, 0] = (
        initial_value
    )

    mean_path = np.empty(
        steps + 1,
        dtype=np.float64,
    )

    variance_path = np.empty(
        steps + 1,
        dtype=np.float64,
    )

    mean_path[0] = float(
        np.mean(x)
    )

    variance_path[0] = float(
        np.var(x)
    )

    if model == "heston":
        heston_mean_variance_path = (
            np.empty(
                steps + 1,
                dtype=np.float64,
            )
        )

        heston_mean_variance_path[0] = (
            float(
                np.mean(
                    variance_state
                )
            )
        )
    else:
        heston_mean_variance_path = None

    for step in range(steps):
        z1 = rng.standard_normal(paths)

        if model == "brownian":
            x = (
                x
                + request.mu * dt
                + request.sigma
                * sqrt_dt
                * z1
            )

        elif model == "gbm":
            x = x * np.exp(
                (
                    request.mu
                    - 0.5
                    * request.sigma**2
                )
                * dt
                +
                request.sigma
                * sqrt_dt
                * z1
            )

        elif model in {
            "ou",
            "vasicek",
        }:
            x = (
                x
                + request.kappa
                * (
                    request.theta
                    - x
                )
                * dt
                + request.sigma
                * sqrt_dt
                * z1
            )

        elif model == "cir":
            positive_x = np.maximum(
                x,
                0.0,
            )

            x = (
                x
                + request.kappa
                * (
                    request.theta
                    - positive_x
                )
                * dt
                + request.sigma
                * np.sqrt(
                    positive_x
                )
                * sqrt_dt
                * z1
            )

            # Full truncation style safeguard.
            x = np.maximum(
                x,
                0.0,
            )

        elif model == "heston":
            z2 = rng.standard_normal(
                paths
            )

            correlated_z = (
                request.rho * z1
                + math.sqrt(
                    max(
                        1.0
                        - request.rho**2,
                        0.0,
                    )
                )
                * z2
            )

            old_variance = (
                np.maximum(
                    variance_state,
                    0.0,
                )
            )

            sqrt_variance = np.sqrt(
                old_variance
            )

            # Spot uses the same old variance
            # in drift and diffusion.
            x = x * np.exp(
                (
                    request.mu
                    - 0.5
                    * old_variance
                )
                * dt
                +
                sqrt_variance
                * sqrt_dt
                * z1
            )

            variance_state = (
                variance_state
                + request.kappa
                * (
                    request.theta
                    - old_variance
                )
                * dt
                + request.xi
                * sqrt_variance
                * sqrt_dt
                * correlated_z
            )

            variance_state = np.maximum(
                variance_state,
                0.0,
            )

        sampled_paths[
            :,
            step + 1,
        ] = x[:sample_count]

        mean_path[
            step + 1
        ] = float(
            np.mean(x)
        )

        variance_path[
            step + 1
        ] = float(
            np.var(x)
        )

        if (
            model == "heston"
            and heston_mean_variance_path
            is not None
        ):
            heston_mean_variance_path[
                step + 1
            ] = float(
                np.mean(
                    variance_state
                )
            )

    empirical_mean = float(
        np.mean(x)
    )

    empirical_variance = float(
        np.var(
            x,
            ddof=1,
        )
    )

    empirical_std = math.sqrt(
        empirical_variance
    )

    theoretical_mean, theoretical_variance = (
        _theoretical_moments(
            request
        )
    )

    theoretical_std = (
        math.sqrt(
            theoretical_variance
        )
        if theoretical_variance
        is not None
        else None
    )

    mean_error = (
        abs(
            empirical_mean
            - theoretical_mean
        )
        if theoretical_mean
        is not None
        else None
    )

    variance_error = (
        abs(
            empirical_variance
            - theoretical_variance
        )
        if theoretical_variance
        is not None
        else None
    )

    feller_left = None
    feller_right = None
    feller_satisfied = None

    if model == "cir":
        feller_left = (
            2.0
            * request.kappa
            * request.theta
        )

        feller_right = (
            request.sigma**2
        )

        feller_satisfied = (
            feller_left
            >= feller_right
        )

    half_life = None

    if (
        model
        in {
            "ou",
            "vasicek",
            "cir",
            "heston",
        }
        and request.kappa > 0
    ):
        half_life = (
            math.log(2.0)
            / request.kappa
        )

    elapsed_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    times = np.linspace(
        0.0,
        request.maturity,
        steps + 1,
    )

    terminal_preview_count = min(
        2000,
        paths,
    )

    terminal_preview_indices = (
        np.linspace(
            0,
            paths - 1,
            terminal_preview_count,
            dtype=int,
        )
    )

    response = {
        "engine": (
            "TravisLabs Python/NumPy "
            "stochastic engine"
        ),

        "model": model,

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
            "maturity": (
                request.maturity
            ),
            "steps": steps,
            "paths": paths,
            "seed": request.seed,
            "dt": dt,
        },

        "empirical": {
            "mean": empirical_mean,
            "variance": (
                empirical_variance
            ),
            "std_dev": empirical_std,
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
                feller_left
            ),
            "feller_right_sigma_squared": (
                feller_right
            ),
            "feller_satisfied": (
                feller_satisfied
            ),
            "runtime_ms": elapsed_ms,
        },

        "chart_data": {
            "times": (
                times.tolist()
            ),
            "sample_paths": (
                sampled_paths.tolist()
            ),
            "empirical_mean": (
                mean_path.tolist()
            ),
            "empirical_variance": (
                variance_path.tolist()
            ),
            "terminal_values": (
                x[
                    terminal_preview_indices
                ].tolist()
            ),
        },
    }

    if (
        model == "heston"
        and variance_state
        is not None
    ):
        response[
            "heston"
        ] = {
            "terminal_mean_variance": (
                float(
                    np.mean(
                        variance_state
                    )
                )
            ),
            "terminal_variance_of_variance": (
                float(
                    np.var(
                        variance_state,
                        ddof=1,
                    )
                )
            ),
            "mean_variance_path": (
                heston_mean_variance_path.tolist()
                if heston_mean_variance_path
                is not None
                else None
            ),
        }

    return response