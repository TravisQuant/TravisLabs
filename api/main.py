import sys
import time
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from stochastic_engine import router as stochastic_router


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="TravisLabs API",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STOCHASTIC PROCESS ROUTER
# ============================================================

app.include_router(stochastic_router)


# ============================================================
# C++ DERIVATIVES ENGINE
# ============================================================

CPP_DIR = (
    Path(__file__).resolve().parent
    / "cpp"
)

if str(CPP_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(CPP_DIR),
    )


CPP_PRICER_AVAILABLE = False
CPP_PRICER_ERROR = None

try:
    import travislabs_pricer

    CPP_PRICER_AVAILABLE = True

except Exception as exc:
    CPP_PRICER_ERROR = str(exc)


# ============================================================
# BANK OF CANADA CURVE SERIES
# ============================================================

BOC_CURVE_SERIES = [
    {
        "series": "BD.CDN.2YR.DQ.YLD",
        "maturity": 2.0,
        "label": "2Y",
    },
    {
        "series": "BD.CDN.3YR.DQ.YLD",
        "maturity": 3.0,
        "label": "3Y",
    },
    {
        "series": "BD.CDN.5YR.DQ.YLD",
        "maturity": 5.0,
        "label": "5Y",
    },
    {
        "series": "BD.CDN.7YR.DQ.YLD",
        "maturity": 7.0,
        "label": "7Y",
    },
    {
        "series": "BD.CDN.10YR.DQ.YLD",
        "maturity": 10.0,
        "label": "10Y",
    },
    {
        "series": "BD.CDN.LONG.DQ.YLD",
        "maturity": 30.0,
        "label": "Long",
    },
]


# ============================================================
# REQUEST MODELS
# ============================================================

class RateShockRequest(BaseModel):
    portfolio_value: float
    duration: float
    convexity: float
    rate_shock_bp: float


class DerivativesPriceRequest(BaseModel):
    spot: float
    strike: float
    rate: float
    volatility: float
    maturity: float

    binomial_steps: int = 500
    mc_paths: int = 100000

    pde_space_steps: int = 400
    pde_time_steps: int = 400


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "TravisLabs Quantitative Finance API",
        "services": {
            "rates": True,
            "stochastic_processes": True,
            "cpp_derivatives_engine": {
                "available": (
                    CPP_PRICER_AVAILABLE
                ),
                "error": (
                    CPP_PRICER_ERROR
                ),
            },
        },
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cpp_derivatives_engine": {
            "available": (
                CPP_PRICER_AVAILABLE
            ),
            "error": (
                CPP_PRICER_ERROR
            ),
        },
        "stochastic_engine": {
            "available": True,
            "engine": "Python/NumPy",
        },
    }


# ============================================================
# RATE SHOCK ENGINE
# ============================================================

@app.post("/rates/shock")
def calculate_rate_shock(
    request: RateShockRequest,
):
    """
    Duration / convexity approximation:

        ΔP/P ≈ -D * Δy
                + 0.5 * C * (Δy)^2

    where Δy is expressed in decimal yield units.
    """

    delta_y = (
        request.rate_shock_bp
        / 10000.0
    )

    estimated_return = (
        -request.duration
        * delta_y
        +
        0.5
        * request.convexity
        * delta_y**2
    )

    estimated_pnl = (
        request.portfolio_value
        * estimated_return
    )

    stressed_value = (
        request.portfolio_value
        + estimated_pnl
    )

    return {
        "original_value": (
            request.portfolio_value
        ),
        "rate_shock_bp": (
            request.rate_shock_bp
        ),
        "estimated_return_pct": (
            estimated_return
            * 100.0
        ),
        "estimated_pnl": (
            estimated_pnl
        ),
        "stressed_value": (
            stressed_value
        ),
    }


# ============================================================
# BANK OF CANADA CURVE
# ============================================================

@app.get("/rates/curve")
def get_rates_curve():
    curve = []

    observation_dates = []

    try:
        for instrument in BOC_CURVE_SERIES:
            series = instrument["series"]

            url = (
                "https://www.bankofcanada.ca/"
                "valet/observations/"
                f"{series}/json"
                "?recent=10"
                "&order_dir=desc"
            )

            response = requests.get(
                url,
                timeout=10,
            )

            response.raise_for_status()

            payload = response.json()

            observations = payload.get(
                "observations",
                [],
            )

            latest_value = None
            latest_date = None

            for observation in observations:
                series_data = observation.get(
                    series,
                    {},
                )

                raw_value = (
                    series_data.get("v")
                )

                if raw_value is None:
                    continue

                try:
                    latest_value = float(
                        raw_value
                    )

                    latest_date = (
                        observation.get("d")
                    )

                    break

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

            if latest_value is None:
                continue

            curve.append(
                {
                    "series": series,
                    "label": (
                        instrument["label"]
                    ),
                    "maturity": (
                        instrument[
                            "maturity"
                        ]
                    ),
                    "yield": latest_value,
                    "date": latest_date,
                }
            )

            if latest_date:
                observation_dates.append(
                    latest_date
                )

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to retrieve "
                "Bank of Canada data: "
                f"{exc}"
            ),
        )

    if not curve:
        raise HTTPException(
            status_code=502,
            detail=(
                "No Bank of Canada "
                "yield observations "
                "were available."
            ),
        )

    as_of = (
        max(observation_dates)
        if observation_dates
        else None
    )

    return {
        "source": (
            "Bank of Canada Valet API"
        ),
        "as_of": as_of,
        "curve": curve,
    }


# ============================================================
# C++ DERIVATIVES PRICING
# ============================================================

@app.post("/derivatives/price")
def price_derivative(
    request: DerivativesPriceRequest,
):
    if not CPP_PRICER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail={
                "message": (
                    "C++ derivatives "
                    "engine unavailable"
                ),
                "error": (
                    CPP_PRICER_ERROR
                ),
            },
        )

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    if request.spot <= 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "spot must be "
                "greater than zero"
            ),
        )

    if request.strike <= 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "strike must be "
                "greater than zero"
            ),
        )

    if request.volatility <= 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "volatility must be "
                "greater than zero"
            ),
        )

    if request.maturity <= 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "maturity must be "
                "greater than zero"
            ),
        )

    if request.binomial_steps < 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "binomial_steps "
                "must be >= 1"
            ),
        )

    if request.mc_paths < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "mc_paths must be >= 2"
            ),
        )

    if request.pde_space_steps < 3:
        raise HTTPException(
            status_code=400,
            detail=(
                "pde_space_steps "
                "must be >= 3"
            ),
        )

    if request.pde_time_steps < 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "pde_time_steps "
                "must be >= 1"
            ),
        )

    S = request.spot
    K = request.strike
    r = request.rate
    sigma = request.volatility
    T = request.maturity


    # --------------------------------------------------------
    # BLACK-SCHOLES
    # --------------------------------------------------------

    start = time.perf_counter()

    black_scholes = (
        travislabs_pricer
        .black_scholes_call(
            S,
            K,
            r,
            sigma,
            T,
        )
    )

    black_scholes_runtime_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    bs_price = float(
        black_scholes["price"]
    )


    # --------------------------------------------------------
    # BINOMIAL
    # --------------------------------------------------------

    start = time.perf_counter()

    binomial = (
        travislabs_pricer
        .binomial_call(
            S,
            K,
            r,
            sigma,
            T,
            request.binomial_steps,
        )
    )

    binomial_runtime_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    binomial_price = float(
        binomial["price"]
    )


    # --------------------------------------------------------
    # MONTE CARLO
    # --------------------------------------------------------

    start = time.perf_counter()

    monte_carlo = (
        travislabs_pricer
        .monte_carlo_call(
            S,
            K,
            r,
            sigma,
            T,
            request.mc_paths,
        )
    )

    monte_carlo_runtime_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    monte_carlo_price = float(
        monte_carlo["price"]
    )


    # --------------------------------------------------------
    # CRANK-NICOLSON
    # --------------------------------------------------------

    start = time.perf_counter()

    crank_nicolson = (
        travislabs_pricer
        .crank_nicolson_call(
            S,
            K,
            r,
            sigma,
            T,
            request.pde_space_steps,
            request.pde_time_steps,
        )
    )

    crank_nicolson_runtime_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    crank_nicolson_price = float(
        crank_nicolson["price"]
    )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "engine": (
            "TravisLabs C++17 "
            "via pybind11"
        ),

        "inputs": {
            "spot": S,
            "strike": K,
            "rate": r,
            "volatility": sigma,
            "maturity": T,
        },

        "black_scholes": {
            **black_scholes,
            "runtime_ms": (
                black_scholes_runtime_ms
            ),
        },

        "binomial": {
            **binomial,
            "absolute_error": abs(
                binomial_price
                - bs_price
            ),
            "runtime_ms": (
                binomial_runtime_ms
            ),
        },

        "monte_carlo": {
            **monte_carlo,
            "absolute_error": abs(
                monte_carlo_price
                - bs_price
            ),
            "runtime_ms": (
                monte_carlo_runtime_ms
            ),
        },

        "crank_nicolson": {
            **crank_nicolson,
            "absolute_error": abs(
                crank_nicolson_price
                - bs_price
            ),
            "runtime_ms": (
                crank_nicolson_runtime_ms
            ),
        },
    }