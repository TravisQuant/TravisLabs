import sys
import time
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="TravisLabs API",
    description="Quantitative finance API for TravisLabs.",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# C++ DERIVATIVES ENGINE
# ============================================================
#
# Project structure:
#
# TravisLabs/
# ├── api/
# │   └── main.py
# └── cpp/
#     ├── option_pricer.cpp
#     ├── bindings.cpp
#     └── travislabs_pricer....pyd
#
# Locally on Windows, pybind11 builds a .pyd module.
# Later Railway will build the corresponding Linux extension.
# ============================================================

CPP_DIR = (
    Path(__file__)
    .resolve()
    .parent
    / "cpp"
)

if str(CPP_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(CPP_DIR),
    )


try:
    import travislabs_pricer

    CPP_PRICER_AVAILABLE = True
    CPP_PRICER_ERROR = None

except Exception as exc:
    travislabs_pricer = None
    CPP_PRICER_AVAILABLE = False
    CPP_PRICER_ERROR = str(exc)


# ============================================================
# BANK OF CANADA CURVE CONFIGURATION
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
        "message": "TravisLabs API is running",
        "cpp_derivatives_engine": CPP_PRICER_AVAILABLE,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cpp_derivatives_engine": {
            "available": CPP_PRICER_AVAILABLE,
            "error": CPP_PRICER_ERROR,
        },
    }


# ============================================================
# RATES — DURATION / CONVEXITY SHOCK
# ============================================================

@app.post("/rates/shock")
def rate_shock(
    request: RateShockRequest,
):
    delta_y = (
        request.rate_shock_bp
        / 10000.0
    )

    percentage_change = (
        -request.duration
        * delta_y
        +
        0.5
        * request.convexity
        * delta_y**2
    )

    pnl = (
        request.portfolio_value
        * percentage_change
    )

    stressed_value = (
        request.portfolio_value
        + pnl
    )

    return {
        "original_value": request.portfolio_value,
        "rate_shock_bp": request.rate_shock_bp,
        "estimated_return_pct": (
            percentage_change
            * 100.0
        ),
        "estimated_pnl": pnl,
        "stressed_value": stressed_value,
    }


# ============================================================
# RATES — BANK OF CANADA YIELD CURVE
# ============================================================

@app.get("/rates/curve")
def get_rates_curve():
    curve = []
    warnings = []

    for config in BOC_CURVE_SERIES:
        series = config["series"]

        url = (
            "https://www.bankofcanada.ca/"
            f"valet/observations/{series}/json"
            "?recent=10&order_dir=desc"
        )

        try:
            response = requests.get(
                url,
                timeout=10,
            )

            response.raise_for_status()

            data = response.json()

            observations = data.get(
                "observations",
                [],
            )

            selected = None

            for observation in observations:
                series_data = observation.get(
                    series
                )

                if not series_data:
                    continue

                value = series_data.get("v")

                if value in (
                    None,
                    "",
                    "NA",
                    "null",
                ):
                    continue

                try:
                    numeric_value = float(value)
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                selected = {
                    "series": series,
                    "maturity": config[
                        "maturity"
                    ],
                    "label": config[
                        "label"
                    ],
                    "yield": numeric_value,
                    "date": observation.get(
                        "d"
                    ),
                }

                break

            if selected is None:
                warnings.append(
                    f"No valid observation found for {series}"
                )
                continue

            curve.append(selected)

        except Exception as exc:
            warnings.append(
                f"{series}: {str(exc)}"
            )

    curve.sort(
        key=lambda point: point[
            "maturity"
        ]
    )

    if not curve:
        raise HTTPException(
            status_code=502,
            detail={
                "message": (
                    "Unable to retrieve Bank of "
                    "Canada yield curve data."
                ),
                "warnings": warnings,
            },
        )

    dates = [
        point["date"]
        for point in curve
        if point.get("date")
    ]

    as_of = (
        max(dates)
        if dates
        else None
    )

    return {
        "source": "Bank of Canada Valet API",
        "curve_type": (
            "Government of Canada "
            "benchmark bond yields"
        ),
        "as_of": as_of,
        "curve": curve,
        "warnings": warnings,
    }


# ============================================================
# DERIVATIVES — C++ PRICING ENGINE
# ============================================================

@app.post("/derivatives/price")
def derivatives_price(
    request: DerivativesPriceRequest,
):
    # --------------------------------------------------------
    # Make sure compiled C++ module is available
    # --------------------------------------------------------

    if not CPP_PRICER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail={
                "message": (
                    "C++ derivatives engine "
                    "is not available."
                ),
                "error": CPP_PRICER_ERROR,
            },
        )


    # --------------------------------------------------------
    # Validate financial inputs
    # --------------------------------------------------------

    if request.spot <= 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "spot must be greater "
                "than zero"
            ),
        )

    if request.strike <= 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "strike must be greater "
                "than zero"
            ),
        )

    if request.volatility <= 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "volatility must be greater "
                "than zero"
            ),
        )

    if request.maturity <= 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "maturity must be greater "
                "than zero"
            ),
        )


    # --------------------------------------------------------
    # Validate numerical settings
    # --------------------------------------------------------

    if request.binomial_steps < 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "binomial_steps must "
                "be at least 1"
            ),
        )

    if request.mc_paths < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "mc_paths must "
                "be at least 2"
            ),
        )

    if request.pde_space_steps < 3:
        raise HTTPException(
            status_code=400,
            detail=(
                "pde_space_steps must "
                "be at least 3"
            ),
        )

    if request.pde_time_steps < 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "pde_time_steps must "
                "be at least 1"
            ),
        )


    # ========================================================
    # BLACK-SCHOLES
    # ========================================================

    start = time.perf_counter()

    black_scholes = (
        travislabs_pricer
        .black_scholes_call(
            request.spot,
            request.strike,
            request.rate,
            request.volatility,
            request.maturity,
        )
    )

    black_scholes_runtime_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    black_scholes[
        "runtime_ms"
    ] = black_scholes_runtime_ms


    benchmark_price = (
        black_scholes["price"]
    )


    # ========================================================
    # CRR BINOMIAL
    # ========================================================

    start = time.perf_counter()

    binomial = (
        travislabs_pricer
        .binomial_call(
            request.spot,
            request.strike,
            request.rate,
            request.volatility,
            request.maturity,
            request.binomial_steps,
        )
    )

    binomial_runtime_ms = (
        time.perf_counter()
        - start
    ) * 1000.0


    binomial[
        "absolute_error"
    ] = abs(
        binomial["price"]
        - benchmark_price
    )

    binomial[
        "runtime_ms"
    ] = binomial_runtime_ms


    # ========================================================
    # MONTE CARLO
    # ========================================================

    start = time.perf_counter()

    monte_carlo = (
        travislabs_pricer
        .monte_carlo_call(
            request.spot,
            request.strike,
            request.rate,
            request.volatility,
            request.maturity,
            request.mc_paths,
        )
    )

    monte_carlo_runtime_ms = (
        time.perf_counter()
        - start
    ) * 1000.0


    monte_carlo[
        "absolute_error"
    ] = abs(
        monte_carlo["price"]
        - benchmark_price
    )

    monte_carlo[
        "runtime_ms"
    ] = monte_carlo_runtime_ms


    # ========================================================
    # CRANK-NICOLSON PDE
    # ========================================================

    start = time.perf_counter()

    crank_nicolson = (
        travislabs_pricer
        .crank_nicolson_call(
            request.spot,
            request.strike,
            request.rate,
            request.volatility,
            request.maturity,
            request.pde_space_steps,
            request.pde_time_steps,
        )
    )

    crank_nicolson_runtime_ms = (
        time.perf_counter()
        - start
    ) * 1000.0


    crank_nicolson[
        "absolute_error"
    ] = abs(
        crank_nicolson["price"]
        - benchmark_price
    )

    crank_nicolson[
        "runtime_ms"
    ] = crank_nicolson_runtime_ms


    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "engine": (
            "TravisLabs C++17 via pybind11"
        ),
        "inputs": {
            "spot": request.spot,
            "strike": request.strike,
            "rate": request.rate,
            "volatility": (
                request.volatility
            ),
            "maturity": request.maturity,
        },
        "black_scholes": black_scholes,
        "binomial": binomial,
        "monte_carlo": monte_carlo,
        "crank_nicolson": (
            crank_nicolson
        ),
    }