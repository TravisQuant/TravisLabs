import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(title="TravisLabs Quant API")


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "TravisLabs Quant API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ============================================================
# RATE SHOCK ENGINE
# ============================================================

class RateShockRequest(BaseModel):
    portfolio_value: float
    duration: float
    convexity: float
    rate_shock_bp: float


@app.post("/rates/shock")
def rate_shock(request: RateShockRequest):

    delta_y = request.rate_shock_bp / 10000

    percentage_change = (
        -request.duration * delta_y
        + 0.5
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
        "original_value":
            request.portfolio_value,

        "rate_shock_bp":
            request.rate_shock_bp,

        "estimated_return_pct":
            percentage_change * 100,

        "estimated_pnl":
            pnl,

        "stressed_value":
            stressed_value,
    }


# ============================================================
# BANK OF CANADA BENCHMARK BOND SERIES
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

        # The Bank calls this the
        # long-term benchmark.
        #
        # We plot it near 30Y for
        # visualization purposes.

        "maturity": 30.0,
        "label": "Long",
    },
]


# ============================================================
# BANK OF CANADA CURVE ENDPOINT
# ============================================================

@app.get("/rates/curve")
def get_rates_curve():

    curve = []

    errors = []


    # --------------------------------------------------------
    # Fetch each series separately
    # --------------------------------------------------------
    #
    # This makes debugging easier and avoids one bad series
    # causing the entire combined request to fail.
    # --------------------------------------------------------

    for item in BOC_CURVE_SERIES:

        series = item["series"]

        url = (
            "https://www.bankofcanada.ca/"
            "valet/observations/"
            f"{series}/json"
        )


        try:

            response = requests.get(
                url,
                params={
                    "recent": 10,
                    "order_dir": "desc",
                },
                headers={
                    "Accept":
                        "application/json",

                    "User-Agent":
                        "TravisLabs/1.0",
                },
                timeout=15,
            )


        except requests.RequestException as error:

            errors.append({
                "series": series,
                "error": str(error),
            })

            continue


        # ----------------------------------------------------
        # HTTP error
        # ----------------------------------------------------

        if response.status_code != 200:

            errors.append({
                "series":
                    series,

                "status_code":
                    response.status_code,

                "response":
                    response.text[:500],
            })

            continue


        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        try:

            data = response.json()

        except ValueError:

            errors.append({
                "series":
                    series,

                "error":
                    "Response was not valid JSON",
            })

            continue


        observations = data.get(
            "observations",
            []
        )


        # ----------------------------------------------------
        # Find newest valid observation
        # ----------------------------------------------------

        latest_value = None
        latest_date = None


        for observation in observations:

            value_object = observation.get(
                series
            )

            if not value_object:
                continue


            value = value_object.get("v")


            if value in (
                None,
                "",
                "NA",
                "na",
            ):
                continue


            try:

                latest_value = float(
                    value
                )

                latest_date = observation.get(
                    "d"
                )

                break

            except (
                ValueError,
                TypeError,
            ):

                continue


        # ----------------------------------------------------
        # No usable value
        # ----------------------------------------------------

        if latest_value is None:

            errors.append({
                "series":
                    series,

                "error":
                    "No valid recent observation found",
            })

            continue


        # ----------------------------------------------------
        # Add point to curve
        # ----------------------------------------------------

        curve.append({

            "series":
                series,

            "maturity":
                item["maturity"],

            "label":
                item["label"],

            "yield":
                latest_value,

            "date":
                latest_date,
        })


    # ========================================================
    # ERROR CHECK
    # ========================================================

    if not curve:

        raise HTTPException(
            status_code=502,
            detail={
                "message":
                    "Unable to retrieve Bank of Canada benchmark yields.",

                "errors":
                    errors,
            },
        )


    # ========================================================
    # SORT CURVE
    # ========================================================

    curve.sort(
        key=lambda point:
            point["maturity"]
    )


    # ========================================================
    # LATEST DATE
    # ========================================================

    dates = [
        point["date"]
        for point in curve
        if point["date"]
    ]


    as_of = (
        max(dates)
        if dates
        else None
    )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "source":
            "Bank of Canada Valet API",

        "curve_type":
            "Government of Canada benchmark bond yields",

        "as_of":
            as_of,

        "curve":
            curve,

        "warnings":
            errors,
    }