from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="TravisLabs Quant API"
)

# Allow the public website to call this API.
# Fine for our public demo endpoints; we are not using logins/cookies.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        + 0.5 * request.convexity * delta_y**2
    )

    pnl = request.portfolio_value * percentage_change
    stressed_value = request.portfolio_value + pnl

    return {
        "original_value": request.portfolio_value,
        "rate_shock_bp": request.rate_shock_bp,
        "estimated_return_pct": percentage_change * 100,
        "estimated_pnl": pnl,
        "stressed_value": stressed_value
    }