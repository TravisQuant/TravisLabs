#define TRAVISLABS_PYBIND

#include <pybind11/pybind11.h>

#include "option_pricer.cpp"

namespace py = pybind11;


/* ============================================================
 * BLACK-SCHOLES PYTHON WRAPPER
 * ============================================================
 */

py::dict black_scholes_python(
    double spot,
    double strike,
    double rate,
    double volatility,
    double maturity
) {
    const auto result =
        travislabs::black_scholes_call(
            spot,
            strike,
            rate,
            volatility,
            maturity
        );

    py::dict output;

    output["price"] =
        result.price;

    output["delta"] =
        result.delta;

    output["gamma"] =
        result.gamma;

    output["vega"] =
        result.vega;

    output["vega_per_1pct"] =
        result.vega / 100.0;

    output["theta"] =
        result.theta;

    output["theta_per_day"] =
        result.theta / 365.0;

    return output;
}


/* ============================================================
 * BINOMIAL PYTHON WRAPPER
 * ============================================================
 */

py::dict binomial_python(
    double spot,
    double strike,
    double rate,
    double volatility,
    double maturity,
    int steps
) {
    const double price =
        travislabs::binomial_call(
            spot,
            strike,
            rate,
            volatility,
            maturity,
            steps
        );

    py::dict output;

    output["price"] =
        price;

    output["steps"] =
        steps;

    return output;
}


/* ============================================================
 * MONTE CARLO PYTHON WRAPPER
 * ============================================================
 */

py::dict monte_carlo_python(
    double spot,
    double strike,
    double rate,
    double volatility,
    double maturity,
    int paths
) {
    const auto result =
        travislabs::monte_carlo_call(
            spot,
            strike,
            rate,
            volatility,
            maturity,
            paths
        );

    py::dict output;

    output["price"] =
        result.price;

    output["standard_error"] =
        result.standard_error;

    output["ci_lower"] =
        result.ci_lower;

    output["ci_upper"] =
        result.ci_upper;

    output["paths"] =
        paths;

    return output;
}


/* ============================================================
 * CRANK-NICOLSON PYTHON WRAPPER
 * ============================================================
 */

py::dict crank_nicolson_python(
    double spot,
    double strike,
    double rate,
    double volatility,
    double maturity,
    int space_steps,
    int time_steps
) {
    const auto result =
        travislabs::crank_nicolson_call(
            spot,
            strike,
            rate,
            volatility,
            maturity,
            space_steps,
            time_steps
        );

    py::dict output;

    output["price"] =
        result.price;

    output["space_steps"] =
        result.space_steps;

    output["time_steps"] =
        result.time_steps;

    return output;
}


/* ============================================================
 * PYBIND11 MODULE
 * ============================================================
 */

PYBIND11_MODULE(
    travislabs_pricer,
    module
) {
    module.doc() =
        "TravisLabs C++17 derivatives pricing engine";


    module.def(
        "black_scholes_call",
        &black_scholes_python,

        py::arg("spot"),
        py::arg("strike"),
        py::arg("rate"),
        py::arg("volatility"),
        py::arg("maturity")
    );


    module.def(
        "binomial_call",
        &binomial_python,

        py::arg("spot"),
        py::arg("strike"),
        py::arg("rate"),
        py::arg("volatility"),
        py::arg("maturity"),
        py::arg("steps") = 500
    );


    module.def(
        "monte_carlo_call",
        &monte_carlo_python,

        py::arg("spot"),
        py::arg("strike"),
        py::arg("rate"),
        py::arg("volatility"),
        py::arg("maturity"),
        py::arg("paths") = 100000
    );


    module.def(
        "crank_nicolson_call",
        &crank_nicolson_python,

        py::arg("spot"),
        py::arg("strike"),
        py::arg("rate"),
        py::arg("volatility"),
        py::arg("maturity"),
        py::arg("space_steps") = 400,
        py::arg("time_steps") = 400
    );
}