#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace travislabs {

    constexpr double PI = 3.14159265358979323846;


    /* ============================================================
     * NORMAL DISTRIBUTION
     * ============================================================
     */

    double normal_pdf(double x) {
        return std::exp(-0.5 * x * x)
            / std::sqrt(2.0 * PI);
    }


    double normal_cdf(double x) {
        return 0.5
            * std::erfc(
                -x / std::sqrt(2.0)
            );
    }


    /* ============================================================
     * RESULT STRUCTURES
     * ============================================================
     */

    struct OptionResult {
        double price;
        double delta;
        double gamma;
        double vega;
        double theta;
    };


    struct MonteCarloResult {
        double price;
        double standard_error;
        double ci_lower;
        double ci_upper;
    };


    struct PDEResult {
        double price;
        int space_steps;
        int time_steps;
    };


    /* ============================================================
     * BLACK-SCHOLES
     * ============================================================
     */

    OptionResult black_scholes_call(
        double spot,
        double strike,
        double rate,
        double volatility,
        double maturity
    ) {
        const double sqrtT =
            std::sqrt(maturity);

        const double d1 =
            (
                std::log(spot / strike)
                +
                (
                    rate
                    +
                    0.5
                    * volatility
                    * volatility
                    )
                * maturity
                )
            /
            (
                volatility
                * sqrtT
                );

        const double d2 =
            d1
            -
            volatility
            * sqrtT;

        const double discount =
            std::exp(
                -rate
                * maturity
            );

        const double price =
            spot
            * normal_cdf(d1)
            -
            strike
            * discount
            * normal_cdf(d2);

        const double delta =
            normal_cdf(d1);

        const double gamma =
            normal_pdf(d1)
            /
            (
                spot
                * volatility
                * sqrtT
                );

        const double vega =
            spot
            * normal_pdf(d1)
            * sqrtT;

        const double theta =
            -
            (
                spot
                * normal_pdf(d1)
                * volatility
                )
            /
            (
                2.0
                * sqrtT
                )
            -
            rate
            * strike
            * discount
            * normal_cdf(d2);

        return {
            price,
            delta,
            gamma,
            vega,
            theta
        };
    }


    /* ============================================================
     * CRR BINOMIAL TREE
     * ============================================================
     */

    double binomial_call(
        double spot,
        double strike,
        double rate,
        double volatility,
        double maturity,
        int steps
    ) {
        const double dt =
            maturity / steps;

        const double up =
            std::exp(
                volatility
                * std::sqrt(dt)
            );

        const double down =
            1.0 / up;

        const double probability =
            (
                std::exp(rate * dt)
                - down
                )
            /
            (
                up - down
                );

        const double discount =
            std::exp(
                -rate * dt
            );

        std::vector<double>
            values(
                steps + 1
            );


        /*
         * Terminal payoff
         */

        for (
            int i = 0;
            i <= steps;
            ++i
            ) {
            const double terminalSpot =
                spot
                * std::pow(
                    up,
                    steps - i
                )
                * std::pow(
                    down,
                    i
                );

            values[i] =
                std::max(
                    terminalSpot - strike,
                    0.0
                );
        }


        /*
         * Backward induction
         */

        for (
            int step = steps - 1;
            step >= 0;
            --step
            ) {
            for (
                int i = 0;
                i <= step;
                ++i
                ) {
                values[i] =
                    discount
                    *
                    (
                        probability
                        * values[i]
                        +
                        (
                            1.0
                            - probability
                            )
                        * values[i + 1]
                        );
            }
        }

        return values[0];
    }


    /* ============================================================
     * MONTE CARLO
     * ============================================================
     */

    MonteCarloResult monte_carlo_call(
        double spot,
        double strike,
        double rate,
        double volatility,
        double maturity,
        int paths
    ) {
        /*
         * Fixed random seed so benchmarking is reproducible.
         */

        std::mt19937_64 generator(42);

        std::normal_distribution<double>
            standardNormal(
                0.0,
                1.0
            );


        const double drift =
            (
                rate
                -
                0.5
                * volatility
                * volatility
                )
            * maturity;


        const double diffusion =
            volatility
            * std::sqrt(maturity);


        double payoffSum =
            0.0;

        double payoffSquaredSum =
            0.0;


        for (
            int i = 0;
            i < paths;
            ++i
            ) {
            const double z =
                standardNormal(
                    generator
                );


            const double terminalSpot =
                spot
                * std::exp(
                    drift
                    +
                    diffusion
                    * z
                );


            const double payoff =
                std::max(
                    terminalSpot - strike,
                    0.0
                );


            payoffSum +=
                payoff;

            payoffSquaredSum +=
                payoff
                * payoff;
        }


        const double meanPayoff =
            payoffSum
            / paths;


        const double variance =
            (
                payoffSquaredSum
                -
                paths
                * meanPayoff
                * meanPayoff
                )
            /
            (
                paths - 1
                );


        const double discount =
            std::exp(
                -rate
                * maturity
            );


        const double price =
            discount
            * meanPayoff;


        const double standardError =
            discount
            * std::sqrt(
                variance
                / paths
            );


        const double ciLower =
            price
            -
            1.96
            * standardError;


        const double ciUpper =
            price
            +
            1.96
            * standardError;


        return {
            price,
            standardError,
            ciLower,
            ciUpper
        };
    }


    /* ============================================================
     * THOMAS ALGORITHM
     * ============================================================
     *
     * Solves a tridiagonal system:
     *
     * a_i x_(i-1)
     * +
     * b_i x_i
     * +
     * c_i x_(i+1)
     * =
     * d_i
     *
     * Complexity: O(N)
     * ============================================================
     */

    std::vector<double>
        thomas_solve(
            const std::vector<double>& lower,
            const std::vector<double>& diagonal,
            const std::vector<double>& upper,
            const std::vector<double>& rhs
        ) {
        const int n =
            static_cast<int>(
                diagonal.size()
                );


        std::vector<double>
            cPrime(
                n,
                0.0
            );

        std::vector<double>
            dPrime(
                n,
                0.0
            );

        std::vector<double>
            solution(
                n,
                0.0
            );


        cPrime[0] =
            upper[0]
            /
            diagonal[0];


        dPrime[0] =
            rhs[0]
            /
            diagonal[0];


        /*
         * Forward elimination
         */

        for (
            int i = 1;
            i < n;
            ++i
            ) {
            const double denominator =
                diagonal[i]
                -
                lower[i]
                * cPrime[i - 1];


            if (
                i < n - 1
                ) {
                cPrime[i] =
                    upper[i]
                    /
                    denominator;
            }


            dPrime[i] =
                (
                    rhs[i]
                    -
                    lower[i]
                    * dPrime[i - 1]
                    )
                /
                denominator;
        }


        /*
         * Back substitution
         */

        solution[n - 1] =
            dPrime[n - 1];


        for (
            int i = n - 2;
            i >= 0;
            --i
            ) {
            solution[i] =
                dPrime[i]
                -
                cPrime[i]
                * solution[i + 1];
        }


        return solution;
    }


    /* ============================================================
     * CRANK-NICOLSON FINITE-DIFFERENCE PDE
     * ============================================================
     */

    PDEResult crank_nicolson_call(
        double spot,
        double strike,
        double rate,
        double volatility,
        double maturity,
        int spaceSteps,
        int timeSteps
    ) {
        /*
         * Upper stock-price boundary.
         */

        const double sMax =
            std::max(
                4.0 * spot,
                4.0 * strike
            );


        const double dS =
            sMax
            / spaceSteps;


        const double dt =
            maturity
            / timeSteps;


        /*
         * Option values across stock-price grid.
         */

        std::vector<double>
            values(
                spaceSteps + 1,
                0.0
            );


        /*
         * Terminal European call payoff:
         *
         * V(S,T) = max(S-K, 0)
         */

        for (
            int i = 0;
            i <= spaceSteps;
            ++i
            ) {
            const double stock =
                i * dS;

            values[i] =
                std::max(
                    stock - strike,
                    0.0
                );
        }


        const int interior =
            spaceSteps - 1;


        std::vector<double>
            lower(
                interior,
                0.0
            );

        std::vector<double>
            diagonal(
                interior,
                0.0
            );

        std::vector<double>
            upper(
                interior,
                0.0
            );

        std::vector<double>
            rhs(
                interior,
                0.0
            );


        /*
         * March from terminal payoff back to today's value.
         */

        for (
            int timeIndex = 0;
            timeIndex < timeSteps;
            ++timeIndex
            ) {
            const double oldTau =
                timeIndex
                * dt;


            const double newTau =
                (
                    timeIndex + 1
                    )
                * dt;


            /*
             * Boundary conditions
             */

            const double lowerOld =
                0.0;

            const double lowerNew =
                0.0;


            const double upperOld =
                sMax
                -
                strike
                * std::exp(
                    -rate
                    * oldTau
                );


            const double upperNew =
                sMax
                -
                strike
                * std::exp(
                    -rate
                    * newTau
                );


            values[0] =
                lowerOld;

            values[spaceSteps] =
                upperOld;


            /*
             * Build Crank-Nicolson tridiagonal system.
             */

            for (
                int i = 1;
                i < spaceSteps;
                ++i
                ) {
                const double index =
                    static_cast<double>(
                        i
                        );


                const double A =
                    0.5
                    * volatility
                    * volatility
                    * index
                    * index
                    -
                    0.5
                    * rate
                    * index;


                const double B =
                    -
                    volatility
                    * volatility
                    * index
                    * index
                    -
                    rate;


                const double C =
                    0.5
                    * volatility
                    * volatility
                    * index
                    * index
                    +
                    0.5
                    * rate
                    * index;


                const int j =
                    i - 1;


                /*
                 * Implicit side
                 */

                lower[j] =
                    -0.5
                    * dt
                    * A;


                diagonal[j] =
                    1.0
                    -
                    0.5
                    * dt
                    * B;


                upper[j] =
                    -0.5
                    * dt
                    * C;


                /*
                 * Explicit side
                 */

                rhs[j] =
                    (
                        0.5
                        * dt
                        * A
                        * values[i - 1]
                        )
                    +
                    (
                        1.0
                        +
                        0.5
                        * dt
                        * B
                        )
                    * values[i]
                    +
                    (
                        0.5
                        * dt
                        * C
                        * values[i + 1]
                        );
            }


            /*
             * Lower boundary contribution.
             */

            {
                const double i =
                    1.0;

                const double A =
                    0.5
                    * volatility
                    * volatility
                    * i
                    * i
                    -
                    0.5
                    * rate
                    * i;


                rhs[0] +=
                    0.5
                    * dt
                    * A
                    * lowerNew;
            }


            /*
             * Upper boundary contribution.
             */

            {
                const double i =
                    static_cast<double>(
                        spaceSteps - 1
                        );


                const double C =
                    0.5
                    * volatility
                    * volatility
                    * i
                    * i
                    +
                    0.5
                    * rate
                    * i;


                rhs[interior - 1] +=
                    0.5
                    * dt
                    * C
                    * upperNew;
            }


            /*
             * Boundary terms are known, not unknown.
             */

            lower[0] =
                0.0;

            upper[interior - 1] =
                0.0;


            /*
             * Thomas algorithm solves the tridiagonal system.
             */

            const std::vector<double>
                solution =
                thomas_solve(
                    lower,
                    diagonal,
                    upper,
                    rhs
                );


            values[0] =
                lowerNew;


            for (
                int i = 1;
                i < spaceSteps;
                ++i
                ) {
                values[i] =
                    solution[
                        i - 1
                    ];
            }


            values[spaceSteps] =
                upperNew;
        }


        /*
         * Interpolate because spot may fall between grid nodes.
         */

        const double rawIndex =
            spot
            / dS;


        int lowerIndex =
            static_cast<int>(
                std::floor(
                    rawIndex
                )
                );


        lowerIndex =
            std::max(
                0,
                std::min(
                    lowerIndex,
                    spaceSteps - 1
                )
            );


        const int upperIndex =
            lowerIndex + 1;


        const double lowerSpot =
            lowerIndex
            * dS;


        const double weight =
            (
                spot
                - lowerSpot
                )
            /
            dS;


        const double price =
            values[lowerIndex]
            * (
                1.0 - weight
                )
            +
            values[upperIndex]
            * weight;


        return {
            price,
            spaceSteps,
            timeSteps
        };
    }


    /* ============================================================
     * RUNTIME BENCHMARK HELPER
     * ============================================================
     */

    template <typename Function>
    double benchmark_ms(
        Function function,
        int repeats = 1
    ) {
        const auto start =
            std::chrono::
            high_resolution_clock::
            now();


        for (
            int i = 0;
            i < repeats;
            ++i
            ) {
            function();
        }


        const auto end =
            std::chrono::
            high_resolution_clock::
            now();


        const std::chrono::duration<
            double,
            std::milli
        > elapsed =
            end - start;


        return (
            elapsed.count()
            / repeats
            );
    }

} // namespace travislabs



/* ============================================================
 * STANDALONE EXECUTABLE
 *
 * When compiling normally:
 *
 * cl /EHsc /O2 /std:c++17 option_pricer.cpp
 *
 * this main() function is included.
 *
 *
 * When bindings.cpp defines:
 *
 * TRAVISLABS_PYBIND
 *
 * before including this file, main() is excluded.
 * ============================================================
 */

#ifndef TRAVISLABS_PYBIND

int main() {

    const double spot =
        100.0;

    const double strike =
        100.0;

    const double rate =
        0.05;

    const double volatility =
        0.20;

    const double maturity =
        1.0;


    std::cout
        << std::fixed
        << std::setprecision(6);


    /* ========================================================
     * BLACK-SCHOLES BENCHMARK
     * ========================================================
     */

    const auto bs =
        travislabs::
        black_scholes_call(
            spot,
            strike,
            rate,
            volatility,
            maturity
        );


    std::cout
        << "=============================================\n";

    std::cout
        << "TravisLabs C++ Numerical Convergence Engine\n";

    std::cout
        << "=============================================\n\n";


    std::cout
        << "Black-Scholes benchmark\n";

    std::cout
        << "-----------------------\n";

    std::cout
        << "Price: "
        << bs.price
        << "\n";

    std::cout
        << "Delta: "
        << bs.delta
        << "\n";

    std::cout
        << "Gamma: "
        << bs.gamma
        << "\n";

    std::cout
        << "Vega: "
        << bs.vega
        << "\n";

    std::cout
        << "Theta: "
        << bs.theta
        << "\n\n";


    /* ========================================================
     * BINOMIAL CONVERGENCE
     * ========================================================
     */

    const std::vector<int>
        binomialSteps = {
            25,
            50,
            100,
            250,
            500,
            1000,
            2000
    };


    std::cout
        << "BINOMIAL CONVERGENCE\n";

    std::cout
        << "Steps,Price,AbsError,Runtime_ms\n";


    for (
        int steps :
    binomialSteps
        ) {
        const double price =
            travislabs::
            binomial_call(
                spot,
                strike,
                rate,
                volatility,
                maturity,
                steps
            );


        const double runtime =
            travislabs::
            benchmark_ms(
                [&]() {
                    volatile double p =
                        travislabs::
                        binomial_call(
                            spot,
                            strike,
                            rate,
                            volatility,
                            maturity,
                            steps
                        );

                    (void)p;
                },
                3
            );


        const double error =
            std::abs(
                price
                -
                bs.price
            );


        std::cout
            << steps
            << ","
            << price
            << ","
            << error
            << ","
            << runtime
            << "\n";
    }


    std::cout
        << "\n";


    /* ========================================================
     * MONTE CARLO CONVERGENCE
     * ========================================================
     */

    const std::vector<int>
        monteCarloPaths = {
            1000,
            5000,
            10000,
            50000,
            100000,
            500000
    };


    std::cout
        << "MONTE CARLO CONVERGENCE\n";

    std::cout
        << "Paths,Price,AbsError,StdError,CI_Lower,CI_Upper,Runtime_ms\n";


    for (
        int paths :
    monteCarloPaths
        ) {
        const auto result =
            travislabs::
            monte_carlo_call(
                spot,
                strike,
                rate,
                volatility,
                maturity,
                paths
            );


        const double runtime =
            travislabs::
            benchmark_ms(
                [&]() {
                    const auto mc =
                        travislabs::
                        monte_carlo_call(
                            spot,
                            strike,
                            rate,
                            volatility,
                            maturity,
                            paths
                        );

                    volatile double p =
                        mc.price;

                    (void)p;
                }
            );


        const double error =
            std::abs(
                result.price
                -
                bs.price
            );


        std::cout
            << paths
            << ","
            << result.price
            << ","
            << error
            << ","
            << result.standard_error
            << ","
            << result.ci_lower
            << ","
            << result.ci_upper
            << ","
            << runtime
            << "\n";
    }


    std::cout
        << "\n";


    /* ========================================================
     * CRANK-NICOLSON CONVERGENCE
     * ========================================================
     */

    const std::vector<int>
        pdeGrids = {
            25,
            50,
            100,
            200,
            400,
            800
    };


    std::cout
        << "CRANK-NICOLSON CONVERGENCE\n";

    std::cout
        << "Grid,Price,AbsError,Runtime_ms\n";


    for (
        int grid :
    pdeGrids
        ) {
        const auto result =
            travislabs::
            crank_nicolson_call(
                spot,
                strike,
                rate,
                volatility,
                maturity,
                grid,
                grid
            );


        const double runtime =
            travislabs::
            benchmark_ms(
                [&]() {
                    const auto pde =
                        travislabs::
                        crank_nicolson_call(
                            spot,
                            strike,
                            rate,
                            volatility,
                            maturity,
                            grid,
                            grid
                        );

                    volatile double p =
                        pde.price;

                    (void)p;
                },
                2
            );


        const double error =
            std::abs(
                result.price
                -
                bs.price
            );


        std::cout
            << grid
            << "x"
            << grid
            << ","
            << result.price
            << ","
            << error
            << ","
            << runtime
            << "\n";
    }


    std::cout
        << "\n";


    /* ========================================================
     * STANDARD SETTINGS
     * ========================================================
     */

    const double binomial500 =
        travislabs::
        binomial_call(
            spot,
            strike,
            rate,
            volatility,
            maturity,
            500
        );


    const auto mc100k =
        travislabs::
        monte_carlo_call(
            spot,
            strike,
            rate,
            volatility,
            maturity,
            100000
        );


    const auto pde400 =
        travislabs::
        crank_nicolson_call(
            spot,
            strike,
            rate,
            volatility,
            maturity,
            400,
            400
        );


    std::cout
        << "METHOD COMPARISON\n";

    std::cout
        << "-----------------\n";


    std::cout
        << "Black-Scholes:   "
        << bs.price
        << "\n";


    std::cout
        << "Binomial (500):  "
        << binomial500
        << "\n";


    std::cout
        << "Monte Carlo:     "
        << mc100k.price
        << "\n";


    std::cout
        << "Crank-Nicolson:  "
        << pde400.price
        << "\n";


    return 0;
}

#endif