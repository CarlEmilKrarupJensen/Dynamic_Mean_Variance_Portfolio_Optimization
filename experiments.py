import pandas as pd
from itertools import product
import os
from embedding import (
    exact_excess_moments,
    simulate_market_paths,
    estimate_lambda_star_mc,
    simulate_embedding_paths,
    summarize_strategy,
)

def run_single_experiment(r, mu, sigma, gamma, T=1.0, N=252, M=1000, x0=1000.0, seed=None):
    """Runs the embedding strategy using the functions in ebbedding.py. Reports the diagnostics and the simulation output."""
    c = gamma / x0
    dt = T / N

    stock_return, bond_return, excess_return_paths = simulate_market_paths(
        mu=mu,
        r=r,
        sigma=sigma,
        T=T,
        N=N,
        M=M,
        seed=seed,
    )

    mu_t, Sigma_t = exact_excess_moments(
        mu=mu,
        r=r,
        sigma=sigma,
        dt=dt,
        N=N,
    )

    lambda_star, ExT, iterations, lambda_history = estimate_lambda_star_mc(
        c=c,
        x0=x0,
        bond_return=bond_return,
        excess_return_paths=excess_return_paths,
        mu_t=mu_t,
        Sigma_t=Sigma_t,
    )

    wealth, stock_holding, A, B, C = simulate_embedding_paths(
        lam=lambda_star,
        c=c,
        x0=x0,
        bond_return=bond_return,
        excess_return_paths=excess_return_paths,
        mu_t=mu_t,
        Sigma_t=Sigma_t,
    )

    diagnostics = summarize_strategy(
        wealth=wealth,
        stock_holding=stock_holding,
        stock_return=stock_return,
        bond_return=bond_return,
        x0=x0,
    )

    result_row = {
        "r": r,
        "mu": mu,
        "sigma": sigma,
        "gamma": gamma,
        "c": c,
        "lambda_star": lambda_star,
        "estimated_ExT": ExT,
        "lambda_iterations": iterations,
    }

    result_row.update(diagnostics)

    simulation_output = {
        "stock_return": stock_return,
        "bond_return": bond_return,
        "excess_return_paths": excess_return_paths,
        "mu_t": mu_t,
        "Sigma_t": Sigma_t,
        "wealth": wealth,
        "stock_holding": stock_holding,
        "A": A,
        "B": B,
        "C": C,
        "lambda_history": lambda_history,
    }

    return result_row, simulation_output


def run_parameter_grid(T=1.0, N=252, M=1000, x0=1000.0):
    """Runs the Monte Carlo simualtion for each parameter configuration specified below,
      and saves the results in a CSV file"""
    r_grid = [0.01, 0.05]
    mu_grid = [0.06, 0.14]
    sigma_grid = [0.15, 0.45]
    gamma_grid = [0.25, 0.75]

    results = []
    saved_example = None

    config_id = 0

    for r, mu, sigma, gamma in product(r_grid, mu_grid, sigma_grid, gamma_grid):
        config_id += 1

        row, simulation_output = run_single_experiment(
            r=r,
            mu=mu,
            sigma=sigma,
            gamma=gamma,
            T=T,
            N=N,
            M=M,
            x0=x0,
            seed=None,
        )

        results.append(row)

        # Saved example for plotting a realization
        example_params = {
            "r": 0.05,
            "mu": 0.14,
            "sigma": 0.15,
            "gamma": 0.25,
        }

        if (
            r == example_params["r"]
            and mu == example_params["mu"]
            and sigma == example_params["sigma"]
            and gamma == example_params["gamma"]
        ):
            saved_example = {
                "params": {
                    "r": r,
                    "mu": mu,
                    "sigma": sigma,
                    "gamma": gamma,
                    "T": T,
                    "N": N,
                    "M": M,
                    "x0": x0,
                },
                "simulation": simulation_output,
            }
    results = pd.DataFrame(results)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    results.to_csv(os.path.join(BASE_DIR, "mc_experiment_results_embedding.csv"),index=False,)
    return results, saved_example