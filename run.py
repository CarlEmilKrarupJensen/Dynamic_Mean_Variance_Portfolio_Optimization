import importlib
import experiments
import embedding
import numerical_bellman
importlib.reload(experiments)
importlib.reload(embedding)
importlib.reload(numerical_bellman)
from experiments import run_parameter_grid, run_single_experiment
from embedding import plot_example_realization
from numerical_bellman import run_bellman_validation, run_bellman_validation, run_policy_comparison_mc

def main():
    # Monte Carlo experiment
    
    results, example = run_parameter_grid(
        T=1.0,
        N=252,
        M=100,
        x0=1000.0,
    )

    print(results)

    params = example["params"]
    sim = example["simulation"]

    plot_example_realization(
        path_id=0,
        stock_return=sim["stock_return"],
        bond_return=sim["bond_return"],
        wealth=sim["wealth"],
        stock_holding=sim["stock_holding"],
        x0=params["x0"],
        lambda_history=sim["lambda_history"],
        title=(
            f"Example path: r={params['r']}, "
            f"mu={params['mu']}, "
            f"sigma={params['sigma']}, "
            f"gamma={params['gamma']}"
        ),
    )

    # Bellman validation experiment
    row, sim = run_single_experiment(
        r=0.05,
        mu=0.14,
        sigma=0.25,
        gamma=0.75,
        T=1.0,
        N=5,
        M=2000,
        x0=1000.0,
        seed=None,
    )

    V_num, U_num, x_grid = run_bellman_validation(sim, row)

    comparison, analytical_wealth, numerical_wealth = run_policy_comparison_mc(
        row=row,
        sim=sim,
        V_num=V_num,
        U_num=U_num,
        x_grid=x_grid,
        x0=1000.0,
    )

    print(comparison)
    #print(analytical_wealth)
    #print(numerical_wealth)

    run_bellman_validation(sim, row)

if __name__ == "__main__":
    main()

