import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import numpy as np
import matplotlib.pyplot as plt
from experiments import run_single_experiment
import pandas as pd

def terminal_value(x, c, lam):
    """Computes the terminal value of the \mathcal{A} problem"""
    return -c * x**2 + lam * x



def solve_bellman_mc(
    x_grid,
    control_bounds,
    control_points,
    bond_return,
    excess_return_samples,
    c,
    lam,
):
    """Numerical Bellman approximation using Monte Carlo integration"""

    N = len(bond_return)

    V_list = [None] * (N + 1)
    U_list = [None] * N

    # Terminal condition
    V_terminal = terminal_value(x_grid, c, lam)
    V_list[N] = V_terminal

    # Backward recursion
    for t in reversed(range(N)):

        R0 = bond_return[t]
        P_samples = excess_return_samples[:, t]

        V_next = V_list[t + 1]

        interpolator = interp1d(
            x_grid,
            V_next,
            kind="linear",
            fill_value="extrapolate",
        )

        optimal_values = np.zeros_like(x_grid)
        optimal_controls = np.zeros_like(x_grid)

        for i, x in enumerate(x_grid):

            def objective(u):
                next_wealth = R0 * x + P_samples * u

                continuation_values = interpolator(next_wealth)

                return -np.mean(continuation_values)

            result = minimize_scalar(
                objective,
                bounds=control_bounds,
                method="bounded",
            )

            optimal_controls[i] = result.x
            optimal_values[i] = -result.fun

        V_list[t] = optimal_values
        U_list[t] = optimal_controls

    return V_list, U_list



def analytical_value_function(A, B, C, x_grid):
    """Computes the analytical value function from the conjecture solution"""
    return A * x_grid**2 + B * x_grid + C


def analytical_control(A_next, B_next, mu, Sigma, R0, x_grid):
    """Computes the analytical control from the conjecture solution"""
    k = np.linalg.pinv(Sigma) @ mu

    controls = []

    for x in x_grid:
        scalar = R0 * x + B_next / (2 * A_next)
        u = -scalar * k
        controls.append(float(u))

    return np.array(controls)


def run_bellman_validation(sim,row):
    """Runs the comarison between analytical and numerical Bellman solution for a single experiment"""
    print(row)

    x_grid = np.linspace(-500, 3000, 250)
    control_bounds = (-5000, 5000)
    control_points = 200

    N_mc = 1000
    N = len(sim["bond_return"])

    rng = np.random.default_rng(123)

    excess_return_samples = np.zeros((N_mc, N))

    for t in range(N):
        mu_t = sim["mu_t"][t, 0]
        sigma_t = np.sqrt(sim["Sigma_t"][t, 0, 0])
        excess_return_samples[:, t] = rng.normal(mu_t, sigma_t, size=N_mc)

    V_num, U_num = solve_bellman_mc(
        x_grid=x_grid,
        control_bounds=control_bounds,
        control_points=control_points,
        bond_return=sim["bond_return"],
        excess_return_samples=excess_return_samples,
        c=row["c"],
        lam=row["lambda_star"],
    )

    A = sim["A"] # Takes A,B and C from an experiment simulation
    B = sim["B"]
    C = sim["C"]
    ## we plot the numerical and analytical value functions and controls at t=0
    V_analytical = analytical_value_function(
        A[0],
        B[0],
        C[0],
        x_grid,
    )

    U_analytical = analytical_control(
        A[1],
        B[1],
        sim["mu_t"][0],
        sim["Sigma_t"][0],
        sim["bond_return"][0],
        x_grid,
    )

    plt.figure(figsize=(10, 5))
    plt.plot(x_grid, V_num[0], label="Numerical Bellman")
    plt.plot(x_grid, V_analytical, "--", label="Analytical quadratic")
    plt.title("Value function comparison at t=0")
    plt.xlabel("Wealth")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(x_grid, U_num[0], label="Numerical control")
    plt.plot(x_grid, U_analytical, "--", label="Analytical control")
    plt.title("Optimal control comparison at t=0")
    plt.xlabel("Wealth")
    plt.ylabel("Optimal stock holding")
    plt.legend()
    plt.show()

    return V_num, U_num, x_grid

def simulate_numerical_bellman_policy(
    U_num,
    x_grid,
    x0,
    bond_return,
    excess_return_paths,
):
    
    """Simulate the wealth path using the numerical Bellman policy"""
    M, N, d = excess_return_paths.shape

    if d != 1:
        raise NotImplementedError("This version assumes one risky asset.")

    wealth = np.full((M, N + 1), np.nan)
    stock_holding = np.full((M, N, d), np.nan)

    wealth[:, 0] = x0

    for t in range(N):
        policy_interp = interp1d(
            x_grid,
            U_num[t],
            kind="linear",
            fill_value="extrapolate",
        )

        u_t = policy_interp(wealth[:, t])

        stock_holding[:, t, 0] = u_t

        wealth[:, t + 1] = (
            bond_return[t] * wealth[:, t]
            + excess_return_paths[:, t, 0] * u_t
        )

    return wealth, stock_holding

def compare_analytical_vs_numerical_policy(
    analytical_wealth,
    numerical_wealth,
    analytical_holding,
    numerical_holding,
    x_grid,
):
    """Computes diagnostics for comparison between analytical and numerical policies """

    terminal_analytical = analytical_wealth[:, -1]
    terminal_numerical = numerical_wealth[:, -1]

    outside_grid = (
        (numerical_wealth < x_grid[0])
        | (numerical_wealth > x_grid[-1])
    )

    comparison = {
        "mean_terminal_analytical": np.mean(terminal_analytical),
        "mean_terminal_numerical": np.mean(terminal_numerical),
        "std_terminal_analytical": np.std(terminal_analytical),
        "std_terminal_numerical": np.std(terminal_numerical),
        "mean_terminal_difference": np.mean(
            terminal_numerical - terminal_analytical
        ),
        "mean_abs_terminal_difference": np.mean(
            np.abs(terminal_numerical - terminal_analytical)
        ),
        "rmse_terminal_difference": np.sqrt(
            np.mean((terminal_numerical - terminal_analytical) ** 2)
        ),
        "terminal_correlation": np.corrcoef(
            terminal_analytical,
            terminal_numerical,
        )[0, 1],
        "mean_abs_holding_difference": np.mean(
            np.abs(numerical_holding - analytical_holding)
        ),
        "fraction_numerical_wealth_outside_grid": np.mean(outside_grid),
    }

    return pd.Series(comparison)

def run_policy_comparison_mc(
    row,
    sim,
    V_num,
    U_num,
    x_grid,
    x0,
):
    """Runs the comparison between the analytical and numerical strategy"""
    from embedding import simulate_embedding_paths

    analytical_wealth, analytical_holding, _, _, _ = simulate_embedding_paths(
        lam=row["lambda_star"],
        c=row["c"],
        x0=x0,
        bond_return=sim["bond_return"],
        excess_return_paths=sim["excess_return_paths"],
        mu_t=sim["mu_t"],
        Sigma_t=sim["Sigma_t"],
    )

    numerical_wealth, numerical_holding = simulate_numerical_bellman_policy(
        U_num=U_num,
        x_grid=x_grid,
        x0=x0,
        bond_return=sim["bond_return"],
        excess_return_paths=sim["excess_return_paths"],
    )

    comparison = compare_analytical_vs_numerical_policy(
        analytical_wealth=analytical_wealth,
        numerical_wealth=numerical_wealth,
        analytical_holding=analytical_holding,
        numerical_holding=numerical_holding,
        x_grid=x_grid,
    )

    return comparison, analytical_wealth, numerical_wealth