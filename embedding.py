import numpy as np
import matplotlib.pyplot as plt

def exact_excess_moments(mu, r, sigma, dt, N):
    """" Computes E(P_t) and E(P_tP_t'),
        using the assumption thatnthe investor knows
          the underlying market dynamics"""
    mu_P = np.array([np.exp(mu * dt) - np.exp(r * dt)])

    Sigma_P = np.array([[
        np.exp((2 * mu + sigma**2) * dt)
        - 2 * np.exp((mu + r) * dt)
        + np.exp(2 * r * dt)
    ]])

    mu_t = np.tile(mu_P, (N, 1))
    Sigma_t = np.tile(Sigma_P, (N, 1, 1))

    return mu_t, Sigma_t

def simulate_market_paths(mu, r, sigma, T, N, M, S0=1.0, seed=None):
    """Simulates the stock and bond paths and computes the excess returns
        Market is simulated M times since this is the number of Monte Carlo repetitions"""
    rng = np.random.default_rng(seed)

    dt = T / N
    t = np.linspace(0, T, N + 1)

    dW = rng.normal(0, np.sqrt(dt), size=(M, N))
    W = np.cumsum(dW, axis=1)
    W = np.column_stack([np.zeros(M), W])

    S = S0 * np.exp((mu - 0.5 * sigma**2) * t[None, :] + sigma * W)

    stock_return = S[:, 1:] / S[:, :-1]
    bond_return = np.full(N, np.exp(r * dt))

    excess_return_paths = (stock_return - bond_return[None, :])[:, :, None]

    return stock_return, bond_return, excess_return_paths


def compute_ABC(lam, c, bond_return, mu_t, Sigma_t):
    """Computes A, B and C from the conjecture solution"""
    N = len(bond_return)

    A = np.full(N + 1, np.nan)
    B = np.full(N + 1, np.nan)
    C = np.full(N + 1, np.nan)

    A[N] = -c
    B[N] = lam
    C[N] = 0.0

    for i in range(N - 1, -1, -1):
        mu_i = mu_t[i]
        Sigma_i = Sigma_t[i]
        R0_i = bond_return[i]

        theta_i = float(mu_i.T @ np.linalg.pinv(Sigma_i) @ mu_i)

        A[i] = A[i + 1] * R0_i**2 * (1 - theta_i)
        B[i] = B[i + 1] * R0_i * (1 - theta_i)
        C[i] = C[i + 1] - (B[i + 1]**2 / (4 * A[i + 1])) * theta_i

    return A, B, C


def simulate_embedding_paths(lam, c, x0, bond_return, excess_return_paths, mu_t, Sigma_t):
    """Computes u andand the resulting wealth path"""
    M, N, d = excess_return_paths.shape

    A, B, C = compute_ABC(lam, c, bond_return, mu_t, Sigma_t)

    wealth = np.full((M, N + 1), np.nan)
    stock_holding = np.full((M, N, d), np.nan)

    wealth[:, 0] = x0

    for i in range(N):
        mu_i = mu_t[i]
        Sigma_i = Sigma_t[i]
        R0_i = bond_return[i]

        k_i = np.linalg.pinv(Sigma_i) @ mu_i

        scalar_i = R0_i * wealth[:, i] + B[i + 1] / (2 * A[i + 1])

        stock_holding[:, i, :] = -scalar_i[:, None] * k_i[None, :] # u_{t_\ell}

        wealth[:, i + 1] = (
            R0_i * wealth[:, i]
            + np.sum(excess_return_paths[:, i, :] * stock_holding[:, i, :], axis=1)
        )

    return wealth, stock_holding, A, B, C


def estimate_lambda_star_mc(
    c,
    x0,
    bond_return,
    excess_return_paths,
    mu_t,
    Sigma_t,
    alpha=0.2,
    tol=1e-8,
    max_iter=200,
):
    """Fixed point iteration to estimate lambda_star"""
    lam = 1 + 2 * c * x0
    lambda_history = [lam]

    for k in range(max_iter):
        wealth, stock_holding, A, B, C = simulate_embedding_paths(
            lam, c, x0, bond_return, excess_return_paths, mu_t, Sigma_t
        )

        expected_terminal_wealth = np.mean(wealth[:, -1])
        lam_new = 1 + 2 * c * expected_terminal_wealth
        lam_damped = (1 - alpha) * lam + alpha * lam_new

        lambda_history.append(lam_damped)

        if abs(lam_damped - lam) < tol:
            lam = lam_damped
            break

        lam = lam_damped

    return lam, expected_terminal_wealth, k + 1, np.array(lambda_history)

def max_drawdown(paths):
    """Computes the max drawdown diagnostic of the strtegy"""
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdown = paths / running_max - 1
    return np.min(drawdown, axis=1)

def summarize_strategy(wealth, stock_holding, stock_return, bond_return, x0):
    """Computes the remaining diagnostics of the strategy"""
    M, N_plus_1 = wealth.shape
    N = N_plus_1 - 1

    wealth_rf = x0 * np.cumprod(
        np.column_stack([np.ones(M), np.tile(bond_return, (M, 1))]),
        axis=1,
    )

    wealth_stock = x0 * np.cumprod(
        np.column_stack([np.ones(M), stock_return]),
        axis=1,
    )

    terminal = wealth[:, -1]
    terminal_rf = wealth_rf[:, -1]
    terminal_stock = wealth_stock[:, -1]

    stock_position = stock_holding[:, :, 0]
    bond_position = wealth[:, :-1] - stock_position
    stock_weight = stock_position / wealth[:, :-1]

    diagnostics = {
        "mean_terminal": np.mean(terminal),
        "std_terminal": np.std(terminal),
        "prob_beat_rf": np.mean(terminal > terminal_rf),
        "prob_beat_stock": np.mean(terminal > terminal_stock),
        "prob_negative_wealth": np.mean(np.min(wealth, axis=1) < 0),
        "mean_max_drawdown": np.mean(max_drawdown(wealth)),
        "avg_max_abs_weight": np.mean(np.max(np.abs(stock_weight), axis=1)),
        "prob_borrowing": np.mean(np.any(bond_position < 0, axis=1)),
    }

    return diagnostics

def plot_example_realization(
    path_id,
    stock_return,
    bond_return,
    wealth,
    stock_holding,
    x0,
    lambda_history,
    title=None,
):
    """Plots an example of one realization of the strategy,
    including the wealth path, the lambda convergence and the stock/bond holdings"""
    N = len(bond_return)
    time = np.arange(N + 1)

    stock_path = x0 * np.cumprod(np.r_[1.0, stock_return[path_id]])
    bond_path = x0 * np.cumprod(np.r_[1.0, bond_return])
    strategy_path = wealth[path_id]

    stock_position = stock_holding[path_id, :, 0]
    bond_position = wealth[path_id, :-1] - stock_position

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(time, stock_path, label="Stock")
    axes[0].plot(time, bond_path, label="Bond")
    axes[0].plot(time, strategy_path, label="Li-Ng strategy")
    axes[0].set_title("Wealth paths")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Wealth")
    axes[0].legend()

    axes[1].plot(lambda_history, marker="o")
    axes[1].set_title("Lambda convergence")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Lambda")

    axes[2].plot(time[:-1], stock_position, label="Stock holding")
    axes[2].plot(time[:-1], bond_position, label="Bond holding")
    axes[2].set_title("Dollar holdings")
    axes[2].set_xlabel("Time step")
    axes[2].set_ylabel("Amount invested")
    axes[2].legend()

    if title is not None:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()