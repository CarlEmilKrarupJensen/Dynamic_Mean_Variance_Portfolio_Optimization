
This Repository contains code for implementing the analytical solutions to solving the embedded problem
and the game theoretical approach described in the article "Dynamic Mean Variance Portfolio Optmimization".
- "embedding.py" contains the implementation of the market simulation, and the approach dscribed in section 3.1 of
the article.
- "experiments.py" contains the implementation of the Monte Carlo simulation experiments
- "numerical_bellman.py" contains the numerical approximation of the Bellman function and a code which compars this
    with the analytical solution
- "run.py" runs the Monte Carlo simulation reported in the article and the numerical vs analytical comparison


Required packages: numpy, pandas, matplotlib, scipy