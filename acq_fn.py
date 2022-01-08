import numpy as np
from scipy import stats


def probability_improvement(f, y_current, x_proposed):
    """
    Return E(I[f_proposed < f_current])

    Parameters
    ----------

    f : GP predict function
    y_current : float
        Current best evaluation f(x+)
    x_proposed : np.array
        Proposal parameters. Shape: (1, 1)

    Returns
    -------
    probability_improvement : float
    """
    mu, var = f(x_proposed)
    std = var ** 0.5
    delta = mu - y_current

    # x / inf = 0
    std[std == 0] = np.inf
    z = delta / std
    unit_norm = stats.norm()
    return unit_norm.cdf(z)


def expected_improvement(f, y_current, x_proposed):
    """
    Return E(max(f_proposed - f_current), 0)

    Parameters
    ----------

    f : GP predict function
    y_current : float
        Current best evaluation f(x+)
    x_proposed : np.array
        Proposal parameters. Shape: (1, 1)

    Returns
    -------
    expected_improvement : float
        E(max(f_proposed - f_current), 0)
    """
    mu, var = f(x_proposed)
    std = var ** 0.5
    delta = mu - y_current

    # x / inf = 0
    std[std == 0] = np.inf
    z = delta / std
    unit_norm = stats.norm()
    return delta * unit_norm.cdf(z) + std * unit_norm.pdf(z)


def ucb(f, y_current, x_proposed, beta=2.0):
    mu, var = f(x_proposed)
    std = var ** 0.5
    return mu + beta * std

