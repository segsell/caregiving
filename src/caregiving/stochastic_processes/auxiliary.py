import numpy as np


def loglike(params, data, start_data):
    """Log-likelihood calculation.

    params: pd.DataFrame
        DataFrame with the parameters.
    data: pd.DataFrame
        DataFrame with the data.
    data: pd.DataFrame
        DataFrame with the start data.

    """

    event = data["death event"]
    death = event.astype(bool)

    return (
        _log_density_function(data[death], params).sum()
        + _log_survival_function(data[~death], params).sum()
        - _log_survival_function(start_data, params).sum()
    )


def _log_density_function(data, params):
    """Calculate the log-density function.

    Log of the density function:

       (log of d[-S(age)]/d(age) = log of - dS(age)/d(age))

    """
    age_coef = params.loc["age", "value"]
    age_contrib = np.exp(age_coef * data["age"]) - 1

    log_lambda_ = sum(
        [params.loc[x, "value"] * data[x] for x in params.index if x != "age"]
    )
    lambda_ = np.exp(log_lambda_)

    return log_lambda_ + age_coef * data["age"] - ((lambda_ * age_contrib) / age_coef)


def _log_survival_function(data, params):
    """Calculate the log-survival function."""
    age_coef = params.loc["age", "value"]
    age_contrib = np.exp(age_coef * data["age"]) - 1

    lambda_ = np.exp(
        sum([params.loc[x, "value"] * data[x] for x in params.index if x != "age"])
    )

    return -(lambda_ * age_contrib) / age_coef
