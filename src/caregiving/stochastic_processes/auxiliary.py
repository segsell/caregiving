import numpy as np
import pandas as pd

# def loglike(params, data, start_data):
#     """Log-likelihood calculation.

#     params: pd.DataFrame
#         DataFrame with the parameters.
#     data: pd.DataFrame
#         DataFrame with the data.
#     data: pd.DataFrame
#         DataFrame with the start data.

#     """

#     event = data["death event"]
#     death = event.astype(bool)

#     return (
#         _log_density_function(data[death], params).sum()
#         + _log_survival_function(data[~death], params).sum()
#         - _log_survival_function(start_data, params).sum()
#     )


# def _log_density_function(data, params):
#     """
#     Calculate the log-density function in a numerically stable way.

#     Uses np.expm1 for the exponential minus one calculation and clips
#     the linear predictor before exponentiating.

#     Returns:
#         A pandas Series with the log-density for each observation.
#     """
#     age_coef = params.loc["age", "value"]
#     # Use np.expm1 for stable computation of exp(age_coef * age) - 1.
#     safe_age = np.clip(age_coef * data["age"], -700, 700)
#     age_contrib = np.expm1(safe_age)

#     # Compute the linear predictor for lambda. (Exclude the 'age' parameter.)
#     lin_pred = sum(
#         [params.loc[x, "value"] * data[x] for x in params.index if x != "age"]
#     )
#     # Clip the linear predictor to avoid overflow in exp.
#     lin_pred_clipped = np.clip(lin_pred, -700, 700)
#     lambda_ = np.exp(lin_pred_clipped)

#     return lin_pred + age_coef * data["age"] - ((lambda_ * age_contrib) / age_coef)


# def _log_survival_function(data, params):
#     """
#     Calculate the log-survival function in a numerically stable way.

#     Uses np.expm1 for the exponential minus one calculation and clips
#     the linear predictor before exponentiating.

#     Returns:
#         A pandas Series with the log-survival for each observation.
#     """
#     age_coef = params.loc["age", "value"]
#     # age_contrib = np.expm1(age_coef * data["age"])
#     safe_age = np.clip(age_coef * data["age"], -700, 700)
#     age_contrib = np.expm1(safe_age)

#     lin_pred = sum(
#         [params.loc[x, "value"] * data[x] for x in params.index if x != "age"]
#     )
#     lin_pred_clipped = np.clip(lin_pred, -700, 700)
#     lambda_ = np.exp(lin_pred_clipped)

#     return -(lambda_ * age_contrib) / age_coef


# def loglike_estimagic(params, data, start_data):
#     """Log-likelihood calculation returning contributions and their sum.

#     The contribution for each individual is defined as:
#       - If the death event occurred:
#             log_density(individual) - log_survival(start_data_individual)
#       - If the event did not occur:
#             log_survival(individual) - log_survival(start_data_individual)

#     Returns a dict with:
#         "contributions": individual log-likelihood contributions,
#         "value": sum of contributions.
#     """
#     # Create a boolean Series indicating which individuals died
#     death = data["death event"].astype(bool)

#     # Initialize a Series to store the contributions (using the index from data)
#     contributions = pd.Series(index=data.index, dtype=float)

#     # For individuals with a death event, use the density part
#     contributions[death] = _log_density_function(
#         data[death], params
#     ) - _log_survival_function(start_data[death], params)

#     # For censored individuals, use the survival function
#     contributions[~death] = _log_survival_function(
#         data[~death], params
#     ) - _log_survival_function(start_data[~death], params)

#     return {"contributions": contributions, "value": contributions.sum()}


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
