from functools import partial

import jax.numpy as jnp
from optimagic.differentiation.derivatives import first_derivative

# from caregiving.estimate import get_moment_error_vec


# def get_analytical_standard_errors(
#     params: dict,
#     params_fixed: dict,
#     options: dict,
#     emp_moments: jnp.array,
#     emp_var: jnp.array,
#     model_loaded: dict,
#     solve_func: callable,
#     initial_states: dict,
#     initial_resources: jnp.array,
# ):
#     """Get analytical standard errors.

#     weighting_mat = jnp.linalg.inv(covariance)

#     """
#     covariance = jnp.diag(emp_var)
#     weighting_mat = jnp.diag(emp_var ** (-1))

#     get_error_partial = partial(
#         get_moment_error_vec,
#         params_fixed=params_fixed,
#         options=options,
#         emp_moments=emp_moments,
#         model_loaded=model_loaded,
#         solve_func=solve_func,
#         initial_states=initial_states,
#         initial_resources=initial_resources,
#     )

#     jac = first_derivative(
#         func=get_error_partial,
#         params=params,
#         base_steps=0.01,
#         method="forward",
#     )
#     _jacobian = list(jac["derivative"].values())

#     jacobian = jnp.stack(_jacobian).T

#     bread = jnp.linalg.inv(jacobian.T @ weighting_mat @ jacobian)
#     butter = jacobian.T @ weighting_mat @ covariance @ weighting_mat @ jacobian
#     variance = bread @ butter @ bread

#     return jnp.sqrt(jnp.diag(variance))
