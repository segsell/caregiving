from caregiving.model.shared import SEX


def health_transition(health, education, period, model_specs):
    """Transition probability for next period health state."""
    trans_mat = model_specs["health_trans_mat"]
    prob_vector = trans_mat[SEX, education, period, health, :]

    return prob_vector
