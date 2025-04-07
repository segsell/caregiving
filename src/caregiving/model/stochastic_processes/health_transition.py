def health_transition(sex, health, education, period, options):
    """Transition probability for next period health state."""
    trans_mat = options["health_trans_mat"]
    prob_vector = trans_mat[sex, education, period, health, :]

    return prob_vector
