from caregiving.model.shared import MOTHER, SEX


def health_transition(health, education, period, options):
    """Transition probability for next period health state."""
    trans_mat = options["health_trans_mat"]
    prob_vector = trans_mat[SEX, education, period, health, :]

    return prob_vector


def health_transition_good_medium_bad(health, education, period, options):
    """Transition probability for next period health state."""
    trans_mat = options["health_trans_mat_three"]
    age = period + options["mother_age_difference"]
    prob_vector = trans_mat[MOTHER, education, age, health, :]

    return prob_vector
