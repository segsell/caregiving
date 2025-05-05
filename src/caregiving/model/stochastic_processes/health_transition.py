from caregiving.model.shared import MOTHER, SEX


def health_transition(health, education, period, options):
    """Transition probability for next period health state."""
    trans_mat = options["health_trans_mat"]
    prob_vector = trans_mat[SEX, education, period, health, :]

    return prob_vector


def health_transition_good_medium_bad(health, education, has_sister, period, options):
    """Transition probability for next period health state."""
    trans_mat = options["health_trans_mat_three"]
    mother_age = period + options["mother_age_difference"][has_sister, education, :]
    prob_vector = trans_mat[MOTHER, education, mother_age, health, :]

    return prob_vector
