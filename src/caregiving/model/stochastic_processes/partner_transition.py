"""Partner transition process."""


def partner_transition(period, education, partner_state, options):
    """Transition probability for next period partner state."""
    trans_mat = options["partner_trans_mat"]
    trans_vector = trans_mat[education, period, partner_state]

    return trans_vector
