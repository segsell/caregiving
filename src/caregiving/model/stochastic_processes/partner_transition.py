"""Partner transition process."""

from caregiving.model.shared import SEX


def partner_transition(period, education, partner_state, model_specs):
    """Transition probability for next period partner state."""
    trans_mat = model_specs["partner_trans_mat"]
    trans_vector = trans_mat[SEX, education, period, partner_state]

    return trans_vector
