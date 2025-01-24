"""Partner transition process."""


def partner_transition(period, education, has_partner, options):
    trans_mat = options["partner_trans_mat"]
    trans_vector = trans_mat[education, period, has_partner]

    return trans_vector
