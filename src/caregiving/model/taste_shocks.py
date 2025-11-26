def shock_function_dict():
    return {
        "taste_shock_scale_per_education": taste_shock_for_education_levels,
    }


def taste_shock_for_education_levels(education, params):
    return (
        params["taste_shock_scale_low_educ"] * (1 - education)
        + params["taste_shock_scale_high_educ"] * education
    )
