def shock_function_dict():
    return {
        "taste_shock_scale_per_state": taste_shock_for_women,
    }


def taste_shock_for_women(params):
    return params["taste_shock_scale"]
