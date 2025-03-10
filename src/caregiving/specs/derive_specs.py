"""Derive specs from the specs file."""

import yaml


def read_and_derive_specs(spec_path):
    """Read and derive specs from the specs file."""
    specs = yaml.safe_load(open(spec_path))

    # Number of periods in model
    specs["n_periods"] = specs["end_age"] - specs["start_age"] + 1

    # Number of education types and choices from labels
    specs["n_education_types"] = len(specs["education_labels"])
    specs["n_sexes"] = len(specs["sex_labels"])
    specs["n_choices"] = len(specs["choice_labels"])

    # Partner states
    specs["n_partner_states"] = len(specs["partner_labels"])

    return specs
