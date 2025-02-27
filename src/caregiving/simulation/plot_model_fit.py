import matplotlib.pyplot as plt
import numpy as np

from caregiving.model.shared import SEX


def plot_average_wealth(
    data_emp,
    data_sim,
    specs,
    path_to_save_plot,
):

    data_emp.loc[:, "age"] = data_emp["period"] + specs["start_age"]
    data_sim.loc[:, "age"] = data_sim["period"] + specs["start_age"]

    sex = SEX
    for edu in range(2):
        mask_sim = (data_sim["sex"] == sex) & (data_sim["education"] == edu)
        data_sim_edu = data_sim[mask_sim]
        mask_obs = (data_emp["sex"] == sex) & (data_emp["education"] == edu)
        data_decision_edu = data_emp[mask_obs]

        ages = np.arange(specs["start_age"] + 1, 90)
        average_wealth_sim = (
            data_sim_edu.groupby("age")["wealth_at_beginning"].median().loc[ages]
        )
        average_wealth_obs = (
            data_decision_edu.groupby("age")["adjusted_wealth"].median().loc[ages]
        )

        fig, ax = plt.subplots()

        ax.plot(ages, average_wealth_sim, label="Simulated")
        ax.plot(
            ages,
            average_wealth_obs,
            label="Median observed wealth by age",
            ls="--",
        )
        ax.legend()
        fig.savefig(path_to_save_plot, transparent=True, dpi=300)


# def plot_choice_shares_single(
#     path_dict,
#     specs,
#     params,
#     model_name,
#     load_df=True,
#     load_solution=True,
#     load_sol_model=True,
#     load_sim_model=True,
# ):
#     # alpha_belief = float(np.loadtxt(
#     # path_dict["est_results"] + "exp_val_params.txt"))

#     # Simulate baseline with subjective belief
#     data_sim = solve_and_simulate_scenario(
#         path_dict=path_dict,
#         params=params,
#         sim_alpha=0.0,
#         expected_alpha=False,
#         resolution=False,
#         initial_SRA=67,
#         model_name=model_name,
#         df_exists=load_df,
#         solution_exists=load_solution,
#         sol_model_exists=load_sol_model,
#         sim_model_exists=load_sim_model,
#     ).reset_index()

#     data_decision = pd.read_pickle(
#         path_dict["intermediate_data"] + "structural_estimation_sample.pkl"
#     )

#     data_decision["age"] = data_decision["period"] + specs["start_age"]
#     data_sim["age"] = data_sim["period"] + specs["start_age"]

#     for sex in range(specs["n_sexes"]):
#         fig, axes = plt.subplots(2, specs["n_choices"])
#         for edu_var, edu_label in enumerate(specs["education_labels"]):
#             data_sim_restr = data_sim[data_sim["sex"] == sex]
#             data_decision_restr = data_decision[data_decision["sex"] == sex]

#             choice_shares_sim = (
#                 data_sim_restr.groupby(["age"])["choice"]
#                 .value_counts(normalize=True)
#                 .unstack()
#             )
#             choice_shares_obs = (
#                 data_decision_restr.groupby(["age"])["choice"]
#                 .value_counts(normalize=True)
#                 .unstack()
#             )
#             if sex == 0:
#                 choice_range = [0, 1, 3]
#             else:
#                 choice_range = range(4)

#             for choice in choice_range:
#                 ax = axes[edu_var, choice]
#                 choice_share_sim = choice_shares_sim[choice]
#                 choice_share_obs = choice_shares_obs[choice]
#                 ax.plot(choice_share_sim, label=f"Simulated")
#                 ax.plot(choice_share_obs, label=f"Observed", ls="--")
#                 choice_label = specs["choice_labels"][choice]
#                 ax.set_title(f"{edu_label}; Choice {choice_label}")
#                 ax.set_ylim([0, 1])
#                 ax.legend()


# def plot_choice_shares(paths):
#     data_sim = pd.read_pickle(
#         paths["intermediate_data"] + "sim_data/data_subj_scale_1.pkl"
#     ).reset_index()
#     data_decision = pd.read_pickle(
#         paths["intermediate_data"] + "structural_estimation_sample.pkl"
#     )

#     specs = yaml.safe_load(open(paths["specs"]))

#     data_decision["age"] = data_decision["period"] + specs["start_age"]
#     data_sim["age"] = data_sim["period"] + specs["start_age"]

#     data_sim.groupby(["age"])["choice"].value_counts(normalize=True).unstack().plot(
#         title="Simulated choice shares by age", kind="bar", stacked=True
#     )

#     data_decision.groupby(["age"])["choice"].value_counts(
#         normalize=True
#     ).unstack().plot(title="Observed choice shares by age", kind="bar", stacked=True)


# def plot_states(paths, specs):
#     data_sim = pd.read_pickle(
#         paths["intermediate_data"] + "sim_data/data_subj_scale_1.pkl"
#     ).reset_index()
#     data_decision = pd.read_pickle(
#         paths["intermediate_data"] + "structural_estimation_sample.pkl"
#     )

#     data_decision["age"] = data_decision["period"] + specs["start_age"]
#     data_sim["age"] = data_sim["period"] + specs["start_age"]

#     params = pickle.load(open(paths["est_results"] + "est_params_pete.pkl", "rb"))
#     # Generate model_specs
#     model, params = specify_model(
#         path_dict=paths,
#         update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
#         policy_state_trans_func=expected_SRA_probs_estimation,
#         params=params,
#         load_model=True,
#         model_type="solution",
#     )
#     discrete_state_names = model["model_structure"]["discrete_states_names"]

#     for state_name in discrete_state_names:
#         data_decision.groupby(["age"])[state_name].value_counts(
#             normalize=True
#         ).unstack().plot()
#         data_sim.groupby(["age"])[state_name].value_counts(
#             normalize=True
#         ).unstack().plot()
#         plt.show()


# def illustrate_simulated_data(paths):
#     df = pd.read_pickle(
#         paths["intermediate_data"] + "sim_data/data_subj_scale_1.pkl"
#     ).reset_index()
#     # import matplotlib.pyplot as plt
#     #
#     # # %%
#     # # Plot choice shares by age
#     # df.groupby(["age"]).choice.value_counts(normalize=True).unstack().plot(
#     #     title="Choice shares by age"
#     # )
#     #
#     # # %%fig_1 = (
#     #
#     # # plot average income by age and choice
#     # df.groupby(["age", "choice"])["labor_income"].mean().unstack().plot(
#     #     title="Average income by age and choice"
#     # )
#     # # plot average income by age and choice
#     # df.groupby(["age", "choice"])["total_income"].mean().unstack().plot(
#     #     title="Average total income by age and choice"
#     # )
#     # %%
#     # plot average consumption by age and choice
#     # df.groupby(["age", "choice"])["consumption"].mean().unstack().plot(
#     #     title="Average consumption by age and choice"
#     # )
#     # %%
#     fig, ax = plt.subplots()
#     # plot average periodic savings by age and choice
#     df.groupby("age")["savings_dec"].mean().plot(ax=ax, label="Average savings")
#     ax.set_title("Average savings by age")
#     ax.legend()
#     fig.savefig(paths["plots"] + "average_savings.png", transparent=True, dpi=300)
