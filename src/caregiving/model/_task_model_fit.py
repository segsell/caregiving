"""Functions to evaluate the fit of the simulated moments to the data moments."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask

from caregiving.config import BLD

_SIM_MOMENTS = np.array(
    [
        3.97190000e-01,
        3.43500000e-01,
        4.43190000e-01,
        3.57900000e-01,
        4.04360000e-01,
        3.50200000e-01,
        3.52920000e-01,
        3.24730000e-01,
        3.27700000e-01,
        3.21960000e-01,
        3.30890000e-01,
        3.42060000e-01,
        3.75110000e-01,
        4.26360000e-01,
        4.76910000e-01,
        5.29050000e-01,
        5.75760000e-01,
        6.41770000e-01,
        7.20600000e-01,
        7.97070000e-01,
        8.66850000e-01,
        9.15300000e-01,
        9.57800000e-01,
        9.88870000e-01,
        9.94840000e-01,
        9.97420000e-01,
        9.99170000e-01,
        9.99750000e-01,
        9.99910000e-01,
        9.99970000e-01,
        3.48790000e-01,
        2.83600000e-01,
        1.47780000e-01,
        2.04130000e-01,
        1.78730000e-01,
        2.39440000e-01,
        2.58270000e-01,
        3.22970000e-01,
        3.50650000e-01,
        3.75720000e-01,
        3.67790000e-01,
        3.64090000e-01,
        3.47470000e-01,
        3.35570000e-01,
        3.16690000e-01,
        2.92040000e-01,
        2.66870000e-01,
        2.22990000e-01,
        1.60160000e-01,
        1.02670000e-01,
        5.40200000e-02,
        2.68100000e-02,
        4.08000000e-03,
        3.40000000e-04,
        1.20000000e-04,
        3.00000000e-05,
        1.00000000e-05,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        2.54020000e-01,
        3.72900000e-01,
        4.09030000e-01,
        4.37970000e-01,
        4.16910000e-01,
        4.10360000e-01,
        3.88810000e-01,
        3.52300000e-01,
        3.21650000e-01,
        3.02320000e-01,
        3.01320000e-01,
        2.93850000e-01,
        2.77420000e-01,
        2.38070000e-01,
        2.06400000e-01,
        1.78910000e-01,
        1.57370000e-01,
        1.35240000e-01,
        1.19240000e-01,
        1.00260000e-01,
        7.91300000e-02,
        5.78900000e-02,
        3.81200000e-02,
        1.07900000e-02,
        5.04000000e-03,
        2.55000000e-03,
        8.20000000e-04,
        2.50000000e-04,
        9.00000000e-05,
        3.00000000e-05,
        -2.54494592e00,
        1.15201183e-01,
        -1.00526114e-03,
        1.17432400e-01,
        5.41931660e-02,
        6.46730981e-02,
        4.43863500e-01,
        3.55593712e-01,
        4.99436647e-01,
        8.48281648e-01,
        4.93794898e-01,
        4.89096023e-02,
        3.70494067e-01,
        3.52409871e-01,
        3.49844331e-01,
        3.53738916e-01,
        1.28076821e-01,
        2.18638972e-04,
        3.69276746e-01,
        3.56876224e-01,
        1.56642350e-01,
        8.13673665e-03,
        0.00000000e00,
        0.00000000e00,
        2.24545995e-01,
        2.96016463e-01,
        2.30109585e-01,
        2.19936597e-01,
        2.37119830e-01,
        2.19185570e-02,
        3.70494067e-01,
        3.52409871e-01,
        3.49844331e-01,
        3.53738916e-01,
        1.28076821e-01,
        2.18638972e-04,
        4.04959938e-01,
        3.51573666e-01,
        4.20046084e-01,
        4.26324487e-01,
        2.97697548e-01,
        2.04427439e-02,
        2.07662217e-04,
        6.83500718e-01,
        4.57518800e-02,
        2.70539740e-01,
        8.72692509e-01,
        7.22552390e-02,
        5.50522523e-02,
        3.72650469e-01,
        3.40938009e-01,
        2.86411522e-01,
        3.41230153e-01,
        2.38036367e-01,
        4.20733480e-01,
        7.49798549e-01,
        2.50201451e-01,
        3.73215871e-01,
        6.26784129e-01,
        9.10612799e-01,
        8.93872010e-02,
        7.63638570e-01,
        2.36361430e-01,
        6.42070696e-01,
        3.57929304e-01,
        3.71293813e-01,
        6.28706187e-01,
        8.70325833e-01,
        1.29674167e-01,
        7.51462716e-01,
        2.48537284e-01,
    ],
)

_SIM_MOMENTS = np.array(
    [
        4.03420000e-01,
        3.45380000e-01,
        4.41540000e-01,
        3.61050000e-01,
        4.01410000e-01,
        3.52200000e-01,
        3.53010000e-01,
        3.26690000e-01,
        3.23900000e-01,
        3.09630000e-01,
        3.09830000e-01,
        3.15190000e-01,
        3.40780000e-01,
        3.92640000e-01,
        4.45390000e-01,
        4.88950000e-01,
        5.21590000e-01,
        5.56790000e-01,
        6.02770000e-01,
        6.73000000e-01,
        7.71960000e-01,
        8.48610000e-01,
        9.27740000e-01,
        9.79110000e-01,
        9.90640000e-01,
        9.95780000e-01,
        9.98400000e-01,
        9.99560000e-01,
        9.99840000e-01,
        9.99970000e-01,
        3.42510000e-01,
        2.80120000e-01,
        1.45840000e-01,
        1.97840000e-01,
        1.74560000e-01,
        2.26690000e-01,
        2.48140000e-01,
        3.15510000e-01,
        3.57690000e-01,
        3.86950000e-01,
        3.84780000e-01,
        3.84580000e-01,
        3.72190000e-01,
        3.63650000e-01,
        3.49270000e-01,
        3.31030000e-01,
        3.20510000e-01,
        3.03300000e-01,
        2.67150000e-01,
        2.08300000e-01,
        1.20770000e-01,
        6.07600000e-02,
        8.19000000e-03,
        7.40000000e-04,
        3.60000000e-04,
        3.00000000e-05,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        2.54070000e-01,
        3.74500000e-01,
        4.12620000e-01,
        4.41110000e-01,
        4.24030000e-01,
        4.21110000e-01,
        3.98850000e-01,
        3.57800000e-01,
        3.18410000e-01,
        3.03420000e-01,
        3.05390000e-01,
        3.00230000e-01,
        2.87030000e-01,
        2.43710000e-01,
        2.05340000e-01,
        1.80020000e-01,
        1.57900000e-01,
        1.39910000e-01,
        1.30080000e-01,
        1.18700000e-01,
        1.07270000e-01,
        9.06300000e-02,
        6.40700000e-02,
        2.01500000e-02,
        9.00000000e-03,
        4.19000000e-03,
        1.60000000e-03,
        4.40000000e-04,
        1.60000000e-04,
        3.00000000e-05,
        -2.70000000e00,
        1.20976132e-01,
        -1.04914479e-03,
        1.07026740e-01,
        3.09536591e-02,
        4.83311217e-02,
        4.45148084e-01,
        3.44254292e-01,
        4.60257357e-01,
        7.60210342e-01,
        4.55801030e-01,
        3.51300913e-02,
        3.90908460e-01,
        3.65366962e-01,
        3.68445127e-01,
        3.80453622e-01,
        1.68027716e-01,
        9.54562810e-05,
        3.83678507e-01,
        3.79986122e-01,
        1.67012539e-01,
        4.47745130e-03,
        0.00000000e00,
        0.00000000e00,
        2.34713152e-01,
        3.12216000e-01,
        2.28071076e-01,
        2.12230072e-01,
        1.96819556e-01,
        1.84230622e-02,
        3.90908460e-01,
        3.65366962e-01,
        3.68445127e-01,
        3.80453622e-01,
        1.68027716e-01,
        9.54562810e-05,
        3.74378388e-01,
        3.22417038e-01,
        4.03483797e-01,
        4.07316306e-01,
        3.13224234e-01,
        3.06414662e-02,
        2.79214493e-04,
        9.32254517e-01,
        1.41887295e-02,
        5.32775392e-02,
        8.53062172e-01,
        8.83842805e-02,
        5.85535471e-02,
        4.00197443e-01,
        3.27189457e-01,
        2.72613100e-01,
        3.39431065e-01,
        2.42087625e-01,
        4.18481311e-01,
        7.28489232e-01,
        2.71510768e-01,
        3.79333008e-01,
        6.20666992e-01,
        9.50870245e-01,
        4.91297552e-02,
        8.21523466e-01,
        1.78476534e-01,
        6.20851890e-01,
        3.79148110e-01,
        2.64530428e-01,
        7.35469572e-01,
        9.17568668e-01,
        8.24313318e-02,
        7.11274245e-01,
        2.88725755e-01,
    ],
)

SIM_MOMENTS = np.array(
    [
        4.93122201e-01,
        2.06861597e-01,
        2.47225603e-01,
        2.02029146e-01,
        3.16645980e-01,
        2.48225944e-01,
        3.06562583e-01,
        2.60713786e-01,
        3.05740455e-01,
        2.76307116e-01,
        3.08866357e-01,
        2.83253773e-01,
        3.24646034e-01,
        3.05298175e-01,
        3.22556623e-01,
        3.07493449e-01,
        4.44866182e-01,
        4.29993191e-01,
        4.12649298e-01,
        4.12198513e-01,
        4.50120144e-01,
        4.46497540e-01,
        4.97120527e-01,
        8.50251879e-01,
        9.53222696e-01,
        9.37370579e-01,
        9.20353333e-01,
        9.27828881e-01,
        9.71225518e-01,
        9.70740158e-01,
        3.09500960e-01,
        6.85461344e-01,
        5.67623684e-01,
        6.44327615e-01,
        5.28794058e-01,
        6.11588135e-01,
        5.48765735e-01,
        6.01457634e-01,
        5.52317040e-01,
        5.89789404e-01,
        5.49401359e-01,
        5.81248142e-01,
        5.35647162e-01,
        5.59774279e-01,
        4.93379588e-01,
        5.08322119e-01,
        3.90650378e-01,
        4.08359960e-01,
        4.21767024e-01,
        4.28439018e-01,
        3.84020825e-01,
        3.07435866e-01,
        2.62196627e-01,
        7.98816568e-02,
        4.67773044e-02,
        6.26294215e-02,
        7.95314039e-02,
        7.07325269e-02,
        2.15940033e-02,
        1.25067340e-02,
        1.97376839e-01,
        1.07677059e-01,
        1.85150712e-01,
        1.53643239e-01,
        1.54559962e-01,
        1.40185921e-01,
        1.44671682e-01,
        1.37828579e-01,
        1.41942505e-01,
        1.33903480e-01,
        1.41732283e-01,
        1.35498085e-01,
        1.39706804e-01,
        1.34927546e-01,
        1.84063788e-01,
        1.84184432e-01,
        1.64483440e-01,
        1.61646849e-01,
        1.65583679e-01,
        1.59362470e-01,
        1.65859031e-01,
        2.46066593e-01,
        2.40682847e-01,
        6.98664641e-02,
        0.00000000e00,
        0.00000000e00,
        1.15262904e-04,
        1.43859207e-03,
        7.18047919e-03,
        1.67531082e-02,
        6.36252415e-01,
        -1.03551596e-02,
        1.80273538e-05,
        -6.01982892e-02,
        -5.46664959e-02,
        1.79804889e-01,
        5.87315766e-03,
        -3.47254916e-02,
        -1.23618014e-01,
        -2.72595461e-01,
        -6.00834693e-01,
        -1.36451750e-02,
        3.19705066e-02,
        1.36538369e-01,
        3.15525529e-01,
        -4.16641009e00,
        -2.11025991e-02,
        -6.53086403e-03,
        4.87762383e-02,
        1.74577161e-01,
        -2.36671772e00,
        2.48857087e-02,
        1.74447389e-02,
        -2.29053502e-02,
        -1.32714953e-01,
        3.07489892e-01,
        3.40791688e-01,
        3.68110441e-01,
        4.48839105e-01,
        6.56702966e-01,
        9.63148673e-01,
        4.27478928e-01,
        4.12055336e-01,
        3.77760781e-01,
        2.79726500e-01,
        1.27350777e-01,
        2.83052136e-02,
        2.65031179e-01,
        2.47152976e-01,
        2.54128778e-01,
        2.71434395e-01,
        2.15946257e-01,
        8.54611375e-03,
        6.65714568e-01,
        2.76517459e-01,
        5.77679733e-02,
        3.87989029e-01,
        4.51704969e-01,
        1.60306001e-01,
        3.82157066e-01,
        4.28884264e-01,
        1.88958670e-01,
        6.02863994e-01,
        3.97136006e-01,
        5.63797859e-01,
        4.36202141e-01,
        9.83946809e-01,
        1.60531913e-02,
        9.82212481e-01,
        1.77875187e-02,
        5.92627093e-01,
        4.07372907e-01,
        5.65979222e-01,
        4.34020778e-01,
        9.83497339e-01,
        1.65026611e-02,
        9.81857652e-01,
        1.81423477e-02,
    ],
)


@pytask.mark.skip()
def task_model_fit(
    path_to_empirical_moments=BLD
    / "moments"
    / "empirical_moments_unconstrained_caregiving_option.csv",
):
    """Evaluate the fit of the simulated moments to the data moments."""
    empirical_moments = pd.read_csv(path_to_empirical_moments, index_col=0).iloc[:, 0]

    simulated_moments = add_index_to_simulated_moments(
        empirical_moments=empirical_moments,
        simulated_moments=SIM_MOMENTS,
    )

    comparison = pd.concat([empirical_moments, simulated_moments], axis=1)
    comparison.columns = ["empirical", "simulated"]

    plot_labor_shares_informal_care(
        comparison,
        "not_working",
        path_to_save=BLD / "descriptives/model_fit/model_fit_caregivers_no_work.png",
    )
    plot_labor_shares_informal_care(
        comparison,
        "part_time",
        path_to_save=BLD / "descriptives/model_fit/model_fit_caregivers_part_time.png",
    )
    plot_labor_shares_informal_care(
        comparison,
        "full_time",
        path_to_save=BLD / "descriptives/model_fit/model_fit_caregivers_full_time.png",
    )

    plot_labor_shares(comparison, "not_working")
    plot_labor_shares(comparison, "part_time")
    plot_labor_shares(comparison, "full_time")


def add_index_to_simulated_moments(
    empirical_moments: pd.DataFrame,
    simulated_moments: np.ndarray,
) -> pd.DataFrame:
    """Convert simulated moments to pandas DataFrame.

    Args:
        empirical_moments (pd.DataFrame): Empirical moments.
        simulated_moments (np.array): Simulated moments.

    Returns:
        pd.DataFrame: Simulated moments as pandas DataFrame.

    """
    idx = dict(enumerate(empirical_moments.index, start=0))

    return pd.Series(simulated_moments, index=idx.values(), name="moment")


def plot_labor_shares(df, working_status):
    """Plots the labor shares of empirical and simulated moments.

    Args:
        df (pandas.DataFrame): DataFrame with the index as "working_status_age_x
            and two columns 'empirical' and 'simulated'.
        working_status (str): Working status to filter the DataFrame on.

    # ages = filtered_df.index.str.replace(pattern, "").astype(
    #     int
    # )

    """
    # Prepare the data: filter rows for the specified working status and extract age
    pattern = f"{working_status}_age_"
    filtered_df = df[df.index.str.startswith(pattern)]
    ages = np.array(
        [
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
        ],
    )

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(ages, filtered_df["empirical"][:30], label="Empirical Moments", marker="o")
    plt.plot(
        ages,
        filtered_df["simulated"][:30],
        label="Simulated Moments",
        linestyle="--",
        marker="x",
    )

    plt.title(f"Labor Shares - {working_status.capitalize()}")
    plt.xlabel("Age")
    plt.ylabel("Labor Shares")
    plt.xticks(
        ages,
        rotation=45,
    )  # Set x-ticks to be the ages, rotate for better visibility
    plt.legend()
    plt.grid()
    plt.tight_layout()


def plot_labor_shares_informal_care(df, working_status, path_to_save):
    """Plots the labor shares of empirical and simulated moments.

    Args:
        df (pandas.DataFrame): DataFrame with the index as
            "working_status_age_lower_upper" and two columns 'empirical' and
            'simulated'.
        working_status (str): Working status to filter the DataFrame on.
        path_to_save (str): Path to save the plot.

    """
    # Define the age intervals
    age_intervals = [(40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70)]

    # Initialize lists to store the data
    ages = []
    empirical_values = []
    simulated_values = []

    # Iterate through each interval and extract the data
    for lower, upper in age_intervals:
        pattern = f"{working_status}_age_{lower}_{upper}"
        if pattern in df.index:
            ages.append(f"{lower}-{upper}")
            empirical_values.append(df.loc[pattern, "empirical"])
            simulated_values.append(df.loc[pattern, "simulated"])

    # Convert ages to a numpy array for plotting
    ages = np.array(ages)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(ages, empirical_values, label="Empirical", marker="o")
    plt.plot(ages, simulated_values, label="Simulated", linestyle="--", marker="x")

    plt.ylim(0, 1)  # Adjust the limits as needed
    plt.xlabel("Age Bin", fontsize=16)
    plt.ylabel("Share", fontsize=16)
    plt.xticks(rotation=45, fontsize=16)  # Rotate for better visibility
    plt.yticks(fontsize=16)
    plt.legend(prop={"size": 16})
    plt.grid()
    plt.tight_layout()

    if path_to_save is not None:
        plt.savefig(path_to_save)


# ====================================================================================


def plot_working_status(empirical, simulated, working_status):
    # Extract ages and corresponding shares from empirical
    ages_empirical = [
        int(idx.split("_")[-1])
        for idx in empirical.index
        if idx.startswith(f"{working_status}_age_")
    ]
    shares_empirical = empirical.loc[
        empirical.index.str.startswith(f"{working_status}_age_")
    ].to_numpy()

    # Extract ages and corresponding shares from simulated
    ages_simulated = [
        int(idx.split("_")[-1])
        for idx in simulated.index
        if idx.startswith(f"{working_status}_age_")
    ]
    shares_simulated = simulated.loc[
        simulated.index.str.startswith(f"{working_status}_age_")
    ].to_numpy()

    # Create DataFrames for easier plotting
    df_empirical = pd.DataFrame(
        {
            "Age": ages_empirical,
            f"Share of {working_status.capitalize()} (Empirical)": shares_empirical,
        },
    )
    df_simulated = pd.DataFrame(
        {
            "Age": ages_simulated,
            f"Share of {working_status.capitalize()} (Simulated)": shares_simulated,
        },
    )

    # Sort the DataFrames by Age for better visualization
    df_empirical = df_empirical.sort_values(by="Age")
    df_simulated = df_simulated.sort_values(by="Age")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(
        df_empirical["Age"],
        df_empirical[f"Share of {working_status.capitalize()} (Empirical)"],
        marker="o",
        linestyle="-",
        label="Empirical",
    )
    plt.plot(
        df_simulated["Age"],
        df_simulated[f"Share of {working_status.capitalize()} (Simulated)"],
        marker="s",
        linestyle="--",
        label="Simulated",
    )
    plt.title(f"Share of {working_status.capitalize()} People by Age")
    plt.xlabel("Age")
    plt.ylabel(f"Share of {working_status.capitalize()}")
    plt.legend()
    plt.grid()
    plt.show()
