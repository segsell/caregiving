import numpy as np


def create_end_of_period_assets():
    """Create a saving grid with sections."""
    section_1 = np.arange(start=0, stop=20, step=5)  # 4
    section_2 = np.arange(start=20, stop=50, step=10)  # 3
    section_3 = np.arange(start=50, stop=300, step=25)  # 10
    section_4 = np.arange(start=300, stop=500, step=50)  # 4
    section_5 = np.arange(start=500, stop=1000, step=100)  # 5
    section_6 = [5_000, 10_000]  # 2
    savings_grid = np.concatenate(
        [
            section_1,
            section_2,
            section_3,
            section_4,
            section_5,
            section_6,
        ]
    ).astype(float)
    # The steps above are made in thousands, so we multiply by 1000
    return savings_grid * 1_000


def create_savings_grid_deprecated():
    """Create a saving grid with sections."""
    section_1 = np.arange(start=0, stop=20, step=2)  # 10
    section_2 = np.arange(start=20, stop=50, step=5)  # 6
    section_3 = np.arange(start=50, stop=100, step=10)  # 3
    section_4 = np.arange(start=100, stop=500, step=100)  # 4
    section_5 = np.arange(start=500, stop=1000, step=200)  # 3
    section_6 = np.arange(start=1000, stop=10000, step=2000)  # 5
    section_7 = np.array([10_000, 100_000])
    savings_grid = np.concatenate(
        [section_1, section_2, section_3, section_4, section_5, section_6, section_7]
    )
    return savings_grid


# def create_savings_grid():
#     """Create a saving grid with sections."""
#     section_1 = np.arange(start=0, stop=20, step=5)  # 4
#     section_2 = np.arange(start=20, stop=50, step=10)  # 3
#     section_3 = np.arange(start=50, stop=300, step=25)  # 10
#     section_4 = np.arange(start=300, stop=500, step=50)  # 4
#     section_5 = np.arange(start=500, stop=1000, step=100)  # 5
#     section_6 = [5_000, 10_000]  # 2
#     savings_grid = np.concatenate(
#         [
#             section_1,
#             section_2,
#             section_3,
#             section_4,
#             section_5,
#             section_6,
#         ]
#     ).astype(float)
#     # The steps above are made in thousands, so we multiply by 1000
#     return savings_grid * 1_000
