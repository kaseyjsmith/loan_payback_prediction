# %%
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import chi2_contingency

try:
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent.__str__()
except Exception as e:
    proj_root = "/home/ksmith/birds/kaggle/predicting_load_payback/"

plot_dir = proj_root + "plots/"

# %%
train_df = pd.read_csv(proj_root + "/data/train.csv")
train_df.head()


# %% Plotting Setup
title_dict = dict(fontweight="bold", fontsize="18")
axes_dict = dict(fontweight="bold")

# %% Sort grade_subgrade column (makes the interpretation of that chart easier)
# Create ordered categorical from existing unique values
grade_order = sorted(train_df["grade_subgrade"].unique())
train_df["grade_subgrade"] = pd.Categorical(
    train_df["grade_subgrade"], categories=grade_order, ordered=True
)

# %% Plot each of the 6 categorical features by loan_paid_back
nrows = 3
ncols = 2
fig, ax = plt.subplots(nrows, ncols, figsize=(18, 10))

fig.suptitle(
    "Proportion of various dimensions on\npaying back loan", **title_dict
)

row = 0
col = 0
for column in train_df.columns[6:12]:
    sns.histplot(
        data=train_df,
        x=column,
        hue="loan_paid_back",
        multiple="fill",
        stat="proportion",
        discrete=True,
        shrink=0.8,
        ax=ax[row][col],
        legend=False,
    )
    ax[row][col].title.set_text(column.replace("_", " ").title())
    print(f"row: {row}, col: {col}")

    # Move to next subplot position
    col += 1
    if col >= ncols:
        col = 0
        row += 1


fig.legend(["Paid back", "Did not pay back"], loc="upper right")
plt.tight_layout()
# plt.subplots_adjust(right=0.9)
plt.savefig(plot_dir + "loan_payback_status_categoricals.png")

"""
All things considered, the data looks like you'd except. There's almost no deviaiton across values in Gender, Marital Status, Education level, and Loan Purpose. Grade Subgrade is what I'd expect too, from grade A loans being paid back with much higher frequency than F grade. 

The standout differences are in Employment status. Unemployed is _heavily_ skewed to not paying back and students are about 75% not payed back.
"""

# %% Plot continuous variables
nrows = 3
ncols = 2
fig, ax = plt.subplots(nrows, ncols, figsize=(18, 10))

fig.suptitle(
    "Proportion of various dimensions on\npaying back loan", **title_dict
)

row = 0
col = 0
for column in train_df.columns[1:6]:
    data_range = train_df[column].max() - train_df[column].min()
    sns.histplot(
        data=train_df,
        x=column,
        hue="loan_paid_back",
        multiple="fill",
        stat="proportion",
        binwidth=data_range / 10,
        ax=ax[row][col],
        legend=False,
    )
    ax[row][col].title.set_text(column.replace("_", " ").title())
    print(f"row: {row}, col: {col}")

    # Move to next subplot position
    col += 1
    if col >= ncols:
        col = 0
        row += 1


fig.legend(["Paid back", "Did not pay back"], loc="upper right")
plt.tight_layout()
plt.savefig(plot_dir + "loan_payback_status_continuous.png")

"""
Similar unsurprising findings here.

There is a strong negative relationship with debt to income ratio as well as interest rate and whether or not the loan is paid back. Similarly, a positive relationship with credit score. 

Income has some minor fluctuations and loan amount has a small drop off at higher loan amounts.
"""

# %% Categorical correlation calculations using chi^2


# Continuous variables correlation with target (straightforward)
continuous_cols = train_df.columns[1:6].tolist()
continuous_corr = (
    train_df[continuous_cols + ["loan_paid_back"]]
    .corr()["loan_paid_back"]
    .drop("loan_paid_back")
)
print("Continuous Variables Correlation with Target:")
print(continuous_corr.sort_values(ascending=False))

# %% Categorical variables - Chi-squared test
categorical_cols = train_df.columns[6:12].tolist()

chi2_results = {}
for col in categorical_cols:
    contingency_table = pd.crosstab(train_df[col], train_df["loan_paid_back"])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    chi2_results[col] = {"chi2": chi2, "p_value": p_value}

chi2_df = pd.DataFrame(chi2_results).T.sort_values("chi2", ascending=False)
print("\nChi-Squared Test Results for Categorical Variables:")
print(chi2_df)

"""
This tells me that for the continuous variables, the big hitters are:
- credit_score
- interest_rate
- debt_to_income_ratio

and for the categorical variables, the big hitters are:
- employment_status
- grade_subgrade

The others seem to be minimal at best. It seems that I _could_ consider loan purpose and education level, but I'll only do that if the performance of my model isn't great.
"""

# %%

"""
FINDINGS AND NEXT STEPS:

I've shown objectively what the intuitive hypothesis is about what impacts someone's liklihood of paying back a loan. High credit score borrowers that get low interest rate loans and have a low debt to income ratio that are employed have good grading on their loans and tend to pay them back. 

TO NOTE: the statement above implies that there is _likely_ some relation between these variables. For now, I'm not going to dig into that too much, but I likely will need to come back to understand their relatioship to one another. 

Variables to include in training:
- credit_score
- interest_rate
- debt_to_income_ratio
- employment_status
- grade_subgrade

"""
