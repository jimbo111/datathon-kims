"""Statistical analysis utilities for datathon work."""

import pandas as pd
import numpy as np
from scipy import stats


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Extended describe — adds skew, kurtosis, IQR, and outlier counts."""
    desc = df.describe().T
    numeric = df.select_dtypes(include="number")
    desc["skew"] = numeric.skew()
    desc["kurtosis"] = numeric.kurtosis()
    desc["iqr"] = desc["75%"] - desc["25%"]
    desc["outliers_iqr"] = [
        ((numeric[c] < desc.loc[c, "25%"] - 1.5 * desc.loc[c, "iqr"])
         | (numeric[c] > desc.loc[c, "75%"] + 1.5 * desc.loc[c, "iqr"])).sum()
        for c in numeric.columns
    ]
    return desc


def test_normality(df: pd.DataFrame, cols: list[str] | None = None, alpha: float = 0.05) -> pd.DataFrame:
    """Shapiro-Wilk normality test for numeric columns."""
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    results = []
    for col in cols:
        data = df[col].dropna()
        if len(data) > 5000:
            data = data.sample(5000, random_state=42)
        stat, p = stats.shapiro(data)
        results.append({"column": col, "statistic": stat, "p_value": p, "normal": p > alpha})
    return pd.DataFrame(results)


def chi_squared_test(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Chi-squared test of independence between two categorical columns."""
    ct = pd.crosstab(df[col_a], df[col_b])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    return {"chi2": chi2, "p_value": p, "dof": dof, "cramers_v": np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))}


def group_compare(df: pd.DataFrame, group_col: str, value_col: str) -> dict:
    """Compare a numeric column across groups (t-test for 2, ANOVA for 3+)."""
    groups = [g[value_col].dropna().values for _, g in df.groupby(group_col)]
    if len(groups) == 2:
        stat, p = stats.ttest_ind(*groups)
        test_name = "t-test"
    else:
        stat, p = stats.f_oneway(*groups)
        test_name = "ANOVA"
    return {"test": test_name, "statistic": stat, "p_value": p, "n_groups": len(groups)}
