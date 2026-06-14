import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import dcor  # For distance correlation (if needed)
from minepy import MINE  # For maximal information coefficient (MIC)
import statsmodels.api as sm


def compute_distance_corr(x, y):

    return dcor.distance_correlation(x, y)

def compute_mic(x, y, alpha=0.6, c=15):

    mine = MINE(alpha=alpha, c=c)
    mine.compute_score(x, y)
    return mine.mic()

def screen_predictors(df, target_col, method="mic", threshold=0.2):

    predictors = [col for col in df.columns if col != target_col]
    selected = []
    for col in predictors:
        x = df[col].values
        y = df[target_col].values

        if method == "mic":
            score = compute_mic(x, y)
        elif method == "dcor":
            score = compute_distance_corr(x, y)
        else:
            raise ValueError("Unknown method specified.")

        # Debug print (can be commented out)
        print(f"{target_col} - {col} ({method}): {score:.3f}")
        if score >= threshold:
            selected.append(col)
    return selected

def report_high_correlation_pairs(df, cols, corr_threshold=0.9):
    print(f"=== High Correlation Pairs (|corr| >= {corr_threshold}) ===")
    corr_matrix = df[cols].corr().abs()
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1, col2 = cols[i], cols[j]
            corr_val = corr_matrix.loc[col1, col2]
            if corr_val >= corr_threshold:
                pairs.append((col1, col2, corr_val))
                print(f"{col1} and {col2}: correlation = {corr_val:.3f}")
    if not pairs:
        print("No pairs found above the threshold.")
    print("")
    return pairs


def report_vif(df, cols):
    print("=== Variance Inflation Factor (VIF) ===")
    X = df[cols].copy()
    X = sm.add_constant(X, has_constant='add')

    vif_data = []
    for i in range(X.shape[1]):
        col_name = X.columns[i]
        if col_name == 'const':
            continue
        vif_val = variance_inflation_factor(X.values, i)
        vif_data.append((col_name, vif_val))

    vif_data.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Variable':<15} VIF")
    print("-" * 28)
    for var, vif_val in vif_data:
        print(f"{var:<15} {vif_val:>6.2f}")
    print("")
    return vif_data


def cluster_columns_by_correlation(df, cols, corr_threshold=0.9):

    corr = df[cols].corr().abs()
    # Convert correlation to a distance measure
    dist = 1 - corr
    # Convert to condensed distance matrix (required by linkage)
    condensed_dist = squareform(dist.values, checks=False)
    Z = linkage(condensed_dist, method='average')

    # fcluster: all columns in a cluster have distances below threshold = (1 - corr_threshold)
    cluster_labels = fcluster(Z, t=1 - corr_threshold, criterion='distance')

    clusters = {}
    for col, label in zip(cols, cluster_labels):
        clusters.setdefault(label, []).append(col)

    print(f"=== Clusters of variables (correlation threshold = {corr_threshold}) ===")
    for label, cluster_cols in clusters.items():
        if len(cluster_cols) > 1:
            print(f"Cluster {label}: {cluster_cols}")
    print("")
    return clusters

def analyze_links(df, corr_threshold=0.9, vif_threshold=10.0):
    # Build a list of columns starting with "E_" or "D_"
    link_cols = [col for col in df.columns if col.startswith("E_") or col.startswith("D_")]
    # Add the "S" column if it exists in the DataFrame
    if "S" in df.columns:
        link_cols.append("S")

    for target in link_cols:
        print("=" * 70)
        print(f" ANALYSIS FOR TARGET: {target}")
        print("=" * 70)

        # Use all other link_cols as predictors
        predictors = [col for col in link_cols if col != target]
        X = df[predictors]
        y = df[target]

        # 3) Report high correlations among the predictors
        report_high_correlation_pairs(df, predictors, corr_threshold=corr_threshold)

        # 4) Report VIF among the predictors
        report_vif(df, predictors)

        # 5) OLS Regression
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm)
        results = model.fit()

        print(results.summary())

###############################################################################
#            SYMBOLIC REGRESSION ANALYSIS & SAVING (OPTIONAL)
###############################################################################
early_stop_condition_str = "(iter, best) -> (iter - best) > 100"

# -------------------------------------------------------------------------------------------
# Optional: These cluster-based methods are kept if you want more advanced grouping approaches
# -------------------------------------------------------------------------------------------

def orthogonalize_columns(df, cols):
    """
    Applies Gram-Schmidt orthogonalization to the specified columns in df.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        cols (list): A list of column names (strings) to orthogonalize.

    Returns:
        pd.DataFrame: A new DataFrame in which the columns in `cols` have been replaced
                      with their orthogonalized versions.
    """
    X = df[cols].values.astype(float)
    Q = np.zeros_like(X)
    for i in range(X.shape[1]):
        v = X[:, i].copy()
        for j in range(i):
            proj_coef = np.dot(Q[:, j], X[:, i]) / np.dot(Q[:, j], Q[:, j])
            v = v - proj_coef * Q[:, j]
        Q[:, i] = v

    df_ortho = df.copy()
    for idx, col in enumerate(cols):
        df_ortho[col] = Q[:, idx]
    return df_ortho


def cluster_columns_by_correlation(df, cols, correlation_threshold=0.9):
    corr = df[cols].corr().abs()
    dist = 1 - corr  # distance = 1 - correlation
    condensed_dist = squareform(dist.values, checks=False)
    Z = linkage(condensed_dist, method='average')

    cluster_labels = fcluster(Z, t=1 - correlation_threshold, criterion='distance')

    clusters = {}
    for col, label in zip(cols, cluster_labels):
        clusters.setdefault(label, []).append(col)
    return clusters


def process_dataframe_for_orthogonalization(df, cols, correlation_threshold=0.9):
    clusters = cluster_columns_by_correlation(df, cols, correlation_threshold)
    for cluster in clusters.values():
        if len(cluster) > 1:
            df = orthogonalize_columns(df, cluster)
    return df
