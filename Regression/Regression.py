
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from pysr import PySRRegressor
from IPython.display import clear_output


from File_processing import load_csv_and_prune
from Stats_methods  import screen_predictors


def quick_r2(df, target, predictors):
    """Return the R² of `target ~ predictors`."""
    if not predictors:
        return 0.0
    X = df[predictors].values
    y = df[target].values
    model = LinearRegression(fit_intercept=True).fit(X, y)
    return r2_score(y, model.predict(X))


def append_sigma_bins(df, n_bins: int = 3, col_name: str = "σ_bin"):
    if "S" not in df.columns:
        return df
    df = df.copy()
    df[col_name] = pd.qcut(df["S"], q=n_bins, labels=False)   # int labels 0…n_bins‑1
    return df

def cluster_rows(df, feature_cols, n_clusters=3, random_state=42):
    X = StandardScaler().fit_transform(df[feature_cols].values)
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state, n_init="auto").fit(X)
    sil = silhouette_score(X, kmeans.labels_)
    print(f"Cluster silhouette score = {sil:5.3f}")
    dfc = df.copy()
    dfc["cluster_id"] = kmeans.labels_
    return dfc


def run_guided_pipeline(
    df: pd.DataFrame,
    *,
    csv_out="equations.csv",
    txt_out="symbolic_log.txt",
    r2_gate=0.96,
    pysr_loss_gate=1e-8,
    pysr_iterations=4000,
    nl_method="mic",
    nl_threshold=0.6,
):
    link_cols = [c for c in df.columns
                 if (c.startswith("E_") or c.startswith("D_")) and
                    int(c.split("_")[1]) < 9]
    if "S" in df.columns:
        link_cols.append("S")

    accepted = []

    with open(txt_out, "w") as log:
        log.write("Guided symbolic‑regression pipeline\n" + "="*80 + "\n")

        for target in link_cols:
            clear_output(wait=True)
            other_link_cols = [c for c in link_cols if c != target]

            link_df = df[link_cols]
            screened = screen_predictors(link_df, target,
                                         method=nl_method,
                                         threshold=nl_threshold)
            if not screened:
                screened = other_link_cols

            base_r2 = quick_r2(df, target, screened)
            log.write(f"\n[{target}] quick R² = {base_r2:7.5f}\n")
            print(f"[{target}] quick R² = {base_r2:7.5f}   "
                  f"{'(skip)' if base_r2 < r2_gate else ''}")

            if base_r2 < r2_gate:
                log.write(f"  ↳ below R² gate ({r2_gate}); skipped.\n")
                continue

            # 3) PySR
            safe_names = ["sigma" if v == "S" else v for v in screened]
            X, y = df[screened].values, df[target].values

            model = PySRRegressor(
                niterations=pysr_iterations,
                binary_operators=["+", "-", "*", "/"],
                model_selection="best",
                maxsize=15,
                constraints={"sigma": 0},  # ← σ must occur
                complexity_of_constants=3,
                progress=False,
            ).fit(X, y, variable_names=safe_names)

            best = model.get_best()
            loss = float(best["loss"])
            r2   = r2_score(y, model.predict(X))

            log.write(f"  PySR best  R²={r2:.6f}  loss={loss:.2e}\n")
            log.write(f"  eqn:  {best['equation']}\n")

            if r2 >= r2_gate and loss <= pysr_loss_gate:
                accepted.append({"target": target,
                                 "equation": best['equation'],
                                 "loss": loss,
                                 "R2": r2,
                                 "predictors": ",".join(screened)})
                log.write("  ✅ accepted\n")
            else:
                log.write("  ❌ rejected (did not meet criteria)\n")

    if accepted:
        pd.DataFrame(accepted).to_csv(csv_out, index=False)
        print(f"\nSaved {len(accepted)} accepted equations → {csv_out}")
    else:
        print("\nNo equations met the acceptance criteria.")

    print(f"Full log written to {txt_out}")



def main():
    x = [f"E_{i}" for i in range(9)]
    y = [f"D_{i}" for i in range(9)]
    perso = ["D_0", "D_1", "D_8", "D_2", "D_4", "D_7", "L", "M", "LM"]
    df = load_csv_and_prune("sigma_mix.csv",
                            to_drop=y + perso)


    do_clustering = False
    if do_clustering:
        feature_cols = [c for c in df.columns if c.startswith(("E_","D_","S"))]
        dfc = cluster_rows(df, feature_cols, n_clusters=6)
        for cid, dfn in dfc.groupby("cluster_id"):
            print(f"\n=== cluster {cid}  (n={len(dfn)}) ===")
            run_guided_pipeline(dfn.reset_index(drop=True),
                                csv_out=f"eqns_cluster{cid}.csv",
                                txt_out=f"log_cluster{cid}.txt")
    else:
        run_guided_pipeline(df)

if __name__ == "__main__":
    main()
