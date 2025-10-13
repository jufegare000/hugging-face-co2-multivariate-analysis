import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.cluster import KMeans

CSV_PATH = "../../assets/huggingface_with_fairness.csv"
OUTPUT_DIR = "../../assets/kmeans_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "performance_score", "co2_eq_emissions", "likes", "downloads", "size"
]
EXTERNAL_LABELS = {
    "model_type": "model_type",
    "clasification_fairness": "clasification_fairness",
    "is_fair": "is_fair",
}

LOG1P_COLS = ["downloads", "likes", "size", "co2_eq_emissions"]
WINSOR_Q = 0.995
USE_ROBUST_SCALER = True

K_GRID = list(range(2, 11))
RANDOM_STATE = 42
N_INIT = "auto"


def _winsorize_series(s: pd.Series, q: float) -> pd.Series:
    lo, hi = s.quantile(1 - q), s.quantile(q)
    return s.clip(lower=lo, upper=hi)


def _winsorize_df(df: pd.DataFrame, q: float, cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = _winsorize_series(out[c], q)
    return out


def _save_csv(df: pd.DataFrame, name: str) -> str:
    p = os.path.join(OUTPUT_DIR, name)
    df.to_csv(p, index=True)
    return p


df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

ext_df = pd.DataFrame()
for k, v in EXTERNAL_LABELS.items():
    if v in df.columns:
        ext_df[k] = df[v]

X = df[FEATURE_COLS].copy()
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

for c in LOG1P_COLS:
    if c in X.columns:
        X[c] = np.log1p(X[c])

X = _winsorize_df(X, WINSOR_Q, FEATURE_COLS)

mask_valid = ~X.isna().any(axis=1)
idx_valid = X.index[mask_valid]

X = X.loc[idx_valid]
ext_df = ext_df.loc[idx_valid] if not ext_df.empty else ext_df
df_valid = df.loc[idx_valid]

scaler = RobustScaler() if USE_ROBUST_SCALER else StandardScaler()
Z = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

rows = []
labels_by_k = {}

for k in K_GRID:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    lbl = km.fit_predict(Z.values)
    labels_by_k[k] = lbl

    sil = silhouette_score(Z.values, lbl, metric="euclidean") if k > 1 else np.nan
    ch = calinski_harabasz_score(Z.values, lbl) if k > 1 else np.nan
    db = davies_bouldin_score(Z.values, lbl) if k > 1 else np.nan
    inertia = float(km.inertia_)

    ari = np.nan
    if "is_fair" in ext_df.columns:
        y = ext_df["is_fair"].dropna()
        inter = np.intersect1d(y.index, Z.index)
        if len(inter) > 0 and y.loc[inter].nunique() > 1:
            ari = adjusted_rand_score(y.loc[inter].astype(bool), pd.Series(lbl, index=Z.index).loc[inter])

    rows.append({
        "k": k,
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "inertia": inertia,
        "ari_vs_is_fair": ari
    })

res_k = pd.DataFrame(rows).sort_values("k")
_save_csv(res_k, "kmeans_resumen_por_k.csv")
print(res_k)

best = (
    res_k.sort_values(
        by=["silhouette", "calinski_harabasz", "davies_bouldin"],
        ascending=[False, False, True]
    ).iloc[0]
)
BEST_K = int(best["k"])
print(f"\nMejor k por criterios internos => k={BEST_K}")

best_labels = labels_by_k[BEST_K]
labels_s = pd.Series(best_labels, index=Z.index, name="cluster_km")

perfil_raw = (pd.concat([X, labels_s], axis=1)
              .groupby("cluster_km")[FEATURE_COLS]
              .agg(["count", "mean", "median"]))
_save_csv(perfil_raw, f"perfil_unidades_originales_k{BEST_K}.csv")

centroides_z = (pd.concat([Z, labels_s], axis=1)
                .groupby("cluster_km")[Z.columns]
                .mean().round(3))
_save_csv(centroides_z, f"centroides_zscores_k{BEST_K}.csv")

ranking_absz = centroides_z.abs().max().sort_values(ascending=False).to_frame("max_abs_z")
_save_csv(ranking_absz, f"ranking_max_abs_z_k{BEST_K}.csv")

for col in ["model_type", "clasification_fairness", "is_fair"]:
    if col in ext_df.columns:
        dist = pd.crosstab(labels_s, ext_df[col], normalize="index").round(3)
        _save_csv(dist, f"distrib_{col}_por_cluster_k{BEST_K}.csv")

df_out = df_valid.copy()
df_out.loc[labels_s.index, "cluster_km"] = labels_s.values
df_out.to_csv(os.path.join(OUTPUT_DIR, f"dataset_con_kmeans_k{BEST_K}.csv"), index=False)

plt.figure()
plt.plot(res_k["k"], res_k["inertia"], marker="o")
plt.xlabel("k");
plt.ylabel("Inertia");
plt.title("Elbow curve (inertia)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "elbow_inertia.png"), dpi=150)
plt.close()

plt.figure()
plt.plot(res_k["k"], res_k["silhouette"], marker="o")
plt.xlabel("k");
plt.ylabel("Silhouette");
plt.title("Silhouette vs k")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "silhouette_vs_k.png"), dpi=150)
plt.close()

plt.figure()
plt.plot(res_k["k"], res_k["calinski_harabasz"], marker="o")
plt.xlabel("k");
plt.ylabel("Calinski-Harabasz");
plt.title("CH vs k")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ch_vs_k.png"), dpi=150)
plt.close()

plt.figure()
plt.plot(res_k["k"], res_k["davies_bouldin"], marker="o")
plt.xlabel("k");
plt.ylabel("Davies-Bouldin");
plt.title("DBI vs k (menor es mejor)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dbi_vs_k.png"), dpi=150)
plt.close()

sizes = pd.Series(best_labels).value_counts().sort_index()
plt.figure()
plt.bar(sizes.index.astype(str), sizes.values)
plt.xlabel("Cluster");
plt.ylabel("n");
plt.title(f"Cluster sizes (k={BEST_K})")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"cluster_sizes_k{BEST_K}.png"), dpi=150)
plt.close()

print("\nArchivos guardados en:", os.path.abspath(OUTPUT_DIR))
