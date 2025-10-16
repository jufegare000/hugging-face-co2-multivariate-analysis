# -*- coding: utf-8 -*-
"""
Clustering jerárquico para modelos de IA (balanceado y robusto)
- Preprocesamiento robusto: log1p + winsorización + RobustScaler
- Comparación de distancias/aglomeraciones (Euclidiana + Mahalanobis; Ward sólo con Euclidiana)
- Corte optimizado con score compuesto: (1-α)*silhouette + α*balance (evita clúster diminutos)
- Métricas: cophenético, silhouette, ARI con is_fair (etiqueta externa), pureza/V de Cramér/chi2/Fisher
- Exporta dendrogramas, resúmenes, perfiles, centroides, ranking, distribuciones y dataset con clúster
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.covariance import LedoitWolf
from sklearn.metrics import silhouette_score, adjusted_rand_score

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
from scipy.stats import chi2_contingency, fisher_exact
import matplotlib.pyplot as plt

# ------------------
# Configuración I/O
# ------------------
CSV_PATH = "../../assets/huggingface_with_fairness.csv"
OUTPUT_DIR = "../../assets/cluster_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------
# Features y etiquetas externas
# ------------------
FEATURE_COLS = [
    "performance_score",
    "co2_eq_emissions",
    "likes",
    "downloads",
    "size",
]

EXTERNAL_LABELS = {
    "model_type": "model_type",
    "clasification_fairness": "clasification_fairness",
    "is_fair": "is_fair",  # sólo para evaluación/interpretación
}

# ------------------
# Preprocesamiento robusto (ajusta a tus datos)
# ------------------
LOG1P_COLS = ["downloads", "likes", "size", "co2_eq_emissions"]  # variables con cola larga
WINSOR_Q = 0.995                 # cap en percentil 99.5 (ajusta 0.990–0.999)
USE_ROBUST_SCALER = True         # RobustScaler (mediana/IQR) en lugar de StandardScaler

# ------------------
# Parámetros de corte y balance
# ------------------
K_TARGET: Optional[int] = None           # fija k si lo quieres forzar (p. ej., 3 ó 4). Si None, optimiza.
DIST_THRESHOLD: Optional[float] = None   # o fija un umbral de distancia. Si None, optimiza.
K_RANGE = range(3, 7)                     # arranca en 3 para evitar 1 vs. resto; ajusta según tu caso
MIN_CLUSTER_PROP = 0.05                   # tamaño mínimo por clúster (5% del total)
BALANCE_ALPHA = 0.40                      # peso del balance en el score compuesto (0..1)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ------------------
# Configuraciones a comparar
# ------------------
@dataclass
class Config:
    name: str
    distance: str  # 'euclidean' | 'mahalanobis'
    linkage: str   # 'ward' | 'average' | 'complete' | 'single'

CONFIGS: List[Config] = [
    Config("Ward-Euclid", "euclidean", "ward"),
    Config("Average-Euclid", "euclidean", "average"),
    Config("Complete-Euclid", "euclidean", "complete"),
    Config("Single-Euclid", "euclidean", "single"),
    Config("Average-Mahalanobis", "mahalanobis", "average"),
    Config("Complete-Mahalanobis", "mahalanobis", "complete"),
    Config("Single-Mahalanobis", "mahalanobis", "single"),
]

# ------------------
# Utilidades
# ------------------
def _winsorize_series(s: pd.Series, q: float) -> pd.Series:
    lo, hi = s.quantile(1 - q), s.quantile(q)
    return s.clip(lower=lo, upper=hi)

def _winsorize_df(df: pd.DataFrame, q: float, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = _winsorize_series(out[c], q)
    return out

def load_and_prepare(csv_path: str,
                     feature_cols: List[str],
                     external: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Etiquetas externas (se alinean luego con mask_valid)
    ext_df = pd.DataFrame()
    for k, v in external.items():
        if v in df.columns:
            ext_df[k] = df[v]

    # Selección y limpieza de features
    X = df[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Transformaciones robustas
    for c in LOG1P_COLS:
        if c in X.columns:
            X[c] = np.log1p(X[c])

    X = _winsorize_df(X, WINSOR_Q, feature_cols)

    # Drop NAs + alinear etiquetas externas
    mask_valid = ~X.isna().any(axis=1)
    dropped = int((~mask_valid).sum())
    X = X.loc[mask_valid]
    if not ext_df.empty:
        ext_df = ext_df.loc[mask_valid]

    print(f"Filas eliminadas por NA/inf: {dropped}")

    # Escalado
    scaler = RobustScaler() if USE_ROBUST_SCALER else StandardScaler()
    Z = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    return df, Z, ext_df

def mahalanobis_pdist(Z: pd.DataFrame) -> np.ndarray:
    X = Z.values
    lw = LedoitWolf().fit(X)
    VI = np.linalg.pinv(lw.covariance_)
    return pdist(X, metric="mahalanobis", VI=VI)

def compute_linkage(Z: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    X = Z.values
    if cfg.linkage == "ward":
        L = linkage(X, method="ward")
        coph_dists = pdist(X, metric="euclidean")
    else:
        if cfg.distance == "euclidean":
            D = pdist(X, metric="euclidean")
        elif cfg.distance == "mahalanobis":
            D = mahalanobis_pdist(Z)
        else:
            raise ValueError("Distancia no soportada")
        L = linkage(D, method=cfg.linkage)
        coph_dists = D
    return L, coph_dists

def threshold_for_k(L: np.ndarray, k: int) -> float:
    heights = L[:, 2]
    n_merges = len(heights)
    idx = n_merges - (k - 1)
    low = heights[idx - 1] - 1e-9
    high = heights[idx] + 1e-9 if idx < n_merges else heights[idx - 1] + 1.0
    return 0.5 * (low + high)

# -------- balance helpers --------
def _cluster_balance_score(labels: np.ndarray) -> float:
    """Score en [0,1] ~ (1 - Herfindahl) normalizado: 1=perfectamente balanceado."""
    counts = np.bincount(labels)[1:]
    if len(counts) < 2 or counts.sum() == 0:
        return 0.0
    p = counts / counts.sum()
    diversity = 1.0 - np.sum(p ** 2)
    k = len(counts)
    denom = 1.0 - 1.0 / k
    return float(diversity / denom) if denom > 0 else 0.0

def _min_cluster_ok(labels: np.ndarray, min_prop: float) -> bool:
    counts = np.bincount(labels)[1:]
    if counts.sum() == 0:
        return False
    return (counts / counts.sum()).min() >= min_prop
# ---------------------------------

def cut_tree(L: np.ndarray,
             k_target: Optional[int] = None,
             dist_threshold: Optional[float] = None,
             k_range = range(2, 9),
             X_for_score: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int, Optional[float], float]:
    """Score compuesto: (1-α)*silhouette + α*balance. Evita particiones con clúster < MIN_CLUSTER_PROP."""
    if dist_threshold is not None:
        labels = fcluster(L, t=dist_threshold, criterion="distance")
        k = len(np.unique(labels))
        sil = silhouette_score(X_for_score, labels, metric="euclidean") if X_for_score is not None and k > 1 else np.nan
        return labels, k, dist_threshold, sil

    if k_target is not None:
        labels = fcluster(L, t=k_target, criterion="maxclust")
        sil = silhouette_score(X_for_score, labels, metric="euclidean") if X_for_score is not None and k_target > 1 else np.nan
        return labels, k_target, None, sil

    best = (-np.inf, None, None, None)       # score, k, labels, sil
    best_relaxed = (-np.inf, None, None, None)

    for k in k_range:
        labels = fcluster(L, t=k, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            continue
        sil = silhouette_score(X_for_score, labels, metric="euclidean") if X_for_score is not None else np.nan
        if np.isnan(sil):
            continue
        bal = _cluster_balance_score(labels)
        score = (1 - BALANCE_ALPHA) * sil + BALANCE_ALPHA * bal

        if _min_cluster_ok(labels, MIN_CLUSTER_PROP):
            if score > best[0]:
                best = (score, k, labels, sil)
        if score > best_relaxed[0]:
            best_relaxed = (score, k, labels, sil)

    chosen = best if best[1] is not None else best_relaxed
    if chosen[1] is None:
        labels = fcluster(L, t=2, criterion="maxclust")
        sil = silhouette_score(X_for_score, labels, metric="euclidean") if X_for_score is not None else np.nan
        return labels, 2, None, sil

    return chosen[2], chosen[1], None, chosen[3]

def summarize_external(labels: np.ndarray, ext_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out = {}
    cl = pd.Series(labels, name="cluster")
    for name, col in ext_df.items():
        try:
            out[name] = pd.crosstab(cl, col)
        except Exception:
            pass
    return out

def get_ari(ext_df: pd.DataFrame, Z: pd.DataFrame, labels: np.ndarray) -> Tuple[float, str]:
    if "is_fair" not in ext_df.columns:
        return np.nan, "no_is_fair"
    y = ext_df["is_fair"].dropna()
    inter = np.intersect1d(y.index, Z.index)
    if len(inter) == 0:
        return np.nan, "sin_interseccion"
    if y.loc[inter].nunique() < 2:
        return np.nan, "una_sola_clase"
    ari = adjusted_rand_score(y.loc[inter].astype(bool), pd.Series(labels, index=Z.index).loc[inter])
    return float(ari), "ok"

def plot_dendrogram(cfg: Config, L: np.ndarray) -> str:
    plt.figure(figsize=(9, 5))
    dendrogram(L, no_labels=True)
    plt.title(f"Dendrograma – {cfg.name}")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f"dendrogram_{cfg.name.replace(' ', '_')}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path

def save_external_tables(ext_tabs: Dict[str, pd.DataFrame], cfg: Config) -> Dict[str, str]:
    paths = {}
    for name, tab in ext_tabs.items():
        p = os.path.join(OUTPUT_DIR, f"tabla_{cfg.name.replace(' ', '_')}_{name}.csv")
        tab.to_csv(p)
        paths[name] = p
    return paths

def _save_table(df: pd.DataFrame, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=True)
    return path

def compute_profiles(df: pd.DataFrame, Z: pd.DataFrame, labels: np.ndarray, ext_df: pd.DataFrame):
    labels_s = pd.Series(labels, index=Z.index, name="cluster")

    perfil_raw = (pd.concat([df[FEATURE_COLS], labels_s], axis=1)
                  .groupby("cluster")[FEATURE_COLS]
                  .agg(["count", "mean", "median"]))
    _save_table(perfil_raw, "perfil_unidades_originales.csv")

    perfil_z = (pd.concat([Z, labels_s], axis=1)
                .groupby("cluster")[Z.columns]
                .mean().round(2))
    _save_table(perfil_z, "centroides_zscores.csv")

    rank_features = perfil_z.abs().max().sort_values(ascending=False).to_frame(name="max_abs_z")
    _save_table(rank_features, "ranking_variables_max_abs_z.csv")

    for col in ["model_type", "clasification_fairness", "is_fair"]:
        if col in ext_df.columns:
            dist = pd.crosstab(labels_s, ext_df[col], normalize="index").round(3)
            _save_table(dist, f"distrib_{col}_por_cluster_proporciones.csv")

    df_out = df.copy()
    df_out.loc[labels_s.index, "cluster_hc"] = labels_s.values
    df_out.to_csv(os.path.join(OUTPUT_DIR, "dataset_con_clusters.csv"), index=False)

# ------------------
# Métricas de asociación para is_fair (conteos)
# ------------------
def _fairness_stats_from_tab(tab: pd.DataFrame) -> dict:
    out = {
        "fair_pureza_media": np.nan,
        "fair_pureza_ponderada": np.nan,
        "fair_chi2": np.nan,
        "fair_pval_chi2": np.nan,
        "fair_v_cramer": np.nan,
    }
    if tab is None or tab.empty:
        return out

    counts = tab.copy()
    row_sums = counts.sum(axis=1).replace(0, np.nan)
    prop = counts.div(row_sums, axis=0)
    purezas = prop.max(axis=1)
    out["fair_pureza_media"] = float(purezas.mean()) if purezas.notna().any() else np.nan
    weights = row_sums / row_sums.sum() if row_sums.sum() > 0 else None
    out["fair_pureza_ponderada"] = float((purezas * weights).sum()) if weights is not None else np.nan

    if counts.shape[0] >= 2 and counts.shape[1] >= 2:
        chi2, pval, dof, _ = chi2_contingency(counts.values)
        out["fair_chi2"] = float(chi2)
        out["fair_pval_chi2"] = float(pval)
        n = counts.values.sum()
        out["fair_v_cramer"] = float(np.sqrt(chi2 / (n * (min(counts.shape) - 1)))) if n > 0 else np.nan

    return out

# ------------------
# Pipeline principal
# ------------------
def run_pipeline(csv_path: str = CSV_PATH, top_n: int = 10):
    df, Z, ext_df = load_and_prepare(csv_path, FEATURE_COLS, EXTERNAL_LABELS)
    results = []

    for cfg in CONFIGS:
        try:
            L, coph_d = compute_linkage(Z, cfg)
        except Exception as e:
            print(f"[ERROR] {cfg.name}: {e}")
            continue

        coph_corr, _ = cophenet(L, coph_d)
        fig_path = plot_dendrogram(cfg, L)
        labels, k, thr, sil = cut_tree(L, K_TARGET, DIST_THRESHOLD, K_RANGE, Z.values)
        if thr is None:
            thr = threshold_for_k(L, k)

        ext_tabs = summarize_external(labels, ext_df)
        ari, ari_reason = get_ari(ext_df, Z, labels)
        fair_stats = _fairness_stats_from_tab(ext_tabs.get("is_fair")) if "is_fair" in ext_tabs else {
            "fair_pureza_media": np.nan,
            "fair_pureza_ponderada": np.nan,
            "fair_chi2": np.nan,
            "fair_pval_chi2": np.nan,
            "fair_v_cramer": np.nan,
        }
        ext_paths = save_external_tables(ext_tabs, cfg)

        row = {
            "config": cfg.name,
            "distance": cfg.distance,
            "linkage": cfg.linkage,
            "cophenetic": float(coph_corr),
            "k": int(k),
            "threshold": float(thr),
            "silhouette": float(sil),
            "ari_vs_is_fair": float(ari) if not np.isnan(ari) else np.nan,
            "ari_reason": ari_reason,
            "dendrogram_path": fig_path,
            "external_tables": ext_paths,
        }
        row.update(fair_stats)
        results.append(row)

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_sorted = res_df.sort_values(["silhouette", "cophenetic"], ascending=False)
        res_path = os.path.join(OUTPUT_DIR, "resumen_configuraciones.csv")
        res_sorted.to_csv(res_path, index=False)

        fair_cols = [
            "config","distance","linkage","k","threshold","silhouette","cophenetic",
            "ari_vs_is_fair","ari_reason","fair_pureza_media","fair_pureza_ponderada",
            "fair_pval_chi2","fair_v_cramer"
        ]
        fairness_path = os.path.join(OUTPUT_DIR, "resumen_fairness.csv")
        res_df[fair_cols].to_csv(fairness_path, index=False)

        top_path = os.path.join(OUTPUT_DIR, f"top_{top_n}_configuraciones.csv")
        res_sorted.head(top_n).to_csv(top_path, index=False)

        print("\nResumen guardado en:", res_path)
        print("Resumen de fairness en:", fairness_path)
        print(f"Top {top_n} configuraciones en:", top_path)
        print("\nTop configuraciones (por silueta, luego cophenético):\n")
        print(res_sorted.head(top_n))
        print("\nDendrogramas, perfiles y tablas en:", os.path.abspath(OUTPUT_DIR))
    else:
        print("No se pudo generar ningún resultado. Verifica columnas y datos.")

    # Perfiles del último clustering del loop
    try:
        compute_profiles(df, Z, labels, ext_df)
    except Exception:
        pass

if __name__ == "__main__":
    try:
        run_pipeline(CSV_PATH)
    except FileNotFoundError:
        print("[ERROR] No se encontró el CSV. Ajusta CSV_PATH antes de ejecutar.")
