# -*- coding: utf-8 -*-
"""


1) Carga y limpia la base.
2) Selecciona y estandariza variables numéricas de interés.
3) Compara múltiples configuraciones (distancia x aglomeración):
   - Distancias: euclidiana, cosine, mahalanobis (regularizada)
   - Aglomeración: ward (solo euclidiana), average, complete, single
4) Para cada configuración:
   - Calcula el coeficiente cophenético
   - Dibuja el dendrograma
   - Genera etiquetas de clúster por dos criterios de corte:
       a) número de clústeres (k) optimizado por silueta (o fijado por el usuario)
       b) umbral de distancia (t) si el usuario lo decide
   - Evalúa: silueta media, distribución de etiquetas externas (tipo_modelo,
     clasificacion_fairness, es_justo) y (opcional) ARI contra es_justo
5) Resumen comparativo de configuraciones.

"""

import warnings

from pandas.core.interchange.dataframe_protocol import DataFrame

warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.metrics import silhouette_score, adjusted_rand_score

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
import matplotlib.pyplot as plt

CSV_PATH = "../../assets/huggingface_with_fairness.csv"
OUTPUT_DIR = "../../assets/cluster_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "performance_score",
    "co2_eq_emissions",
    "likes",
    "downloads",
    "size",
]

EXTERNAL_LABELS = {
    "tipo_modelo": "tipo_modelo",
    "clasificacion_fairness": "clasificacion_fairness",
    "es_justo": "es_justo",
}

K_TARGET: Optional[int] = None
DIST_THRESHOLD: Optional[float] = None
K_RANGE = range(2, 9)

RANDOM_STATE = 42

@dataclass
class Config:
    name: str
    distance: str  # 'euclidean' | 'cosine' | 'mahalanobis'
    linkage: str  # 'ward' | 'average' | 'complete' | 'single'


CONFIGS: List[Config] = [
    Config("Ward-Euclid", "euclidean", "ward"),  # ward solo con euclidiana
    Config("Average-Euclid", "euclidean", "average"),
    Config("Complete-Euclid", "euclidean", "complete"),
    Config("Single-Euclid", "euclidean", "single"),
    Config("Average-Cosine", "cosine", "average"),
    Config("Complete-Cosine", "cosine", "complete"),
    Config("Single-Cosine", "cosine", "single"),
    Config("Average-Mahalanobis", "mahalanobis", "average"),
    Config("Complete-Mahalanobis", "mahalanobis", "complete"),
    Config("Single-Mahalanobis", "mahalanobis", "single"),
]

np.random.seed(RANDOM_STATE)


def load_and_prepare(csv_path: str,
                     feature_cols: List[str],
                     external: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_frame = pd.read_csv(csv_path)
    data_frame.columns = [c.strip() for c in data_frame.columns]

    external_columns_df = pd.DataFrame()
    for k, v in external.items():
        if v in data_frame.columns:
            external_columns_df[k] = data_frame[v]

    features = data_frame[feature_cols].copy()

    for c in features.columns:
        features[c] = pd.to_numeric(features[c], errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)

    mask_valid = ~features.isna().any(axis=1)
    dropped = int((~mask_valid).sum())
    features = features.loc[mask_valid]

    if not external_columns_df.empty:
        external_columns_df = external_columns_df.loc[mask_valid]

    print(f"Filas eliminadas por NA/inf: {dropped}")

    scaler = StandardScaler()
    Z_scores_df = pd.DataFrame(scaler.fit_transform(features), index=features.index, columns=features.columns)

    return data_frame, Z_scores_df, external_columns_df


def mahalanobis_pdist(Z_scores_df: pd.DataFrame) -> np.ndarray:
    features = Z_scores_df.values
    lw = LedoitWolf().fit(features)
    VI = np.linalg.pinv(lw.covariance_)
    d = pdist(features, metric="mahalanobis", VI=VI)
    return d


def compute_linkage(Z: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    X = Z.values
    if cfg.linkage == "ward":
        # Ward requiere datos (no distancias) y equivale a euclidiana
        L = linkage(X, method="ward")
        coph_dists = pdist(X, metric="euclidean")
    else:
        if cfg.distance == "euclidean":
            D = pdist(X, metric="euclidean")
        elif cfg.distance == "cosine":
            D = pdist(X, metric="cosine")
        elif cfg.distance == "mahalanobis":
            D = mahalanobis_pdist(Z)
        else:
            raise ValueError("Distancia no soportada")
        L = linkage(D, method=cfg.linkage)
        coph_dists = D
    return L, coph_dists


# =========================
# 5) Corte y evaluación
# =========================

def cut_tree(L: np.ndarray,
             k_target: Optional[int] = None,
             dist_threshold: Optional[float] = None,
             k_range=range(2, 9),
             X_for_score: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int, Optional[float], float]:
    """Devuelve labels, k elegido, threshold (si aplica) y silueta."""
    if dist_threshold is not None:
        labels = fcluster(L, t=dist_threshold, criterion="distance")
        k = len(np.unique(labels))
        sil = silhouette_score(X_for_score, labels, metric="euclidean") if X_for_score is not None and k > 1 else np.nan
        return labels, k, dist_threshold, sil

    if k_target is not None:
        labels = fcluster(L, t=k_target, criterion="maxclust")
        sil = silhouette_score(X_for_score, labels,
                               metric="euclidean") if X_for_score is not None and k_target > 1 else np.nan
        return labels, k_target, None, sil

    # Optimiza k por silueta en el rango
    best = (-np.inf, None, None)
    for k in k_range:
        labels = fcluster(L, t=k, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            continue
        sil = silhouette_score(X_for_score, labels, metric="euclidean") if X_for_score is not None else np.nan
        if np.isnan(sil):
            continue
        if sil > best[0]:
            best = (sil, k, labels)
    if best[1] is None:
        # Fallback: k=2
        labels = fcluster(L, t=2, criterion="maxclust")
        sil = silhouette_score(X_for_score, labels, metric="euclidean") if X_for_score is not None else np.nan
        return labels, 2, None, sil
    return best[2], best[1], None, best[0]


def summarize_external(labels: np.ndarray, ext_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out = {}
    cl = pd.Series(labels, name="cluster")
    for name, col in ext_df.items():
        try:
            tab = pd.crosstab(cl, col)
            out[name] = tab
        except Exception:
            pass
    return out



def run_pipeline(csv_path: str = CSV_PATH):
    df, Z, ext_df = load_and_prepare(csv_path, FEATURE_COLS, EXTERNAL_LABELS)
    results = []

    for cfg in CONFIGS:
        L, coph_d = get_linkage(cfg, Z)
        coph_corr, _ = cophenet(L, coph_d)
        fig_path = plot_dendogram(cfg, L)
        labels, k, _, sil = cut_tree(L, K_TARGET, DIST_THRESHOLD, K_RANGE, Z.values)
        ext_tabs = summarize_external(labels, ext_df)
        ari = get_ari(ext_df, Z, labels)
        ext_paths = save_external_tables(ext_tabs, cfg)
        append_resume_row(results, cfg, coph_corr, k, sil, ari, fig_path, ext_paths)

    # Ranking por silueta y cophenético
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(["silhouette", "cophenetic"], ascending=False)
        res_path = os.path.join(OUTPUT_DIR, "resumen_configuraciones.csv")
        res_df.to_csv(res_path, index=False)
        print("\nResumen guardado en:", res_path)
        print("\nTop configuraciones (por silueta, luego cophenético):\n")
        print(res_df.head(10))
        print("\nDendrogramas y tablas por configuración en:", os.path.abspath(OUTPUT_DIR))
    else:
        print("No se pudo generar ningún resultado. Verifica columnas y datos.")


def get_linkage(cfg: Config, Z: DataFrame) -> [np.ndarray, np.ndarray]:
    try:
        L, coph_d = compute_linkage(Z, cfg)
        return (L, coph_d)
    except Exception as e:
        print(f"[ERROR] {cfg.name}: {e}")


def plot_dendogram(cfg: Config, L: np.ndarray) -> str:
    plt.figure(figsize=(9, 5))
    dendrogram(L, no_labels=True)
    plt.title(f"Dendrograma – {cfg.name}")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f"dendrogram_{cfg.name.replace(' ', '_')}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def get_ari(ext_df: DataFrame, Z: DataFrame, labels: np.ndarray) -> float:
    ari = np.nan
    if "es_justo" in ext_df.columns:
        y = ext_df["es_justo"].dropna()
        inter = np.intersect1d(y.index, Z.index)
        if len(np.unique(y.loc[inter])) > 1:
            ari = adjusted_rand_score(y.loc[inter].astype(int), pd.Series(labels, index=Z.index).loc[inter])
    return ari


def save_external_tables(ext_tabs: dict[str, DataFrame], cfg: Config) -> dict[str, str]:
    ext_paths = {}
    for name, tab in ext_tabs.items():
        p = os.path.join(OUTPUT_DIR, f"tabla_{cfg.name.replace(' ', '_')}_{name}.csv")
        tab.to_csv(p)
        ext_paths[name] = p
    return ext_paths


def append_resume_row(results: list, cfg: Config, coph_corr: np.ndarray, k: int, sil: float, ari: float,
                      fig_path: str, ext_paths: dict[str, str]):
    results.append({
        "config": cfg.name,
        "distance": cfg.distance,
        "linkage": cfg.linkage,
        "cophenetic": coph_corr,
        "k": k,
        "silhouette": sil,
        "ari_vs_es_justo": ari,
        "dendrogram_path": fig_path,
        "external_tables": ext_paths,
    })


if __name__ == "__main__":
    # Ejecuta el pipeline
    try:
        run_pipeline(CSV_PATH)
    except FileNotFoundError:
        print("[ERROR] No se encontró el CSV. Ajusta CSV_PATH antes de ejecutar.")
