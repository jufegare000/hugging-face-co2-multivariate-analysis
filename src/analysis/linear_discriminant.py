import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2, kurtosis, shapiro, anderson
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf


CSV_PATH = "../../assets/huggingface_with_fairness.csv"
OUTPUT_DIR = "../../assets/lda_diagnostics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variables numéricas a evaluar (mismas del análisis previo)
FEATURE_COLS = [
    "performance_score", "co2_eq_emissions", "likes", "downloads", "size"
]

# Etiqueta a evaluar (elige una: "is_fair", "clasification_fairness", "model_type")
TARGET_LABEL = "is_fair"

# Transformaciones opcionales (True para mejorar simetría si tus datos son muy sesgados)
APPLY_LOG1P = True
LOG1P_COLS = ["co2_eq_emissions", "likes", "downloads", "size"]

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ------------------------
# UTILIDADES NUMÉRICAS
# ------------------------
def _to_numeric(df, cols):
    X = df[cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X

def _mardia_skew_kurt(X: np.ndarray):
    """
    Mardia's multivariate skewness and kurtosis.
    Referencias:
    - Mardia, K. V. (1970, 1974). Measures of multivariate skewness and kurtosis.
    Aproximaciones:
    - b1,p (skewness) ~ Chi-square con df = p(p+1)(p+2)/6
    - b2,p (kurtosis) ~ Normal con media = p(p+2), var = 8p(p+2)/n
    """
    n, p = X.shape
    # Estimación robusta de covarianza para invertir con estabilidad
    lw = LedoitWolf().fit(X)
    S_inv = np.linalg.pinv(lw.covariance_)
    Xm = X - X.mean(axis=0, keepdims=True)

    # Distancias de Mahalanobis
    D = np.einsum('ij,jk,ik->i', Xm, S_inv, Xm)  # n-vector

    # Mardia kurtosis:
    b2p = np.mean(D**2)
    mean_b2p = p*(p+2)
    var_b2p = (8*p*(p+2)) / n
    z_kurt = (b2p - mean_b2p) / np.sqrt(var_b2p) if var_b2p > 0 else np.nan
    pval_kurt = 2*(1 - chi2.cdf((z_kurt**2), df=1)) if np.isfinite(z_kurt) else np.nan  # aproximado via z^2 ~ chi2_1

    # Mardia skewness (requiere suma sobre pares):
    # Implementación vectorizada usando productos internos
    # b1p = (1/n^2) * sum_{i,j} ( (x_i^T S^{-1} x_j)^3 )
    # Calculamos A = Xm * S_inv^(1/2) (usando S_inv directamente con producto simétrico)
    # Más estable: usamos M = Xm @ S_inv @ Xm^T y elevamos al cubo elemento a elemento
    M = Xm @ S_inv @ Xm.T  # n x n
    b1p = np.mean(M**3)
    df_skew = p*(p+1)*(p+2)/6
    # Aproximación: n*b1p/6 ~ Chi2_df
    chi_skew = n * b1p / 6.0
    pval_skew = 1 - chi2.cdf(chi_skew, df=int(df_skew))

    out = {
        "n": n, "p": p,
        "mardia_b1p_skew": float(b1p),
        "mardia_chi_skew": float(chi_skew),
        "mardia_df_skew": int(df_skew),
        "mardia_p_skew": float(pval_skew),

        "mardia_b2p_kurt": float(b2p),
        "mardia_z_kurt": float(z_kurt),
        "mardia_p_kurt": float(pval_kurt),
        "mahalanobis_D": D.astype(float)  # para QQ
    }
    return out

def _qq_mahalanobis(D: np.ndarray, p: int, title: str, path: str):
    """
    QQ plot: D ~ chi2_p  (si normalidad multivariante).
    """
    D_sorted = np.sort(D)
    n = len(D_sorted)
    probs = (np.arange(1, n+1) - 0.5) / n
    chi_theor = chi2.ppf(probs, df=p)

    plt.figure(figsize=(5,5))
    plt.scatter(chi_theor, D_sorted, s=12)
    maxv = max(chi_theor.max(), D_sorted.max())
    plt.plot([0, maxv], [0, maxv])
    plt.xlabel(r"Chi-square$_{p}$ theoretical quantiles")
    plt.ylabel("Empirical Mahalanobis distances")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def _box_m_test(X: np.ndarray, y: np.ndarray):
    """
    Box's M test for equality of covariance matrices across groups.
    Aproximación clásica (puede ser sensible con n pequeños).
    Retorna: M, df, p-value (aprox. chi2)
    """
    # Requiere al menos 2 clases
    classes, counts = np.unique(y, return_counts=True)
    g = len(classes)
    p = X.shape[1]
    if g < 2:
        return np.nan, np.nan, np.nan

    # Covarianzas por grupo y pooled
    covs = []
    sum_cov = np.zeros((p,p))
    N = 0
    for cls, nk in zip(classes, counts):
        Xk = X[y == cls]
        Sk = np.cov(Xk, rowvar=False, bias=False)
        covs.append(Sk)
        sum_cov += (nk - 1) * Sk
        N += nk
    Spooled = sum_cov / (N - g)

    # Estadístico M (Box, 1949): M = (N - g)*ln|Spooled| - sum_k (n_k - 1)*ln|S_k|
    sign_pooled, logdet_pooled = np.linalg.slogdet(Spooled)
    if sign_pooled <= 0:
        return np.nan, np.nan, np.nan

    M = (N - g) * logdet_pooled
    for Sk, nk in zip(covs, counts):
        sign_k, logdet_k = np.linalg.slogdet(Sk)
        if sign_k <= 0:
            return np.nan, np.nan, np.nan
        M -= (nk - 1) * logdet_k

    # Corrección (aprox) para que M ~ chi2_df
    # df = (g - 1)*p*(p + 1)/2
    df = int((g - 1) * p * (p + 1) / 2)
    # Factor de corrección c (muy usado en textos aplicados)
    c = 1 - ( (2*p**2 + 3*p - 1) / (6*(sum(counts) - g)) ) * ( sum(1/(nk - 1) for nk in counts) - 1/(sum(counts) - g) )
    M_corr = M * c
    pval = 1 - chi2.cdf(M_corr, df=df)
    return float(M_corr), df, float(pval)


# ------------------------
# CARGA Y PREPROCESAMIENTO
# ------------------------
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

if TARGET_LABEL not in df.columns:
    raise ValueError(f"TARGET_LABEL '{TARGET_LABEL}' no está en el CSV.")

X = _to_numeric(df, FEATURE_COLS)

# Transformaciones opcionales
if APPLY_LOG1P:
    for c in LOG1P_COLS:
        if c in X.columns:
            X[c] = np.log1p(X[c])

# Filtrado de NA/inf y alineación con y
mask_valid = ~X.isna().any(axis=1)
X = X.loc[mask_valid]
y = df.loc[mask_valid, TARGET_LABEL]

# Si y es booleana de texto/num, homogenizar
if y.dtype == bool:
    y_clean = y.values
else:
    # intenta convertir 'True'/'False'/'0'/'1'
    y_clean = y.copy()
    if y_clean.dtype == object:
        y_clean = y_clean.str.strip().replace({"TRUE": True, "True": True, "true": True,
                                               "FALSE": False, "False": False, "false": False})
    # Si sigue siendo object, mantenlo categórico
    if y_clean.dtype == object:
        y_clean = y_clean.astype("category").values
    else:
        y_clean = y_clean.values

# Estandarización (z-score) para diagnóstico multivariante
scaler = StandardScaler()
Z = scaler.fit_transform(X.values)

n, p = Z.shape
classes, counts = np.unique(y_clean, return_counts=True)
print(f"n={n}, p={p}, clases={list(classes)}, tamaños={list(counts)}")

# ------------------------
# DIAGNÓSTICOS
# ------------------------

# 1) Mardia global (sobre todas las observaciones, sin estratificar)
mardia_all = _mardia_skew_kurt(Z)
print("\n[Mardia - Global]")
for k, v in mardia_all.items():
    if k == "mahalanobis_D": continue
    print(f"{k}: {v}")

# QQ global
qq_path = os.path.join(OUTPUT_DIR, "qq_mahalanobis_global.png")
_qq_mahalanobis(mardia_all["mahalanobis_D"], p, "Mahalanobis QQ (global)", qq_path)
print("QQ global guardado en:", qq_path)

# 2) Mardia por clase (si hay >= 2 clases y tamaños suficientes)
mardia_by_class = {}
if len(classes) >= 2:
    for cls in classes:
        Zk = Z[y_clean == cls]
        if Zk.shape[0] > p + 5:  # umbral mínimo grosero
            mardia_by_class[cls] = _mardia_skew_kurt(Zk)
            # QQ por clase
            qqk = os.path.join(OUTPUT_DIR, f"qq_mahalanobis_class_{cls}.png")
            _qq_mahalanobis(mardia_by_class[cls]["mahalanobis_D"], p,
                            f"Mahalanobis QQ (class={cls})", qqk)
            print(f"QQ class {cls} guardado en:", qqk)

# 3) Box's M (igualdad de covarianzas entre clases)
M_corr, df_box, p_box = _box_m_test(Z, np.array(y_clean))
print(f"\n[Box's M] M_corr={M_corr:.3f}, df={df_box}, p-value={p_box}")

# 4) Reglas de tamaño: n_k vs p
size_checks = {cls: {"n_k": int(nk), "p": p, "rule_5p": bool(nk >= 5*p), "rule_10p": bool(nk >= 10*p)}
               for cls, nk in zip(classes, counts)}
print("\n[Tamaños por clase y reglas n_k vs p]")
for cls, info in size_checks.items():
    print(f"Clase {cls}: {info}")

# ------------------------
# VEREDICTO AUTOMÁTICO
# ------------------------
def verdict(mardia_global, mardia_by_class, p_box, size_checks):
    notes = []
    ok_normal_global = (mardia_global["mardia_p_skew"] > 0.05) and (abs(mardia_global["mardia_p_kurt"]) > 0.05)
    # En práctica solemos exigir evidencia por clase:
    ok_normal_by_class = True
    if len(mardia_by_class) > 0:
        for cls, res in mardia_by_class.items():
            ok_sk = (res["mardia_p_skew"] > 0.05)
            ok_ku = (abs(res["mardia_p_kurt"]) > 0.05)
            if not (ok_sk and ok_ku):
                ok_normal_by_class = False
                notes.append(f"Normalidad multivariante NO soportada en clase {cls} (Mardia).")
    else:
        ok_normal_by_class = False
        notes.append("No se pudo evaluar Mardia por clase (tamaño insuficiente o 1 sola clase).")

    # Box's M (p>0.05 sugiere covarianzas similares)
    ok_cov_equal = (p_box is not None) and (p_box > 0.05)
    if not ok_cov_equal:
        notes.append("Box's M sugiere covarianzas diferentes entre clases (p<=0.05).")

    # Regla de tamaño 5p
    ok_sizes = all(info["rule_5p"] for info in size_checks.values())
    if not ok_sizes:
        notes.append("Alguna clase no cumple n_k >= 5*p (tamaño insuficiente para LDA estable).")

    # Decisión
    if ok_normal_by_class and ok_cov_equal and ok_sizes:
        dec = "LDA aprobada (supuestos razonables)."
        alt = "—"
    elif ok_normal_by_class and (not ok_cov_equal) and ok_sizes:
        dec = "LDA NO recomendada por covarianzas desiguales."
        alt = "Sugerido: QDA o LDA regularizada (shrinkage)."
    else:
        dec = "LDA NO recomendable (violación de normalidad y/o tamaños)."
        alt = "Sugerido: LDA regularizada, QDA si covarianzas difieren, o modelos no paramétricos (logística, árboles, SVM)."

    return dec, alt, notes

decision, alternative, notes = verdict(mardia_all, mardia_by_class, p_box, size_checks)
print("\n[VEREDICTO]")
print("Decisión:", decision)
print("Alternativa:", alternative)
print("Notas:")
for s in notes:
    print(" -", s)

# Guardar resumen en CSV
summary = {
    "n": n, "p": p,
    "classes": [str(list(classes))],
    "sizes": [str(list(counts))],
    "mardia_p_skew_global": mardia_all["mardia_p_skew"],
    "mardia_p_kurt_global": mardia_all["mardia_p_kurt"],
    "box_m_pvalue": p_box,
    "decision": decision,
    "alternative": alternative
}
pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_DIR, "diagnostico_lda_resumen.csv"), index=False)
print("\nResumen guardado en:", os.path.join(OUTPUT_DIR, "diagnostico_lda_resumen.csv"))
print("Gráficos QQ en:", OUTPUT_DIR)
