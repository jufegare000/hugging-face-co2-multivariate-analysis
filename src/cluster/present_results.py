import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image

OUTPUT_DIR = "../../assets/cluster_output"
TOP_N_PLOT = 10

pd.set_option("display.max_rows", 50)
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 140)

def _hdr(text, level=3):
    display(Markdown("#"*level + " " + text))

def _read_csv_safe(fname, title=None, **kwargs):
    path = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(path):
        df = pd.read_csv(path, **kwargs)
        if title:
            _hdr(title)
        display(df)
        return df
    else:
        display(Markdown(f"> ⚠️ **No encontrado:** `{fname}`"))
        return None

def _plot_barh_from_df(df, x_col, y_col, title, top=None):
    if df is None or x_col not in df.columns or y_col not in df.columns:
        return
    data = df.copy()
    if top:
        data = data.sort_values(y_col, ascending=False).head(top)
    _hdr(title)
    plt.figure(figsize=(8, 4))
    plt.barh(data[x_col], data[y_col])
    plt.gca().invert_yaxis()
    plt.xlabel(y_col)
    plt.tight_layout()
    plt.show()

def _plot_scatter(df, x_col, y_col, title):
    if df is None or {x_col, y_col}.difference(df.columns):
        return
    _hdr(title)
    plt.figure(figsize=(5, 4))
    plt.scatter(df[x_col], df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.show()

def _display_pngs(pattern, title):
    paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, pattern)))
    _hdr(title)
    if not paths:
        display(Markdown(f"> ⚠️ No hay archivos que coincidan con `{pattern}`"))
        return
    for p in paths:
        display(Markdown(f"**{os.path.basename(p)}**"))
        display(Image(filename=p))


resumen_config = _read_csv_safe("resumen_configuraciones.csv", "Resumen de configuraciones")
resumen_fairness = _read_csv_safe("resumen_fairness.csv", "Resumen de fairness")
top_configs = _read_csv_safe("top_10_configuraciones.csv", "Top 10 configuraciones (si existe)")


perfil = _read_csv_safe("perfil_unidades_originales.csv", "Perfil (unidades originales)")
centroides = _read_csv_safe("centroides_zscores.csv", "Centroides en Z")
ranking = _read_csv_safe("ranking_variables_max_abs_z.csv", "Ranking variables (máx |z| por clúster)")

if ranking is not None and "max_abs_z" in ranking.columns:
    _hdr("Ranking ordenado (desc)")
    display(ranking.sort_values("max_abs_z", ascending=False))

_ = _read_csv_safe("distrib_model_type_por_cluster_proporciones.csv",
                   "Distribución por clúster – model_type (proporciones)", index_col=0)
_ = _read_csv_safe("distrib_clasification_fairness_por_cluster_proporciones.csv",
                   "Distribución por clúster – clasification_fairness (proporciones)", index_col=0)
dist_is_fair = _read_csv_safe("distrib_is_fair_por_cluster_proporciones.csv",
                              "Distribución por clúster – is_fair (proporciones)", index_col=0)


dataset = _read_csv_safe("dataset_con_clusters.csv", "Dataset con etiqueta de clúster (head)")
if dataset is not None:
    _hdr("Vista rápida del dataset")
    display(dataset.head(10))

# -----------------------------
# -----------------------------
_plot_barh_from_df(resumen_config, x_col="config", y_col="silhouette",
                   title=f"Silhouette por configuración (Top {TOP_N_PLOT})", top=TOP_N_PLOT)

_plot_scatter(resumen_config, "cophenetic", "silhouette", "Relación Cophenetic vs. Silhouette")

if dist_is_fair is not None:
    _hdr("Gráfica: Proporción is_fair por clúster")
    dist_plot = dist_is_fair.div(dist_is_fair.sum(axis=1), axis=0)
    ax = dist_plot.plot(kind="bar", stacked=True, figsize=(6, 4))
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proporción")
    plt.tight_layout()
    plt.show()

_display_pngs("dendrogram_*.png", "Dendrogramas generados")


tablas_ext = sorted(glob.glob(os.path.join(OUTPUT_DIR, "tabla_*_*.csv")))
_hdr("Tablas externas por configuración (conteos)")
if tablas_ext:
    for p in tablas_ext:
        display(Markdown(f"**{os.path.basename(p)}**"))
        display(pd.read_csv(p, index_col=0))
else:
    display(Markdown("> ⚠️ No se encontraron `tabla_*_*.csv`"))


_hdr("Sugerencias de exploración")
display(Markdown("""
1. Revisa **Top configuraciones** para elegir una candidata por *silhouette* y *cophenetic*.
2. Mira sus **tablas externas** (`tabla_*_*.csv`) para entender composición por etiquetas.
3. Interpreta **centroides** y **ranking** para drivers de cada clúster.
4. Cruza con **dataset_con_clusters.csv** para análisis por instancia (filtrar por cluster).
"""))

