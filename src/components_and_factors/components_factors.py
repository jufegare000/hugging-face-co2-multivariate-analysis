import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns

class ComponentsAndFactors():

    def plot_dispersion_diagram(self, df_pca):
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111)
        scatter = ax.scatter(df_pca[0], df_pca[1], s=50)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Gráfico de Dispersión')
        fig.colorbar(scatter, ax=ax, label='')
        plt.show()

    def get_explained_matrix(self, pca):
        cum_var = pd.DataFrame(np.cumsum(pca.explained_variance_ratio_) * 100, columns=['Varianza Acumulada (%)'])
        exp_var = pd.DataFrame(pca.explained_variance_ratio_ * 100, columns=['Varianza Explicada (%)'])

        pc_names = [f'PC{i + 1}' for i in range(len(pca.explained_variance_ratio_))]

        final_df = pd.concat([exp_var, cum_var], axis=1)
        final_df.index = pc_names

        return final_df

    def get_data_frame_scaled(self, imputed_dataframe):
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(imputed_dataframe), columns=imputed_dataframe.columns)
        return df_scaled

    def get_eigenvaules_and_vectors_fa(self, df_scaled):
        corr_matrix = df_scaled.corr()
        eigenvalues_fa, eigenvectors_fa = np.linalg.eig(corr_matrix)
        eigenvalues_fa.sort()
        eigenvalues_fa = eigenvalues_fa[::-1]
        return (eigenvalues_fa, eigenvectors_fa)

    def get_loadings_fa(self, fa, df_scaled, n_factors):
        return pd.DataFrame(fa.components_.T, columns=[f'Factor {i+1}' for i in range(n_factors)], index=df_scaled.columns)

    def plot_sediment_diagram(self, eigenvalues_fa):
        scree_df = pd.DataFrame({
            'Factor': range(1, len(eigenvalues_fa) + 1),
            'Valor Propio': eigenvalues_fa
        })

        plt.figure(figsize=(5, 5))
        sns.lineplot(x='Factor', y='Valor Propio', data=scree_df, marker='o')

        # Añadimos una línea de referencia para el criterio de Kaiser (valor propio > 1)
        plt.axhline(y=1, color='r', linestyle='--', label='Criterio de Kaiser (Valor Propio > 1)')

        plt.title('Gráfico de Sedimento (Scree Plot)', fontsize=16)
        plt.xlabel('Número de Factores', fontsize=12)
        plt.ylabel('Valor Propio', fontsize=12)
        plt.xticks(np.arange(1, len(eigenvalues_fa) + 1))
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_biplot_pca(self, score, labels):
        coeff = np.transpose(score)
        xs = score[:, 0]
        ys = score[:, 1]
        n = coeff.shape[0]
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())
        plt.figure(figsize=(12, 6))
        plt.scatter(xs * scalex, ys * scaley, s=5)
        for i in range(n):
            plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
            if labels is None:
                plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='green', ha='center',
                         va='center')
            else:
                plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')

        plt.xlabel("PC{}".format(1))
        plt.ylabel("PC{}".format(2))
        plt.grid()