# hugging-face-co2-multivariate-analysis
# Análisis de la Huella de Carbono en Modelos de Machine Learning

## 🎯 Objetivo del Proyecto

Este repositorio contiene un análisis estadístico multivariado sobre los factores que impactan la huella de carbono (CO2e) durante el entrenamiento de modelos de Machine Learning. El estudio se basa en un conjunto de datos extraído del popular repositorio **Hugging Face Hub**, con el fin de identificar patrones y relaciones clave entre las características de un modelo y su impacto ambiental.

## 🛠️ Metodología y Técnicas

El análisis explora la relación entre las emisiones de CO2 y diversas variables como el tamaño del modelo, el tamaño del dataset y las métricas de rendimiento. Se utilizaron las siguientes técnicas:

**Preprocesamiento y Limpieza de Datos:** Scripts en Python (usando Pandas) para transformar el dataset crudo en un formato numérico y analizable.

**Análisis de Correlación:** Cálculo y visualización de la matriz de correlación para identificar las relaciones lineales más fuertes entre las variables.

**Análisis de Componentes Principales (ACP):** Técnica de reducción de dimensionalidad para encontrar las principales fuentes de varianza en los datos.

**Análisis Factorial (AF):** Método para descubrir factores latentes o constructos subyacentes que explican los patrones de correlación.

## 📂 Contenido del Repositorio

**/data**: Contiene el dataset original (HFCO2.csv), el archivo preprocesado y los resultados de las matrices (correlation_matrix.csv).

**/scripts**: Incluye los scripts de Python utilizados para la limpieza de datos y la ejecución de los análisis estadísticos.

**/results**: Almacena las visualizaciones generadas, como el mapa de calor de correlaciones.

README.md: Esta descripción del proyecto.`

## 結論 Conclusiones Principales

El análisis revela una fuerte correlación positiva entre las emisiones de CO2 y variables como el **tamaño del modelo** y el **tamaño del dataset**, confirmando que los modelos más grandes y complejos tienden a tener un mayor impacto ambiental.