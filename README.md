# hugging-face-co2-multivariate-analysis
# Análisis multivariado de emisión de CO2 de modelos de IA

## 🎯 Objetivo del Proyecto

Este repositorio contiene un análisis estadístico multivariado sobre los factores que impactan la huella de carbono (CO2e) durante el entrenamiento de modelos de Machine Learning. El estudio se basa en un conjunto de datos extraído del popular repositorio **Hugging Face Hub**, con el fin de identificar patrones y relaciones clave entre las características de un modelo y su impacto ambiental.

## 🛠️ Metodología y Técnicas

El análisis explora la relación entre las emisiones de CO2 y diversas variables como el tamaño del modelo, el tamaño del dataset y las métricas de rendimiento. Se utilizaron las siguientes técnicas:

**Preprocesamiento y Limpieza de Datos:** Scripts en Python (usando Pandas y numpy) para transformar el dataset crudo en un formato numérico y analizable.

**Análisis de Correlación:** Cálculo y visualización de la matriz de correlación para identificar las relaciones lineales más fuertes entre las variables.

**Análisis de Componentes Principales (ACP):** Técnica de reducción de dimensionalidad para encontrar las principales fuentes de varianza en los datos.

**Análisis Factorial (AF):** Método para descubrir factores latentes o constructos subyacentes que explican los patrones de correlación.

## 📂 Contenido del Repositorio

**/assets**: Contiene el dataset original (HFCO2.csv), el archivo preprocesado y los resultados de las matrices.

**/notebooks**: Incluye los notebooks de jupyter en Python utilizados para el análisis estadísticos.

**/src**: Almacena los scripts útiles necesarios para la ejecución del análisis estadístico, para eviar colocar funcciones que sólo agregan ruido al análisis.

README.md: Esta descripción del proyecto.`

## Dependencias
Las dependencias apra porder correr el proyecto se encuentran en [requirements.txt](requirements.txt) para correr el proyecto 
es necesario usar la versión de **Python 3.11.13**  **jupyter notebook 7.4.5**

para instalar las dependencias se usa el comando: 
```bash 
pip install -r requirements.txt
```

## Reporte del proyecto
El resultado del análisis puede verse en la siguiente ruta:
[Multivariated_Data_Analysis.pdf](assets/Multivariated_Data_Analysis.pdf)