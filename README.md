# hugging-face-co2-multivariate-analysis
# Análisis multivariado de emisión de CO2 de modelos de IA

## 🎯 Objetivo del Proyecto

Este repositorio contiene un análisis estadístico multivariado sobre los factores que impactan la huella de carbono (CO2e) durante el entrenamiento de modelos de Machine Learning. El estudio se basa en un conjunto de datos extraído del popular repositorio **Hugging Face Hub**, con el fin de identificar patrones y relaciones clave entre las características de un modelo y su impacto ambiental.

## 📂 Contenido del Repositorio

**/assets**: Contiene el dataset original (HFCO2.csv), el archivo preprocesado y los resultados de las matrices. También contiene los reportes

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

## Reportes del proyecto
Resultados del análisis pueden verse en las siguientes rutas:
<br>
- [Components and factor analysis](assets/Multivariated_Data_Analysis.pdf)
- [Classification Analysis](assets/report/classification/document.pdf)
