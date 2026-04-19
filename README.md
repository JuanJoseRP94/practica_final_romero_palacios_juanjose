# Práctica Final — Análisis y Modelado de Datos

## Proyecto realizado por: Juan José Romero Palacios

Descripción del proyecto

Este proyecto consiste en el análisis exploratorio, modelado estadístico y análisis de series temporales sobre distintos conjuntos de datos.

El objetivo es aplicar técnicas de estadística y machine learning para extraer información relevante, identificar patrones y evaluar modelos predictivos.


Estructura del proyecto:
practica_final_romero_palacios_juanjose/  
│  
├── data/  
│   └── airbnb.csv  
│  
├── output/  
│   ├── gráficas generadas (.png)  
│   └── resultados de análisis (.txt)  
│  
├── ejercicio1_descriptivo.py  
├── ejercicio2_inferencia.py  
├── ejercicio3_regresion_multiple.py  
├── ejercicio4_series_temporales.py  
│  
├── Respuestas.md  
└── README.md  
*ligeramente modificado por orden alfabético generado automáticamente*

## Ejercicio 1 — Análisis descriptivo

Se realiza un análisis exploratorio del dataset de Airbnb, incluyendo:

Tipos de variables
Valores nulos
Estadísticos descriptivos
Detección de outliers mediante IQR
Histogramas y boxplots
Matriz de correlación

**Variable objetivo: price**

## Ejercicio 2 — Regresión con Scikit-Learn

Se entrena un modelo de regresión lineal para predecir el precio de los alojamientos.

Se evalúa mediante:

MAE
RMSE
R²

El modelo muestra capacidad explicativa moderada, con margen de mejora debido a la variabilidad del dataset.


## Ejercicio 3 — Regresión lineal en NumPy

Se implementa la regresión lineal múltiple desde cero utilizando la fórmula:

  β=(XTX)−1XTy

Se comparan los coeficientes obtenidos con los valores reales y se evalúa el rendimiento del modelo.


## Ejercicio 4 — Series temporales

Se analiza una serie temporal sintética con los siguientes componentes:

Tendencia lineal creciente
Estacionalidad anual
Ciclo de largo plazo
Ruido aleatorio

Se realiza descomposición y análisis del residuo para comprobar si se comporta como ruido blanco.

**Tecnologías utilizadas**
*Python 3.x*
*Pandas*
*NumPy*
*Matplotlib*
*Seaborn*
*Scikit-learn*
*Statsmodels*
*SciPy*

### Conclusión

El proyecto permite aplicar técnicas fundamentales de análisis de datos y modelado estadístico, desde exploración inicial hasta regresión y series temporales, obteniendo una visión completa del flujo de trabajo en ciencia de datos.

## **Cómo ejecutar el proyecto**

### Crear entorno (opcional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

### Instalar dependencias
pip install -r requirements.txt

### Ejecutar ejercicios
python ejercicio1_descriptivo.py
python ejercicio2_regresion_sklearn.py
python ejercicio3_numpy.py
python ejercicio4_series_temporales.py