# Regresión Lineal Múltiple desde cero (NumPy)

## Descripción
Este proyecto implementa un modelo de regresión lineal múltiple desde cero utilizando la solución analítica de Mínimos Cuadrados Ordinarios (OLS), sin usar sklearn.

## Fórmula utilizada
β = (XᵀX)⁻¹ Xᵀy

## Funcionalidades
- Entrenamiento del modelo de regresión lineal múltiple
- Predicción sobre datos de test
- Cálculo de métricas:
  - MAE
  - RMSE
  - R²
- Generación de gráfico real vs predicho
- Exportación de resultados a carpeta `output/`

## Librerías utilizadas
- numpy
- matplotlib

## Cómo ejecutar el proyecto
```bash
python ejercicio3.py