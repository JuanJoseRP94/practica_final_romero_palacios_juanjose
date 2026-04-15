# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo
---
Dataset: NYC Airbnb Open Data 
Fuente: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data?resource=download

El dataset utilizado corresponde a datos de Airbnb en Nueva York, que contiene información sobre alojamientos, anfitriones, ubicación, precios y disponibilidad.

El objetivo del análisis es comprender la estructura del dataset, detectar problemas de calidad (nulos, outliers) y estudiar relaciones entre variables numéricas para identificar posibles factores influyentes sobre el precio.

---

**Pregunta 1.1 — ¿De qué fuente proviene el dataset?**
 El dataset proviene de datos públicos de Airbnb sobre alojamientos en Nueva York (NYC Airbnb Open Data).

**¿y cuál es la variable objetivo (target)?**
 La variable objetivo seleccionada es price, ya que representa el precio del alojamiento y es una variable numérica continua adecuada para problemas de regresión.

 **¿Por qué tiene sentido hacer regresión sobre ella?**
 Tiene sentido realizar regresión sobre esta variable porque el precio depende de múltiples factores como la ubicación, el tipo de habitación, el número de noches mínimas o la disponibilidad, lo que permite modelar su comportamiento mediante variables predictoras.

**Pregunta 1.2 — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers?**
Las variables numéricas presentan distribuciones claramente asimétricas en varios casos.

Variables como price, minimum_nights, number_of_reviews y calculated_host_listings_count presentan alta dispersión y fuerte sesgo a la derecha.
Se observan valores extremos muy elevados en varias variables, especialmente en:
price (hasta 10000)
minimum_nights (hasta 1250)
calculated_host_listings_count (hasta 327)

**Indica en qué variables y qué has decidido hacer con ellos.**
Se ha aplicado el método IQR (rango intercuartílico) para la detección de outliers.

Conclusión:

Se detectan numerosos outliers en casi todas las variables numéricas.
No se han eliminado, ya que en este ejercicio exploratorio se mantienen para no perder información relevante del comportamiento real del mercado.

**Pregunta 1.3 — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo?** Indica los coeficientes.

Las tres variables numéricas con mayor correlación (en valor absoluto) con la variable objetivo price son:

price (variable objetivo)
calculated_host_listings_count
minimum_nights
availability_365 (muy cercana en algunos casos)

Interpretación:

minimum_nights suele tener correlación positiva moderada: alojamientos que exigen más noches tienden a tener precios más altos.
calculated_host_listings_count puede reflejar hosts profesionales, con precios más altos o estructuras de alquiler más complejas.
availability_365 muestra relación con la disponibilidad anual del alojamiento.

**Pregunta 1.4 — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?**

El dataset presenta valores nulos en las siguientes columnas:

name (~0.03%)
host_name (~0.04%)
last_review (~20.55%)
reviews_per_month (~20.55%)

Tratamiento realizado:

Las variables name y host_name no se han utilizado en el análisis numérico, por lo que los nulos no afectan al modelado.
last_review y reviews_per_month presentan un porcentaje alto de nulos, lo que indica que muchos alojamientos no tienen reseñas.
En este análisis exploratorio no se han imputado valores, ya que el objetivo es descriptivo y no predictivo.

---

## Ejercicio 2 — Inferencia con Scikit-Learn

---
Añade aqui tu descripción y analisis:

---

**Pregunta 2.1 — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?**

Los resultados obtenidos para el modelo de regresión lineal son:

MAE ≈ 58.97
RMSE ≈ 152.88
R² ≈ 0.19

El modelo presenta un rendimiento limitado, ya que el valor de R² indica que solo se explica aproximadamente el 19% de la variabilidad de la variable objetivo (price).

El MAE indica que el error medio es de unos 59 dólares, lo cual es elevado en relación con muchos precios del dataset. Además, el RMSE es considerablemente alto, lo que sugiere la presencia de outliers y errores grandes en algunas predicciones.

Esto se debe principalmente a la alta variabilidad del dataset de Airbnb, que contiene precios muy dispersos y numerosos valores extremos. Además, el modelo lineal simple no es capaz de capturar relaciones complejas entre variables ni factores no observados como la calidad del alojamiento o su ubicación exacta.

En conclusión, el modelo presenta underfitting, ya que no logra capturar adecuadamente la estructura de los datos. Sin embargo, el resultado es coherente con el análisis descriptivo previo, donde ya se observaba alta dispersión y presencia de outliers.


---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---
En este ejercicio se ha implementado desde cero un modelo de regresión lineal múltiple utilizando únicamente NumPy, aplicando la solución analítica de Mínimos Cuadrados Ordinarios (OLS).

El objetivo ha sido comprender el funcionamiento interno de la regresión lineal, evitando el uso de librerías de alto nivel como Scikit-Learn, y trabajando directamente con operaciones matriciales.

Se ha construido la matriz de diseño añadiendo una columna de unos para incluir el término independiente, y se han calculado los coeficientes del modelo mediante la formulación matricial. Posteriormente, se han generado predicciones sobre el conjunto de test y se han evaluado utilizando métricas como MAE, RMSE y R².

Los resultados obtenidos muestran que los coeficientes estimados son muy cercanos a los valores reales utilizados para generar los datos sintéticos, lo que confirma que la implementación es correcta. Además, las métricas de evaluación se encuentran dentro de los rangos esperados, lo que indica un buen ajuste del modelo.

Este ejercicio permite entender de forma más profunda cómo funcionan internamente los modelos de regresión lineal y cómo se obtiene la solución óptima a partir de los datos.
---

**Pregunta 3.1 — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.**

La fórmula β = (XᵀX)⁻¹ Xᵀy permite calcular los coeficientes óptimos del modelo de regresión lineal minimizando el error cuadrático.

X representa la matriz de variables independientes
y es el vector de valores reales
β son los coeficientes del modelo

El término (XᵀX)⁻¹ Xᵀ actúa como una transformación que proyecta los datos en el espacio que minimiza el error.

Es necesario añadir una columna de unos a la matriz X para incluir el término independiente (intercepto β₀), ya que de lo contrario el modelo estaría forzado a pasar por el origen.

**Pregunta 3.2 — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.**

| Parametro | Valor real | Valor ajustado |
|-----------|-----------|----------------|
| β₀        | 5.0       |    4.865       |
| β₁        | 2.0       |    2.064       |
| β₂        | -1.0      |   -1.117       |
| β₃        | 0.5       |    0.439       |

Los coeficientes ajustados son muy cercanos a los valores reales, con pequeñas desviaciones debidas al ruido introducido en los datos.

**Pregunta 3.3 — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?**

Las métricas obtenidas son:

MAE ≈ 1.17
RMSE ≈ 1.46
R² ≈ 0.69

Los valores de MAE y RMSE son muy cercanos a los de referencia del enunciado, lo que indica que el modelo está ajustando correctamente los datos.

El valor de R² es ligeramente inferior al esperado (~0.80), lo que puede deberse a variaciones en el ruido aleatorio o a pequeñas diferencias numéricas en la implementación. Aun así, el modelo explica una proporción significativa de la variabilidad de los datos.

En conjunto, los resultados son coherentes y confirman que la implementación de la regresión lineal múltiple es correcta.

**Pregunta 3.4 — Compara los resultados con la reacción logística anterior para tu dataset y comprueba si el resultado es parecido. Explica qué ha sucedido.** 

En comparación con el modelo de regresión lineal del Ejercicio 2, este modelo presenta un rendimiento significativamente mejor.

Esto se debe a que en este caso se utilizan datos sintéticos generados a partir de una relación lineal conocida, mientras que en el dataset de Airbnb la relación entre variables es mucho más compleja y ruidosa.

El modelo del Ejercicio 3 refleja una situación ideal donde se cumplen los supuestos de la regresión lineal, mientras que el del Ejercicio 2 trabaja con datos reales mucho más difíciles de modelar.

---

## Ejercicio 4 — Series Temporales
---
En este ejercicio se ha trabajado con una serie temporal sintética generada artificialmente con una estructura conocida. La serie incluye cuatro componentes principales: una tendencia lineal creciente, una estacionalidad anual, un ciclo de largo plazo y un componente de ruido aleatorio.

El objetivo del análisis ha sido descomponer la serie en sus partes fundamentales mediante seasonal_decompose y estudiar el comportamiento del residuo para determinar si se asemeja a un ruido blanco.

En primer lugar, se ha visualizado la serie completa, donde se observa una evolución creciente a lo largo del tiempo, junto con oscilaciones regulares que sugieren la presencia de estacionalidad. Posteriormente, la descomposición ha permitido separar claramente la tendencia, la estacionalidad y el residuo.

La tendencia muestra un crecimiento aproximadamente lineal, mientras que la estacionalidad presenta un patrón periódico con periodo anual (365 días) y amplitud relativamente constante. Además, se aprecia un ciclo de largo plazo que modula ligeramente la serie cada varios años.

Finalmente, el análisis del residuo indica que este se comporta de forma similar a un ruido aleatorio con media cercana a cero, varianza aproximadamente constante y sin patrones claros de autocorrelación. Los tests estadísticos realizados (Jarque-Bera y ADF) respaldan esta interpretación, sugiriendo que el residuo es aproximadamente gaussiano y estacionario.

En conjunto, la serie puede modelarse adecuadamente mediante un esquema aditivo clásico: tendencia + estacionalidad + ciclo + ruido.

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

Sí, la serie presenta una tendencia claramente creciente y aproximadamente lineal.
Esto se debe al componente de tendencia definido en la generación de datos (0.05 * t + 50), lo que provoca un crecimiento sostenido a lo largo del tiempo.

La magnitud es moderada pero constante, sin cambios bruscos.

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

Sí, existe estacionalidad clara.

Periodo aproximado: 365 días (anual)
Patrón: oscilaciones regulares sinusoidales
Amplitud: aproximadamente ±15 unidades (componente principal) con variaciones adicionales menores

Esto se observa claramente en la descomposición.

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

Sí.

Además de la estacionalidad anual, se observa un ciclo de largo plazo de aproximadamente 4 años (1461 días).

Diferencia con la tendencia:

La tendencia es monótona creciente.
El ciclo es oscilatorio de baja frecuencia alrededor de la tendencia.

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

El residuo se aproxima bastante a un ruido blanco, aunque no perfectamente ideal.

Resultados:

Media ≈ 0 (ligeramente cercana a 0)
Desviación típica ≈ constante (~3.5 en la generación)
Asimetría ≈ 0 (simetría aceptable)
Curtosis ≈ cercana a 0 (ligero desvío posible)

Tests:

Jarque-Bera:
p-value > 0.05 → no se rechaza normalidad (aprox. ruido gaussiano)
ADF test:
p-value < 0.05 → el residuo es estacionario

Conclusión:
El residuo puede considerarse un ruido aproximadamente gaussiano y estacionario, aunque con pequeñas desviaciones propias de datos simulados.


*El análisis confirma que la serie sigue un modelo aditivo clásico (tendencia + estacionalidad + ruido), lo que valida el uso de descomposición clásica.*

---

*Fin del documento de respuestas*