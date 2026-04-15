import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# carpeta output
os.makedirs("output", exist_ok=True)

# cargar dataset
df = pd.read_csv("data/airbnb.csv")

# -------------------------------
# PREPROCESAMIENTO
# -------------------------------

# Eliminamos columnas irrelevantes
df = df.drop(columns=["id", "name", "host_name", "last_review"])

# Eliminamos nulos
df = df.dropna()

# Variable objetivo
target = "price"

# Variables categóricas → one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Separar X e y
X = df.drop(columns=[target])
y = df[target]

# -------------------------------
# TRAIN / TEST
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODELO
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -------------------------------
# MÉTRICAS
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- MÉTRICAS ---")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# Guardar métricas
with open("output/ej2_metricas_regresion.txt", "w") as f:
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R2: {r2:.4f}\n")

# -------------------------------
# RESIDUOS
# -------------------------------
residuos = y_test - y_pred

plt.figure(figsize=(8,5))
plt.scatter(y_pred, residuos, alpha=0.5)
plt.axhline(0)
plt.xlabel("Valores predichos")
plt.ylabel("Residuos")
plt.title("Gráfico de residuos")

plt.savefig("output/ej2_residuos.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nOutput generado correctamente")