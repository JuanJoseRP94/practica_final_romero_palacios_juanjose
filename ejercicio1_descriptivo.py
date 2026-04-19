import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os

# configuración matplotlib
matplotlib.rcParams['text.usetex'] = False

# carpeta output
os.makedirs("output", exist_ok=True)

# cargar dataset
df = pd.read_csv("data/airbnb.csv")
df = df.copy()
df.columns = df.columns.str.replace('$', '', regex=False)


print(df.head())
print(df.shape)
df.info()

print("\n--- RESUMEN ESTRUCTURAL ---")
print("Filas y columnas:", df.shape)
print("\nTipos de datos:\n", df.dtypes)
print("\nValores nulos (%):\n", (df.isnull().mean() * 100))

print("\n--- ESTADÍSTICOS DESCRIPTIVOS ---")

desc = df.describe()
print(desc)

print("\n--- ESTADÍSTICOS COMPLEMENTARIOS ---")

num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    print(f"\nColumna: {col}")
    print("Media:", df[col].mean())
    print("Mediana:", df[col].median())
    print("Std:", df[col].std())
    print("Var:", df[col].var())
    print("Min:", df[col].min())
    print("Max:", df[col].max())


print("\n--- DETECCIÓN DE OUTLIERS (IQR) ---")

num_cols = df.select_dtypes(include=np.number).columns

outliers_summary = {}

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    outliers_summary[col] = len(outliers)

    print(f"{col}: {len(outliers)} outliers")

print("\n--- GENERANDO HISTOGRAMAS ---")

num_cols = df.select_dtypes(include=np.number).columns

df[num_cols].hist(figsize=(12, 10), bins=30)
plt.suptitle("Histogramas de variables numéricas")

plt.savefig("output/ej1_histogramas.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n--- GENERANDO BOXPLOTS ---")

target = "price" 

cat_cols = ["neighbourhood_group", "neighbourhood", "room_type"]

for col in cat_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df[col], y=df[target])
    plt.xticks(rotation=45)
    plt.title(f"{target} vs {col}")

    plt.savefig(f"output/boxplot_{col}.png", dpi=150, bbox_inches="tight")
    plt.close()

print("\n--- CORRELACIÓN ---")

corr = df.select_dtypes(include=np.number).corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Matriz de correlación")

plt.savefig("output/ej1_heatmap_correlacion.png", dpi=150, bbox_inches="tight")
plt.close()

# Top correlaciones con price
print("\nTop correlaciones con PRICE:")
print(corr["price"].sort_values(ascending=False))