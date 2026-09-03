"""Problema 1 - Arsenio nas unhas (parte de modelagem).

Cobre os itens:
  (a) Ajuste do modelo de regressao linear multipla.
  (b) Previsao para idade=30, beber=5, cozinhar=5, arsenio na agua=0.135.
  (d) R^2.
  (e) R^2 ajustado (e por que e preferido).

Regressores: Idade, Uso_Beber, Uso_Cozinhar, Arsenio_Agua.
Resposta:    Arsenio_Unhas. (Sexo nao entra no modelo.)
"""

import numpy as np

from classes.dataset import load_dataset, column
from classes.MRegression import MRegression

# --- Carregamento dos dados (sem pandas) ---
header, data = load_dataset("Docs/arsenio_dataset.csv")

idade = column(header, data, "Idade")
beber = column(header, data, "Uso_Beber")
cozinhar = column(header, data, "Uso_Cozinhar")
agua = column(header, data, "Arsenio_Agua")
y = column(header, data, "Arsenio_Unhas")

X = np.column_stack((idade, beber, cozinhar, agua))

# --- (a) Ajuste do modelo ---
model = MRegression(X, y).fit()
nomes = ["Intercepto", "Idade", "Uso_Beber", "Uso_Cozinhar", "Arsenio_Agua"]

print("=" * 60)
print("(a) MODELO DE REGRESSAO LINEAR MULTIPLA")
print("=" * 60)
for nome, coef in zip(nomes, model.beta):
    print(f"  {nome:<14} = {coef: .6f}")

print("\n  Equacao:")
print(f"  Arsenio_Unhas = {model.beta[0]:.5f}")
for nome, coef in zip(nomes[1:], model.beta[1:]):
    sinal = "+" if coef >= 0 else "-"
    print(f"                  {sinal} {abs(coef):.5f} * {nome}")

# --- (b) Previsao pedida ---
x_new = np.array([30, 5, 5, 0.135])  # idade, beber, cozinhar, arsenio_agua
pred = model.predict(x_new)[0]

print("\n" + "=" * 60)
print("(b) PREVISAO")
print("=" * 60)
print("  idade=30, beber=5, cozinhar=5, arsenio_agua=0.135")
print(f"  Arsenio previsto nas unhas = {pred:.5f} ppm")

# --- (d) R^2 ---
r2 = model.r2()
print("\n" + "=" * 60)
print("(d) R^2")
print("=" * 60)
print(f"  R^2 = {r2:.5f}")

# --- (e) R^2 ajustado ---
r2_adj = model.r2_adj()
print("\n" + "=" * 60)
print("(e) R^2 AJUSTADO")
print("=" * 60)
print(f"  R^2 ajustado = {r2_adj:.5f}   (n={model.N}, p={model.p})")
print(
    "\n  Por que usar o R^2 ajustado: o R^2 comum nunca diminui ao adicionar\n"
    "  regressores, mesmo os inuteis, entao ele tende a superestimar o ajuste.\n"
    "  O R^2 ajustado penaliza pelo numero de variaveis (p) em relacao ao\n"
    "  numero de observacoes (n), so subindo quando a variavel nova melhora o\n"
    "  modelo mais do que o esperado ao acaso. Por isso e preferido para\n"
    "  comparar modelos com quantidades diferentes de preditores."
)

# --- (f) Modelo alternativo: somente arsenio na agua ---
modelo_agua = MRegression(agua.reshape(-1, 1), y).fit()

# --- (g) Modelo completo com intercepto forcado a zero ---
modelo_sem_intercepto = MRegression(X, y, fit_intercept=False).fit()


# --- (h) Comparacao das metricas ---
def mostrar_metricas(nome, modelo, mostrar_r2_ajustado=True):
    print(f"\n{nome}")
    print(f"Coeficientes: {modelo.beta}")
    print(f"R^2:          {modelo.r2():.5f}")
    if mostrar_r2_ajustado:
        print(f"R^2 ajustado: {modelo.r2_adj():.5f}")
    print(f"MSE:          {modelo.mse():.5f}")
    print(f"RMSE:         {modelo.rmse():.5f}")
    print(f"MAE:          {modelo.mae():.5f}")


mostrar_metricas("(h) Modelo completo", model)
mostrar_metricas("(f/h) Modelo somente com arsenio na agua", modelo_agua)
mostrar_metricas(
    "(g) Modelo com intercepto forcado a zero",
    modelo_sem_intercepto,
    mostrar_r2_ajustado=False,
)
