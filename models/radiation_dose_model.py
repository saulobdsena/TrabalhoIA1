import numpy as np
# SO PARA RESOLVER O PROBLEMA DE IMPORT NO MEU WSL (KAIO) vvvvv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# SO PARA RESOLVER O PROBLEMA DE IMPORT NO MEU WSL (KAIO) ^^^^^^
from classes.MRegression import MRegression
from classes.ModelStatus import ModelStatus

ROOT = Path(__file__).resolve().parents[1]


data = np.loadtxt(
    ROOT / "Docs" / "dose_radiacao_expandido.csv",
    delimiter=",",
    skiprows=1,          # tira o header
    usecols=(1, 2, 3),   # tira a coluna de index
)

y = data[:, 0]    # Dose_de_Radiacao
X = data[:, 1:]   # mAmp, Tempo_de_Exposicao

# print(f"X => {X}")
# print("\t\t\t\t")
# print(f"y => {y}")


model = MRegression(X=X, y=y)
model.fit()
print(f"(a) Ajuste um modelo de regressao linear multipla a esses dados, com dose de radiacao como variavel resposta:")
try:
    model.show()
except ValueError as e:
    print(e)

predict =model.predict(np.array([[15, 5]]))
print(f"(b) Use o modelo para prever a dose de radiacao quando a corrente for de 15 miliamperes e o tempo de exposicao for de 5 minutos:\n{predict[0]:.2f}")

status = ModelStatus(y_true=y, y_pred=model.predict(X))
print(f"(c) Coeficiente de determinacao do modelo:")
print(f"R^2 = {status.coefficient_determination():.4f}")

print(f"(d) Muitos usuarios de regressao preferem usar uma estatistica de valor ajustado de R2. Por que? Ela foi melhor que R2 comum? Se sim, por que?")
print(f"R^2 ajustado = {status.adjusted_coefficient_determination(k=X.shape[1]):.4f}")
print(f"-> Por que o R2 ajustado impede que o modelo pareça melhor do que realmente é ao adicionarmos variáveis inúteis. R2 ajustado adiciona uma penalidade no número de variáveis, se uma variável não agregar com o modelo, ela puxa o resultado da avaliação para baixo.")

print(f"(e) Compare este modelo com um modelo alternativo que use apenas a Corrente como preditor. Qual modelo ´e melhor? Por que?")
X_alternativo = data[:, 1:2]
model_alternativo = MRegression(X=X_alternativo, y=y)
model_alternativo.fit()
try:
    model_alternativo.show()
except ValueError as e:
    print(e)

status_alt = ModelStatus(y_true=y, y_pred=model_alternativo.predict(X_alternativo))
print(f"R^2 = {status_alt.coefficient_determination():.4f}")
print(f"R^2 ajustado = {status_alt.adjusted_coefficient_determination(k=X_alternativo.shape[1]):.4f}")
