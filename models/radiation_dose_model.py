import numpy as np
# SO PARA RESOLVER O PROBLEMA DE IMPORT NO MEU WSL (KAIO) vvvvv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# SO PARA RESOLVER O PROBLEMA DE IMPORT NO MEU WSL (KAIO) ^^^^^^
from classes.MRegression import MRegression

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
try:
    model.show()
except ValueError as e:
    print(e)

predict =model.predict(np.array([[15, 5]]))
print(predict)


