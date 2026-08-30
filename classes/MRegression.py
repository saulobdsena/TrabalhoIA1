import numpy as np

class MRegression:
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=float) # Matriz de regressão -> Dados que serão calculados
        self.y = np.asarray(y, dtype=float) # Vetor de respostas
        self.beta = None # coefs angulares
        self.N = self.X.shape[0] # linhas do dataset

    def fit(self):
        Xb = np.column_stack((np.ones(self.N), self.X)) # Nao modifica diretamente o self.X por conta do R^2
        self.beta = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ self.y
        return self

    def predict(self, X_new):
        N = X_new.shape[0]
        X_new = np.column_stack((np.ones(N), X_new))
        return X_new @ self.beta

    def show(self) -> None:
        if self.beta is None:
            raise ValueError("Modelo ainda não foi treinado. Chame fit() antes.")
        for i, b in enumerate(self.beta):
            print(f"b_{i} = {b:.4f}")