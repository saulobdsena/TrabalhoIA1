import numpy as np

class MRegression:
    def __init__(self, X, y, fit_intercept: bool = True):
        self.X = np.asarray(X, dtype=float) # Matriz de regressão -> Dados que serão calculados
        self.y = np.asarray(y, dtype=float) # Vetor de respostas
        self.fit_intercept = fit_intercept # se False, forca o intercepto (b_0) a ser zero
        self.beta = None # coefs angulares
        self.N = self.X.shape[0] # linhas do dataset

    def _design_matrix(self, X):
        # Monta a matriz de regressao, com ou sem a coluna de 1s do intercepto
        X = np.asarray(X, dtype=float)
        if not self.fit_intercept:
            return X
        return np.column_stack((np.ones(X.shape[0]), X))

    def fit(self):
        Xb = self._design_matrix(self.X) # Nao modifica diretamente o self.X por conta do R^2
        self.beta = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ self.y
        return self

    def predict(self, X_new):
        return self._design_matrix(X_new) @ self.beta

    def show(self) -> None:
        if self.beta is None:
            raise ValueError("Modelo ainda não foi treinado. Chame fit() antes.")
        start = 0 if self.fit_intercept else 1
        for i, b in enumerate(self.beta, start=start):
            print(f"b_{i} = {b:.4f}")
