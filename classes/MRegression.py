import numpy as np

class MRegression:
    def __init__(self, X, y, fit_intercept=True):
        self.X = np.atleast_2d(np.asarray(X, dtype=float)) # Matriz de regressao -> Dados que serao calculados
        self.y = np.asarray(y, dtype=float) # Vetor de respostas
        self.fit_intercept = fit_intercept # se False, ajusta o modelo sem intercepto (beta_0 = 0)
        self.beta = None # coeficientes estimados
        self.N = self.X.shape[0] # numero de observacoes (linhas)
        self.p = self.X.shape[1] # numero de regressores (sem contar o intercepto)

    def _design(self, X):
        # Monta a matriz de projeto, adicionando a coluna de 1s quando ha intercepto.
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.fit_intercept:
            return np.column_stack((np.ones(X.shape[0]), X))
        return X

    def fit(self):
        # OLS (minimos quadrados): beta = (XtX)^-1 Xt y
        Xb = self._design(self.X) # nao modifica self.X para nao afetar o calculo do R^2
        self.beta = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ self.y
        return self

    def predict(self, X_new):
        Xb = self._design(X_new)
        return Xb @ self.beta

    # --- Diagnostico e metricas (implementadas na mao, sem sklearn) ---

    def fitted(self):
        # Valores ajustados y_hat para as observacoes de treino.
        return self.predict(self.X)

    def residuals(self):
        # Residuos e_i = y_i - y_hat_i.
        return self.y - self.fitted()

    def r2(self):
        # R^2 = 1 - SSR/SST
        e = self.residuals()
        ss_res = np.sum(e ** 2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        return 1 - ss_res / ss_tot

    def r2_adj(self):
        # R^2 ajustado = 1 - (1 - R^2) * (n - 1) / (n - p - 1)
        r2 = self.r2()
        return 1 - (1 - r2) * (self.N - 1) / (self.N - self.p - 1)

    def mse(self):
        # Erro quadratico medio.
        return np.mean(self.residuals() ** 2)

    def rmse(self):
        # Raiz do erro quadratico medio.
        return np.sqrt(self.mse())

    def mae(self):
        # Erro absoluto medio.
        return np.mean(np.abs(self.residuals()))
