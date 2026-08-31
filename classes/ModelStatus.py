import numpy as np

class ModelStatus:
    def __init__(self, y_true, y_pred):
        self.y_true = np.asarray(y_true, dtype=float) # valores observados
        self.y_pred = np.asarray(y_pred, dtype=float) # valores previstos pelo modelo

    def coefficient_determination(self) -> float:
        ybar = np.mean(self.y_true)
        ss_res = np.sum((self.y_true - self.y_pred) ** 2) # soma dos quadrados dos residuos
        ss_tot = np.sum((self.y_true - ybar) ** 2)        # soma dos quadrados totais
        return 1 - ss_res / ss_tot

    def adjusted_coefficient_determination(self, k: int) -> float:
        # k = numero de variaveis regressoras (sem contar o intercepto)
        n = self.y_true.shape[0]
        r2 = self.coefficient_determination()
        return 1 - (1 - r2) * (n - 1) / (n - k - 1)
