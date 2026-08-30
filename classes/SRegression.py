import numpy as np

class SRegression:
    def __init__(self,x, y): #construtor
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.b0 = None #coeficiente intercepto
        self.b1 = None #coeficiente angular

    def fit(self): #treinamento
        xbar = np.mean(self.x)
        ybar = np.mean(self.y)
        self.b1 = np.sum((self.y-ybar)*(self.x-xbar))/ \
        np.sum((self.x-xbar)**2)
        self.b0 = ybar - self.b1 * xbar
        return self #representa o proprio objeto da classe,
    #armazena os valores

    def predict(self, x_new):
        return self.b0 + self.b1 * np.array(x_new) 
    
    def summary(self):
        print(f"Modelo: y = {self.b0} + {self.b1} * x")
        print(f"Intercepto = {self.b0}")
        print(f"Coefiente Angular = {self.b1}")
