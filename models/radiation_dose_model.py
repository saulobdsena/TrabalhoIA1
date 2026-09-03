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

print(f"(f) Cenário com Intercepto Forçado a Zero: Suponha que, por razões teóricas, você imponha a restrição de que o intercepto deve ser zero. Qual é a interpretação prática dessa decisão? Compare as métricas R2 e RMSE deste modelo com o modelo com intercepto. Qual você escolheria e por quê?")
model_sem_intercepto = MRegression(X=X, y=y, fit_intercept=False)
model_sem_intercepto.fit()
try:
    model_sem_intercepto.show()
except ValueError as e:
    print(e)

status_sem_intercepto = ModelStatus(y_true=y, y_pred=model_sem_intercepto.predict(X))

print(f"Modelo COM intercepto:  R^2 = {status.coefficient_determination():.4f} | RMSE = {status.rmse():.4f}")
print(f"Modelo SEM intercepto:  R^2 = {status_sem_intercepto.coefficient_determination():.4f} | RMSE = {status_sem_intercepto.rmse():.4f}")

print(f"-> Interpretacao pratica: forcar b_0 = 0 significa assumir que, sem corrente (mAmp = 0) e sem tempo de exposicao (Tempo = 0), a dose de radiacao e exatamente zero, ou seja, a reta de regressao passa obrigatoriamente pela origem e todo o efeito e atribuido apenas aos preditores. Em tese isso faz sentido fisico (aparelho desligado nao emite radiacao), mas na pratica os dados so foram observados numa faixa distante da origem, entao a restricao extrapola o modelo para uma regiao onde nao ha observacoes.")
print(f"-> Comparacao: o modelo sem intercepto e um caso restrito do modelo completo, entao seu ajuste nunca pode ser melhor. Aqui o intercepto estimado e grande e negativo (b_0 = -433.81), o que mostra que a restricao b_0 = 0 esta longe do que os dados indicam: o R^2 cai de 0.8431 para 0.7631 e o RMSE sobe de 233.52 para 286.92 (cerca de 23% de erro a mais). Os coeficientes tambem mudam bastante (b_1 vai de 17.90 para 4.88), porque sem o intercepto eles precisam absorver o nivel medio da resposta.")
print(f"-> Escolha: eu ficaria com o modelo COM intercepto. A perda de ajuste ao forcar b_0 = 0 e grande e os coeficientes ficam distorcidos, perdendo a interpretacao de efeito marginal de cada variavel. O intercepto negativo nao precisa ter sentido fisico: ele so ajusta o nivel da reta dentro da faixa observada dos dados, e nao deve ser lido como a dose prevista em mAmp = 0 e Tempo = 0, que esta fora dessa faixa.")
print(f"-> Obs: o R^2 do modelo sem intercepto foi calculado com a mesma formula (1 - SQres/SQtot em torno da media), entao ele nao tem a interpretacao usual de proporcao de variancia explicada e pode ate ficar negativo; por isso o RMSE e a comparacao mais confiavel entre os dois modelos.")

print(f"(g/h) Alem do R2: Calcule e interprete pelo menos outras duas metricas de erro para o modelo completo e para o modelo alternativo (apenas com Corrente). Sugestoes: MSE, RMSE e MAE.")
print(f"Modelo completo (mAmp + Tempo):   MSE = {status.mse():.4f} | RMSE = {status.rmse():.4f} | MAE = {status.mae():.4f}")
print(f"Modelo alternativo (so mAmp):     MSE = {status_alt.mse():.4f} | RMSE = {status_alt.rmse():.4f} | MAE = {status_alt.mae():.4f}")

print(f"-> MSE: media dos erros ao quadrado. Como eleva ao quadrado, penaliza muito mais os erros grandes, mas fica em rad^2, ou seja, nao esta na mesma unidade da dose e serve mais para comparar modelos do que para interpretar sozinho.")
print(f"-> RMSE: raiz do MSE, volta para a unidade da resposta (rad). No modelo completo o erro tipico e de cerca de {status.rmse():.0f} rad, contra cerca de {status_alt.rmse():.0f} rad do modelo so com a corrente.")
print(f"-> MAE: media dos erros em valor absoluto, tambem em rad, e mais robusta a valores extremos porque nao eleva ao quadrado. O MAE menor que o RMSE nos dois modelos indica que existem algumas observacoes com erro bem acima do tipico puxando o RMSE para cima.")
print(f"-> Comparacao: as tres metricas apontam na mesma direcao do R2 do item (e). O modelo completo erra bem menos ({status_alt.rmse() / status.rmse():.2f}x menor no RMSE e {status_alt.mae() / status.mae():.2f}x menor no MAE), confirmando que o tempo de exposicao carrega a maior parte da informacao sobre a dose e que a corrente sozinha explica muito pouco.")
