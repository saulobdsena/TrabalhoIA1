# Guia de estudo — Problema 1 (Modelagem)

> Minha parte no trabalho: **leitor de dados + métricas na classe + itens (a), (b), (d) e (e)** do Problema 1.
> Este documento é para eu **entender e saber explicar** cada passo na arguição — não é o relatório final (esse é do Kaio).

---

## 1. Contexto do problema

Um estudo (Cancer Epidemiology, 1996) mediu a concentração de **arsênio nas unhas do pé** de 21 pessoas para avaliar se isso serve como indicador da ingestão de água contaminada. Para cada pessoa temos:

| Coluna | Significado |
|--------|-------------|
| `Idade` | idade em anos |
| `Sexo` | 1 = masculino, 2 = feminino (**não usamos** no modelo) |
| `Uso_Beber` | frequência do poço para beber (categorias 1 a 5) |
| `Uso_Cozinhar` | frequência do poço para cozinhar (categorias 1 a 5) |
| `Arsenio_Agua` | arsênio na água (ppm) |
| `Arsenio_Unhas` | arsênio nas unhas (ppm) — **variável resposta (y)** |

**Objetivo:** ajustar uma regressão linear múltipla que preveja `Arsenio_Unhas` a partir de `Idade`, `Uso_Beber`, `Uso_Cozinhar` e `Arsenio_Agua`.

---

## 2. Fundamento teórico

### 2.1. O que é regressão linear múltipla

Supomos que a resposta é uma combinação linear dos regressores mais um erro:

```
y = β₀ + β₁·Idade + β₂·Uso_Beber + β₃·Uso_Cozinhar + β₄·Arsenio_Agua + ε
```

- `β₀` = **intercepto**: valor previsto de y quando todos os regressores são 0.
- `β₁…β₄` = **coeficientes angulares**: quanto y muda quando aquele regressor aumenta em 1 unidade, **mantendo os outros constantes**.
- `ε` = erro (o que o modelo não explica).

### 2.2. Como encontramos os coeficientes — Mínimos Quadrados (OLS)

Queremos os `β` que **minimizam a soma dos erros ao quadrado** entre o valor real e o previsto:

```
minimizar  Σ (yᵢ − ŷᵢ)²
```

Escrevendo em forma matricial, montamos a **matriz de projeto X** onde cada linha é uma observação e adicionamos uma coluna de 1s à esquerda (essa coluna "carrega" o intercepto β₀):

```
      ┌ 1  Idade₁  Beber₁  Cozinhar₁  Agua₁ ┐
  X = │ 1  Idade₂  Beber₂  Cozinhar₂  Agua₂ │      y = vetor das 21 respostas
      │ ...                                  │
      └ 1  Idade₂₁ ...                       ┘
```

A solução analítica (equações normais) é:

```
β = (Xᵀ X)⁻¹ Xᵀ y
```

**Por que essa fórmula?** Derivando a soma dos quadrados em relação a β e igualando a zero, chega-se exatamente a `XᵀX β = Xᵀy`. É a solução exata do problema de minimização — não é iterativa, não precisa de "treino" como uma rede neural.

**Detalhe de implementação:** no código uso `np.linalg.pinv` (pseudo-inversa) em vez de `inv`. A pseudo-inversa devolve o mesmo resultado quando `XᵀX` é invertível, mas **não quebra** se houver colinearidade (matriz mal-condicionada). É a escolha numericamente segura.

---

## 3. O que eu implementei (e onde)

### 3.1. Leitor de dados — [`classes/dataset.py`](dataset.py)

A regra do trabalho **proíbe pandas**, então leio o CSV na mão com o módulo `csv`:

- `load_dataset(path)` → abre o arquivo, separa o cabeçalho, ignora linhas em branco e devolve `(header, data)` com `data` como matriz numpy de floats.
- `column(header, data, nome)` → devolve uma coluna pelo nome (`data[:, header.index(nome)]`).

> Uso `encoding="utf-8-sig"` para descartar o BOM que às vezes vem no começo do arquivo.

### 3.2. A classe do modelo — [`classes/MRegression.py`](../classes/MRegression.py)

Guarda X, y e os coeficientes. Métodos principais:

| Método | O que faz |
|--------|-----------|
| `fit()` | calcula `β = (XᵀX)⁻¹Xᵀy` (com coluna de 1s se houver intercepto) |
| `predict(X_new)` | aplica `X_new · β` para prever novos casos |
| `fitted()` | valores previstos ŷ para os próprios dados de treino |
| `residuals()` | resíduos `e = y − ŷ` |
| `r2()` | R² |
| `r2_adj()` | R² ajustado |
| `mse()`, `rmse()`, `mae()` | métricas de erro |

O parâmetro `fit_intercept` permite ajustar **sem intercepto** (β₀ = 0) — isso é usado na parte do Saulo (cenário do item g).

> **Cuidado de projeto:** o `fit` monta uma cópia da matriz com a coluna de 1s numa variável local (`Xb`), sem alterar `self.X`. Isso evita bagunçar o cálculo do R², que precisa do X original.

### 3.3. Script de resultados — [`problem1_model.py`](../problem1_model.py)

Carrega os dados, ajusta o modelo e imprime os itens (a), (b), (d), (e).

---

## 4. Resultados e como interpretá-los

### (a) Modelo ajustado

```
Arsenio_Unhas = 0,48751
                − 0,00077 · Idade
                − 0,02274 · Uso_Beber
                − 0,04150 · Uso_Cozinhar
                + 13,24001 · Arsenio_Agua
```

**Interpretação dos coeficientes** (o que a professora provavelmente vai perguntar):

- **Arsênio na água (+13,24):** é o preditor dominante. A cada aumento de 1 ppm de arsênio na água, o arsênio nas unhas sobe ~13,24 ppm (mantendo o resto constante). Faz sentido físico: a água contaminada é a fonte direta da exposição.
- **Idade (−0,00077):** efeito praticamente nulo — quase não influencia.
- **Uso_Beber (−0,0227) e Uso_Cozinhar (−0,0415):** efeitos pequenos e negativos. O sinal negativo é contraintuitivo (esperaríamos que usar mais o poço aumentasse o arsênio); provavelmente é ruído por causa da **amostra pequena (21 pessoas)** e da correlação dessas variáveis com o arsênio na água.
- **Intercepto (0,4875):** valor base teórico quando todos os regressores são 0. Aqui não tem interpretação prática direta (idade 0 não existe), serve para ajustar o nível da reta.

### (b) Previsão pedida

Para idade=30, beber=5, cozinhar=5, arsênio na água=0,135:

```
Arsenio_Unhas previsto = 1,931 ppm
```

Repare que o resultado é puxado quase todo pelo termo `13,24 × 0,135 ≈ 1,79`.

### (d) R² = 0,812

O R² mede a **fração da variação de y que o modelo explica**:

```
R² = 1 − SSR/SST
   SSR = Σ(yᵢ − ŷᵢ)²   (o que o modelo erra)
   SST = Σ(yᵢ − ȳ)²    (variação total em torno da média)
```

**0,812 = o modelo explica ~81% da variação** do arsênio nas unhas. É um ajuste bom. Se o modelo fosse perfeito, R²=1; se não fosse melhor que "chutar a média", R²=0.

### (e) R² ajustado = 0,765

Problema do R² comum: **ele nunca diminui quando você adiciona um regressor**, mesmo um inútil. Então ele "premia" modelos com muitas variáveis, o que engana.

O R² ajustado corrige isso penalizando pelo número de variáveis:

```
R²_adj = 1 − (1 − R²) · (n − 1)/(n − p − 1)
```

com n=21 observações e p=4 regressores. Ele **só sobe se a variável nova melhorar o modelo mais do que o esperado por acaso**.

Aqui: R²=0,812 caiu para R²_adj=0,765. A diferença (~0,047) reflete o "preço" de ter 4 variáveis com poucos dados. Para **comparar modelos com números diferentes de preditores** (é o que os itens f/g fazem), o R² ajustado é a métrica mais honesta.

---

## 5. Perguntas prováveis na arguição (e respostas curtas)

**"Por que não usaram sklearn?"**
Regra do trabalho: proibido. Implementamos OLS na mão com as equações normais em numpy.

**"Por que `pinv` e não `inv`?"**
A pseudo-inversa dá o mesmo resultado quando dá para inverter, mas é estável se a matriz for mal-condicionada (colinearidade). Não quebra o código.

**"O que a coluna de 1s faz?"**
É o intercepto β₀. Sem ela, a reta seria forçada a passar pela origem.

**"Por que o Sexo não entrou?"**
O enunciado pede só idade, uso para beber, uso para cozinhar e arsênio na água como regressores.

**"O R² de 0,81 é bom?"**
Sim, explica 81% da variação. Mas com só 21 observações e 4 variáveis, olhamos também o R² ajustado (0,765) para não superestimar.

**"Diferença entre R² e R² ajustado?"**
R² nunca cai ao adicionar variáveis; o ajustado penaliza variáveis a mais e pode cair. Por isso o ajustado é melhor para comparar modelos.

**"Como o modelo prevê um caso novo?"**
Monta o vetor `[1, idade, beber, cozinhar, agua]` e faz o produto escalar com β.

---

## 6. Como rodar

```bash
pip install numpy
python problem1_model.py
```
