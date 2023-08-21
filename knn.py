# -*- coding: utf-8 -*-
"""KNN.ipynb

## **Programa de Pós-Graduação em Computação - INF/UFRGS**
### Disciplina CMP263 - Aprendizagem de Máquina
#### *Profa. Mariana Recamonde-Mendoza (mrmendoza@inf.ufrgs.br)*
#### *Aluno  Tiago Knorst (tknorst@inf.ufrgs.br)*
<br>

<br>

## **Trabalho Final**

<br>

##**Classificação de tráfego de rede malicioso**

blablabla

---

###Carregando e inspecionando os dados

Primeiramente, vamos carregar algumas bibliotecas importantes do Python e os dados a serem utilizados neste estudo. Os dados são disponibilizados através de um link.
"""

# Commented out IPython magic to ensure Python compatibility.
## Carregando as bibliotecas necessárias
# A primeira linha é incluída para gerar os gráficos logo abaixo dos comandos de plot
# %matplotlib inline
import pandas as pd             # biblioteca para análise de dados
import matplotlib.pyplot as plt # biblioteca para visualização de informações
import seaborn as sns           # biblioteca para visualização de informações
import numpy as np              # biblioteca para operações com arrays multidimensionais
import ipaddress                # biblioteca para converter string IP para inteiro
import glob                     # biblioteca para acessar arquivos facilmente
from sklearn.neighbors import KNeighborsClassifier # biblioteca para treinar KNN
sns.set()

columns=["Address A","Port A","Address B","Port B","Packets","Bytes","Packets A → B","Bytes A → B","Packets B → A","Bytes B → A","Rel Start","Duration","Bits/s A → B","Bits/s B → A"]
data = pd.DataFrame(columns=columns)

print(glob.glob("*/"))
print(glob.glob("Benign/*"))
print(glob.glob("Malware/*"))

for path in glob.glob("*/"):
    for file in glob.glob(path+"*_TCP.csv"):
        temp = pd.read_csv(file, #index_col=0,
                                 dtype={'Address A':str,
                                        'Port A':int,
                                        'Address B':str,
                                        'Port B':int,
                                        'Packets':int,
                                        'Bytes':int,
                                        'Packets A → B':int,
                                        'Bytes A → B':int,
                                        'Packets B → A':int,
                                        'Bytes B → A':int,
                                        'Rel Start':float,
                                        'Duration':float,
                                        'Bits/s A → B':float,
                                        'Bits/s B → A':float},)
    
        if(path[0]=='M'): # Se esta na pasta Malware
            temp.insert(temp.shape[1], "Malware", np.ones(temp.shape[0]))
        else:
            temp.insert(temp.shape[1], "Malware", np.zeros(temp.shape[0]))

        #print(temp.head())  # para visualizar apenas as 5 primeiras linhas
        #print(temp.tail())  # para visualizar apenas as 5 ultimas linhas

        ## Características gerais do dataset
        print("O conjunto de dados "+file+" possui {} linhas e {} colunas".format(temp.shape[0], temp.shape[1]))

        data = pd.concat([data,temp.iloc[1:]])
        #data.reset_index(inplace=True) # reinicia indexacao apos concatenar diferentes dataframes
        #print(data.head())
        #print(data.iloc[7516:])
        #print(data.tail())


data.reset_index(inplace=True) # reinicia indexacao apos concatenar diferentes dataframes
data = data.drop(columns=['index'])

## Características gerais do dataset
print("O conjunto de dados completo possui {} linhas e {} colunas".format(data.shape[0], data.shape[1]))


data.columns = data.columns.str.replace(' ', '') # elimina espaçamentos nos nomes dos atributos

print(data.head())
print(data.tail())

#data.AddressA=int(ipaddress.ip_address(data.AddressA))
for i in (range(data.shape[0])):
    data.at[i, 'AddressA'] = int(ipaddress.ip_address(data.at[i, 'AddressA'])) # converte IP para inteiro
    data.at[i, 'AddressB'] = int(ipaddress.ip_address(data.at[i, 'AddressB'])) # converte IP para inteiro
    #print("AddressA i="+str(i)+" IP="+data.at[i, 'AddressA']+" int="+str(int(ipaddress.ip_address(data.at[i, 'AddressA'])))) 
    #print("AddressB i="+str(i)+" IP="+data.at[i, 'AddressB']+" int="+str(int(ipaddress.ip_address(data.at[i, 'AddressB'])))) 



"""A coluna *'diagnosis'* contém a classificação de cada amostra referente ao tipo de tumor, se maligno (M) ou benigno (B). Vamos avaliar como as instâncias estão distribuídas entre as classes presentes no dataset."""

## Distribuição do atributo alvo, 'diagnosis'
plt.hist(data['Malware'])
plt.title("Distribuição do atributo alvo - Malware")
plt.show()

"""Podemos perceber que existem mais instâncias classificadas como 'Benigno' do que como 'Maligno'. Portanto, existe um certo **desbalanceamento entre as classes**. Não vamos entrar em detalhes nesta atividade do possível impacto deste desbalanceamento no desempenho do modelo e tampouco como mitigar seus efeitos. Discutiremos esse assunto mais adiante. Por enquanto, é importante sabermos que temos mais exemplos do tipo 'Benigno' nos dados de treinamento, e portanto, é provável que qualquer modelo treinado tenha mais facilidade de acertar exemplos desta classe.

Vamos avaliar a distribuição de valores dos atributos preditivos. Faremos isto tanto através da sumarização da distribuição a partir do método `describe()`, como através da visualização dos histogramas para cada atributo utilizando o método `hist()`.
"""

data.drop(['Malware'],axis=1).describe()

data.drop(['Malware'],axis=1).hist(bins=15, figsize=(20,18))
plt.show()


plt.figure(figsize=(15,15))
sns.heatmap(data.corr(), annot=True, cmap="PuOr", annot_kws={"size": 9})
plt.show()

"""---


### Criando conjuntos de treino e teste para avaliação de modelos

Um dos princípios mais importantes no desenvolvimento de modelos de Aprendizado de Máquina é nunca avaliar um modelo com os mesmos dados nos quais ele foi treinado. Se cometermos este erro, teremos uma avaliação muito otimista, pois o modelo pode simplesmente decorar todos os dados analisados durante o treinamento e demonstrar excelente desempenho preditivo para estes dados - o que não necessariamente se repetirá ao ser aplicado a novos dados. Assim, a validação de modelos deve ser sempre feita com dados independentes.

Em muitos casos, não temos um conjunto de treino e teste já definidos. Ou seja, recebemos um único conjunto de dados para o desenvolvimento do modelo. Desta forma, o método **Holdout** é uma estratégia simples e flexível para gerar conjuntos de dados independentes: um conjunto é usado para treinar o modelo e o outro para testar o modelo. É imprescindível que estes conjuntos de dados sejam disjuntos, isto é, não podem ter nenhuma instância em comum.

Algumas proporções para a divisão dos dados em treino/teste são comumente adotadas na literatura, por exemplo, 80%/20% e 75%/25%.

Para o Holdout, iremos utilizar o método `train_test_split()` da biblioteca [scikit-learn](https://scikit-learn.org/stable/), uma das bibliotecas de Aprendizado de Máquina mais conhecidas e utilizadas do Python. Leia a documentação do método [aqui](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). O parâmetro *stratify* define que a divisão será feita seguindo a proporção de exemplos por classe determinada no vetor $y$ (rótulos).
"""

## Separa o dataset em duas variáveis: os atributos/entradas (X) e a classe/saída (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, 14].values

## substitui 'B' por 0, 'M' por 1
y = np.array([0 if y==0.0 else 1 for y in y])

## Importando o método train_test_split da biblioteca scikit-learn, utilizado para
## aplicação do holdout. Este método já permite especificar a proporção do conjunto
## de teste
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=42,stratify=y)

"""---

### Compreendendo a importância da normalização

Antes de seguirmos adiante com o treinamento do modelo KNN com os dados de câncer de mama, é importante entendermos como atributos que possuem valores em intervalos bem diferentes podem influenciar no processo de modelagem.

Para tanto, vamos fazer uma breve análise com dois atributos selecionados manualmente a partir da análise dos histogramas (mostrados mais acima). Escolha dois atributos que claramente possuem intervalos de valores bem diferentes entre si (por exemplo, radius_mean e smoothness_mean, ou qualquer outro par que preferir).
"""

## Defina nos campos ao lado o nome dos atributos para analisar
input_atr1 = "AddressA" #@param {type:"string"}
input_atr2 = "AddressB" #@param {type:"string"}

atr = np.argwhere(data.columns.isin([input_atr1,input_atr2])).ravel()
print(atr)

import random
## Selecionar aleatoriamente 60 instâncias para explorar a visualização da fronteira
## de decisão gerada pelo algoritmo KNN
ninstances_training = 60

selected_instances_training = list(random.sample(range(len(X_train)), ninstances_training))
X_train_subset = X_train[selected_instances_training, :]
y_train_subset = y_train[selected_instances_training]

## Analisar distribuição das instâncias entre as classes no conjunto de treino
unique, counts = np.unique(y_train_subset, return_counts=True)
dict(zip(unique, counts))

X_train_subset

"""Vamos analisar o resultado de um KNN aplicado sobre os dados originais, sem normalização. Para tanto, vamos observar o espaço de entrada e a distribuição espacial dos $k$ vizinhos mais próximos para as 4 primeiras instâncias de teste (apenas para fins de exemplo)."""

## Treinar um KNN (k=5) com atributos selecionados, SEM normalização

knn_nonorm = KNeighborsClassifier(n_neighbors=5,metric='euclidean',weights='uniform')
knn_nonorm.fit(X_train_subset[:,atr], y_train_subset)
k_index_nonorm = knn_nonorm.kneighbors(X_test[:,atr], n_neighbors=5, return_distance=False)

"""O código a seguir vai gerar um gráfico de dispersão para cada instância de teste, considerando os dois atributos selecionados acima. As instâncias de treino são representadas por azul claro, a instância de teste analisada em cada caso é representada em vermelho, e as instâncias que são selecionadas como os $k$ vizinhos mais próximos são destacadas em azul escuro."""

k=5
fig,ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
for ii in range(4):
    sample = X_test[ii,atr]
    neighbors = k_index_nonorm[ii]
    ax[ii].scatter(X_train_subset[:, atr[0]], X_train_subset[:, atr[1]], c="skyblue")
    ax[ii].scatter(X_train_subset[neighbors, atr[0]], X_train_subset[neighbors, atr[1]], edgecolor="darkblue")
    ax[ii].scatter(sample[0], sample[1], marker="+", c="red", s=100)
    ax[ii].set(xlabel=input_atr1)
    ax[ii].set(ylabel=input_atr2)

plt.tight_layout()

"""**Responda >>>** Descreva que padrão você observa na distribuição dos k-vizinhos mais próximos em relação aos eixos x e y.

> ***Sua resposta aqui:*** como o atributo 1 (radius) possui escala maior que o atributo 2 (smoothness), o atributo 1 domina no cálculo da distância entre os vizinhos, fazendo com que o algoritmo traga vizinhos com valores muito próximos de atributo 1.

Para comparar os resultados, vamos aplicar a normalização min-max vista em aula, e reexecutar a análise dos k-vizinhos mais próximos para as 4 primeiras instâncias de teste. A normalização é aplicada com o método `MinMaxScaler()` da biblioteca scikit-learn.
"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

## estima os parâmetros para normalização a partir dos dados de treino
scaler.fit(X_train_subset)

## aplica a normalização nos conjuntos de treino e teste
X_train_norm_subset = scaler.transform(X_train_subset)
X_test_norm = scaler.transform(X_test)

"""A seguir, os mesmos passos são aplicados para gerar o gráfico do espaço de entrada para as mesmas primeiras instâncias de teste analisadas no exemplo anterior. Entretanto, agora fazemos o gráfico usando os dados **normalizados**."""

## Treinar um KNN (k=5) com atributos selecionados, COM normalização
knn_norm = KNeighborsClassifier(n_neighbors=5,metric='euclidean',weights='uniform')
knn_norm.fit(X_train_norm_subset[:,atr], y_train_subset)
k_index_norm = knn_norm.kneighbors(X_test_norm[:,atr], n_neighbors=5, return_distance=False)
#print(k_index_norm[:5,])

k=5
fig,ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
for ii in range(4):
    sample = X_test_norm[ii,atr]
    neighbors = k_index_norm[ii]
    # print(neighbors)
    ax[ii].scatter(X_train_norm_subset[:, atr[0]], X_train_norm_subset[:, atr[1]], c="skyblue")
    ax[ii].scatter(X_train_norm_subset[neighbors, atr[0]], X_train_norm_subset[neighbors, atr[1]], edgecolor="darkblue")
    ax[ii].scatter(sample[0], sample[1], marker="+", c="red", s=100)
    ax[ii].set(xlabel=input_atr1)
    ax[ii].set(ylabel=input_atr2)

plt.tight_layout()

"""**Responda >>>** Descreva que alterações você observou no padrão de distribuição dos k-vizinhos mais próximos em relação aos eixos e à própria instância de teste (marcada em vermelho), ao comparar estes gráficos com os anteriores.

> ***Sua resposta aqui:*** quando os dados não estavam normalizados tinhamos agrupamentos de vizinhos enviezados por conta da escala dos atributos. Após a normalização temos agrupamentos em que cada atributo contribui igualmente.

---


###Avaliando o desempenho do modelo e impacto do valor de $k$


Nesta seção, vamos aplicar a estratégia de Holdout para treinar e **avaliar o desempenho** de modelos, e em seguida comparar o desempenho obtido para diferentes valores de $k$. O objetivo é poder selecionar, para um determinado conjunto de dados ou prolema de pesquisa, qual valor de $k$ gera o melhor modelo.

<br>

#### Normalizando os dados

Iniciamos aplicando a normalização sobre todos os dados de treinamento e de teste, separadamente. Utilizamos a mesma definição de treino e teste feita mais acima do notebook. Além disso, estamos usando todo o conjunto de instâncias e de atributos disponíveis no dataset.
"""

## Estima os parâmetros para normalização a partir dos dados de treino
scaler = MinMaxScaler()
scaler.fit(X_train)

## Aplica a normalização nos conjuntos de treino (X_train) e teste (X_test)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

"""#### Treinando e avaliando um modelo de KNN

Abaixo vamos treinar um modelo de KNN, e aplicar o modelo para prever a saída nos dados de teste. Observe que não estamos otimizando o valor do hiperparâmetro $k$: mantemos o valor de $k$ fixo e igual a 5.
"""

## Treina um modelo KNN a partir dos dados normalizados, usando k=5
## O algoritmo adota a distância euclidiana para seleção dos k-vizinhos
## mais próximos e assume uma votação uniforme entre eles (todos possuem o mesmo
## "peso" na decisão da classe para uma nova instância)
clf = KNeighborsClassifier(n_neighbors=5,metric='euclidean',weights='uniform')
clf.fit(X_train, y_train)


## Apĺica o modelo treinado para prever a saída dos dados de teste
y_pred = clf.predict(X_test)

## A variável y_pred contém a classe predita para cada instância de teste (0 ou 1)
print(y_pred)

## Também podemos extrair a probabilidade atribuída pelo modelo a cada classe
## Esta função retorna um array com vetor de probabilidades para cada instância de teste.
## Cada vetor possui um comprimento igual ao número de classes (neste caso, 2)
y_prob = clf.predict_proba(X_test)
print(y_prob[:20,]) # imprimindo apenas as 20 primeiras instâncias

"""Com `y_test` (classe real) e `y_pred` (classe predita), podemos fazer a análise de desempenho deste modelo. A biblioteca scikit-learn contém um amplo conjunto de métricas de desempenho implementadas e disponíveis para uso. Abaixo vamos utilizar apenas a matriz de confusão (que quantifica número de Falsos Positivos, Verdadeiros Positivos, Falsos Negativos e Verdadeiros Negativos) e as métricas já vistas em aula: acurácia, recall (sensibilidade) e precisão."""

from sklearn.metrics import confusion_matrix, recall_score, precision_score,accuracy_score,ConfusionMatrixDisplay
## Avaliando o desempenho do modelo usando a matriz de confusão, e três métricas
## de desempenho: acurácia, recall (sensibilidade) e precisão.

cm = confusion_matrix(y_test, y_pred,labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp = disp.plot(include_values=True, cmap='Blues', ax=None, xticks_rotation='horizontal')
plt.grid(False)
plt.show()

print(round(accuracy_score(y_test, y_pred),3))
print(round(recall_score(y_test, y_pred),3))
print(round(precision_score(y_test, y_pred),3))

"""**Responda >>>** Quantos FP, FN, VP e VN o modelo KNN com $k$=5 obteve? Considere a classe 1 (Maligno) como a classe positiva.

> ***Sua resposta aqui:*** FP=1, FN=4, VP=38, VN=71

#### Otimizando o valor de $k$

Para otimizar os hiperparâmetros de um algoritmo, precisamos definir um conjunto de validação. Evitamos, desta forma, um viés na avaliação de desempenho do modelo final ao usar o mesmo conjunto de holdout para escolher o melhor hiperparâmetro e estimar o poder de generalização deste modelo.

A seguir, vamos usar as mesmas funções do scikit-learn para fazer um split dos dados seguindo a estratégia de *3-way holdout*: primeiro faremos uma divisão em treino e teste, e em seguida dividiremos o treino em dois subconjuntos: treino e validação. O conjunto de validação será utilizado para a escolha do melhor valor de k. O melhor modelo serão, então, avaliado com o conjunto de teste. O uso do random_state viabiliza reprodutibilidade dos experimentos.
"""

## Definindo as proporções de treino, validação e teste.
train_ratio = 0.70
test_ratio = 0.15
validation_ratio = 0.15

## Fazendo a primeira divisão, para gerar o conjunto de teste.
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_ratio,stratify=y,random_state=42)

## Fazendo a segunda divisão, para gerar o conjunto de treino e validação a partir de X_train da divisão anterior
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=validation_ratio/(train_ratio+test_ratio),stratify=y_temp,random_state=42)

print(X_train.shape)
print(X_test.shape)
print(X_valid.shape)

## Normalizando os dados:
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

## Definindo um array para armazenar o recall de cada modelo treinado e avaliado
recall_valid = []
precision_valid=[]
accuracy_valid=[]

## Definindo kmin e kmax
k_min=1
k_max=51

# Calculando a acurácia para os modelos com k entre 1 e 45 (inclusive)
for ii in range(k_min,(k_max+1),2):
    knn = KNeighborsClassifier(n_neighbors=ii,metric='euclidean',weights='uniform')
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_valid)
    recall_valid.append(recall_score(y_valid, pred_i))
    precision_valid.append(precision_score(y_valid, pred_i))
    accuracy_valid.append(accuracy_score(y_valid, pred_i))

plt.figure(figsize=(12, 6))
plt.plot(range(k_min,(k_max+1),2), recall_valid, color='steelblue', linestyle='dashed', marker='o', markerfacecolor='darkblue', markersize=10)
plt.title('Recall vs K Value')
plt.xlabel('K Value')
plt.ylabel('Recall')

plt.figure(figsize=(12, 6))
plt.plot(range(k_min,(k_max+1),2), precision_valid, color='steelblue', linestyle='dashed', marker='o', markerfacecolor='darkblue', markersize=10)
plt.title('Precision vs K Value')
plt.xlabel('K Value')
plt.ylabel('Precision')

plt.figure(figsize=(12, 6))
plt.plot(range(k_min,(k_max+1),2), accuracy_valid, color='steelblue', linestyle='dashed', marker='o', markerfacecolor='darkblue', markersize=10)
plt.title('Accuracy vs K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')

"""**Responda >>>** Qual valor (ou valores) de $k$ apresentou melhor recall e qual foi o recall obtido neste caso? Modifique o código, repetindo a mesma análise para acurácia e precisão, e reporte os resultados

> ***Sua resposta aqui:*** pensando em um balanço entre as três métricas, K=11 é a que mais se aproxima de 100% em todas. Agora pensando em minimizar os falsos negativos (bastante indesejados nessa aplicação) o ideal seria maximizar o recall, onde K=9 entrega recall=100%.

---

## Entrega da atividade


Entregue a sua solução no Moodle, enviando o Colab Notebook com o seu código e suas análises para esta atividade, exportando-o da seguinte forma:

*   File > Download .ipynb
*   File > Print e então salve em **PDF**

---


## Sugestões de experimentos extras (**opcionais**):

**I)** Treine novos modelos com o algoritmo KNN para os dados de câncer de mama, avaliando o impacto sobre o desempenho do modelo ao mudar:

1.   o valor do hiperparâmetro *weights*, definindo-o como *'distance'*. Esta opção define que o peso na votação entre os k-vizinhos mais próximos não será mais uniforme, mais sim ponderado pela distância de cada k-vizinho mais próximo à nova instância.
2.   a medida de distância utilizada na escolha dos k-vizinhos mais próximos. Sugere-se testar a CanberraDistance ('canberra'). Veja outras opções de medidas de distância na [documentação](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html#sklearn.metrics.DistanceMetric) do scikit-learn (teste ao menos uma outra medida).

Para estes experimentos, selecione primeiramente o valor $k$ ótimo de acordo com a métrica de recall, e posteriormente avalie este melhor modelo nos dados de teste, comparando os resultados obtidos no exercício anterior e com estas variações no uso do KNN.


**II)** Verifique se há influência sobre o desempenho do modelo se você fizer um filtro dos atributos utilizados, reduzindo a dimensionalidade dos dados. Você pode, por exemplo, usar apenas atributos relacionados à média das características (ignorando desvio padrão e maior valor de cada atributo), ou se basear no gráfico com a análise de correlação para remover atributos muito correlacionados.
"""

def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.

    Inputs:
        x: features dataframe
        threshold: features with correlations greater than this value are removed

    Output:
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)

    return x

"""---


### Análise visual do impacto do valor $k$ (opcional)

Com os dados devidamente separados em treino e teste, vamos treinar um modelo de Aprendizado de Máquina utilizando o algoritmo KNN e o conjunto de treino. Nesta parte, nosso foco será em **analisar visualmente alterações na fronteira de decisão** do modelo em decorrência de diferentes valores de $k$. Ou seja, por enquanto não iremos avaliar o desempenho do modelo.  

Nas linhas a seguir, iremos selecionar 60 instâncias aleatórias do conjunto de treino para poder melhor inspecionar as fronteiras de decisão geradas pelo algoritmo KNN com diferentes configurações de $k$ e entender o funcionamento do algoritmo. Esta seleção é feita **somente para fins de visualização**. Posteriormente faremos experimentos com o conjunto de dados completo.

A função auxiliar a seguir é declarada para permitir treinar um KNN e visualizar sua fronteira de decisão, o que nos permitirá melhor compreender o impacto do valor de $k$. De forma bem simplificada, a função define uma região bidimensional em torno de **2 atributos** presentes nos dados (por padrão, os dois primeiros atributos do dataframe passado como argumento) e utiliza os dados de treinamento para criar o modelo de KNN. Em seguida, a função estima a probabilidade para cada ponto desta região utilizando o modelo treinado e desta forma, colore a região de acordo com a classe mais provável estimada pelo modelo.

Não se preocupe caso não compreenda em detalhes o funcionamento desta função, ela é apenas utilitária para a visualização dos dados.

Para este exercício, usaremos apenas 2 atributos para facilitar a visualização da fronteira de decisão em um espaço bidimensional. Como por padrão a função utiliza os dois primeiros atributos do dataset, estão sendo selecionados os atributos *radius_mean* e *texture_mean*.
"""

## Função para plotar a fronteira de decisão
## Fonte: https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Utilities/ML-Python-utils.py
def plot_decision_boundaries(X, y, model_class, **model_params):
    """
    Function to plot the decision boundaries of a classification model.
    This uses just the first two columns of the data for fitting
    the model as we need to find the predicted value for every point in
    scatter plot.
    Arguments:
            X: Feature data as a NumPy-type array.
            y: Label data as a NumPy-type array.
            model_class: A Scikit-learn ML estimator class
            e.g. GaussianNB (imported from sklearn.naive_bayes) or
            LogisticRegression (imported from sklearn.linear_model)
            **model_params: Model parameters to be passed on to the ML estimator

    Typical code example:
            plt.figure()
            plt.title("KNN decision boundary with neighbros: 5",fontsize=16)
            plot_decision_boundaries(X_train,y_train,KNeighborsClassifier,n_neighbors=5)
            plt.show()
    """
    try:
        X = np.array(X)
        y = np.array(y).flatten()
    except:
        print("Coercing input data to NumPy arrays failed")

    # Reduces to the first two columns of data - for a 2D plot!
    reduced_data = X[:, :2]
    # Instantiate the model object
    model = model_class(**model_params)
    # Fits the model with the reduced data
    model.fit(reduced_data, y)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 0.5, reduced_data[:, 0].max() + 0.5
    y_min, y_max = reduced_data[:, 1].min() - 0.5, reduced_data[:, 1].max() + 0.5
    # Meshgrid creation
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh using the model.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predictions to obtain the classification results
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plotting
    plt.contourf(xx, yy, Z, alpha=0.2,cmap='viridis')
    g=plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6,s=50, edgecolor='k', cmap='viridis' )
    plt.xlabel("Feature-1",fontsize=15)
    plt.ylabel("Feature-2",fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(handles=g.legend_elements()[0],labels=('benign','malign'))
    return plt

"""No código a seguir, vamos definir o número de $k$ vizinhos mais próximos para utilizar no treinamento do KNN. Execute o código para diferentes valores de N_NEIGHBORS, observando a variação na fronteira de decisão. Inicie com 1 vizinho mais próximo e vá gradativamente aumentando o número de vizinhos. Lembre-se que é importante evitar empates na decisão, portanto, para problemas binários o ideal é utilizar valores que não sejam múltiplos de 2. Teste também para o caso extremo, de N_NEIGHBORS = 60.

*OBS*: é esperado que a execução demore mais conforme você aumente o valor de N_NEIGHBORS.

"""

N_NEIGHBORS =  1 # N_NEIGHBORS: define o número de k-vizinhos mais próximos. Faça alterações aqui.
plt.figure(figsize=(10, 8))
plt.title("KNN: fronteira de decisão com {} vizinhos".format(N_NEIGHBORS),fontsize=16)
plot_decision_boundaries(X_train_subset,y_train_subset,KNeighborsClassifier,n_neighbors=N_NEIGHBORS,metric='euclidean',weights='uniform')
plt.show()

"""Reflita sobre as alterações visuais que você percebe na fronteira de decisão ao ir aumentando o valor de $k$ no código acima. Perceba aspectos como o quanto a fronteira de decisão incorpora algumas particularidades ou artefatos nos dados (como um exemplo que aparenta ser um ruído), o grau de suavidade da fronteira, o efeito de aumentar muito o valor de $k$, etc.
Você pode salvar as figuras geradas a cada execução, clicando com o botão direito do mouse sobre a figura e selecionado "Salvar imagem como".
"""