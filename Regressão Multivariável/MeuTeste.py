import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import boxcox
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

######################################################
# Lendo os dados
######################################################

dadosTreinamento = pd.read_csv('./Dados/conjunto_de_treinamento.csv')
dadosTeste = pd.read_csv('./Dados/conjunto_de_teste.csv')

######################################################
# Arrumando os dados para o ajuste do regressor
######################################################

x_treinamento =dadosTreinamento.drop(columns = 
                                     ['preco',
                                      'diferenciais', #Mesmas informações que tenho em outras colunas
                                      'bairro' #Muito grande para fazer o OneHotEncoding
                                      ])

y_treinamento = dadosTreinamento['preco']

######################################################
# Aplicando o One Hot Enconder para tranformar strings em inteiros
######################################################
for col in x_treinamento.columns:
    # Verifica se a coluna possui valores de string
    if x_treinamento[col].dtype == 'object':

        # Cria um objeto LabelEncoder
        label_encoder = LabelEncoder()

        # Aplica o LabelEncoder na coluna
        x_treinamento[col] = label_encoder.fit_transform(x_treinamento[col])

######################################################
# Aplicando boxcox para ajutar a gausiana dos dado de cada coluna
######################################################

#for coluna in x_treinamento.columns:
    #x_treinamento[coluna] = boxcox(x_treinamento[coluna])
    
######################################################
# Aplicando PolynomialFeatures
######################################################

#polyFeat = PolynomialFeatures(degree=2)
#xPoly = PolynomialFeatures.fit_transform(x_treinamento)
#xPoly = PolynomialFeatures.tranform(x_treinamento)

######################################################
# Arrumando a escala dos valores
######################################################

######################################################
# Fazendo testes de desempenho
######################################################

x_train, x_test, y_train, y_test = train_test_split(x_treinamento, y_treinamento, test_size=0.7)

rfRegressor = RandomForestRegressor()
rfRegressor.fit(x_train, y_train)

rfPredictions = rfRegressor.predict(x_test)
rfError = mean_squared_error(y_test, rfPredictions)
rfScore = r2_score(y_test, rfPredictions)
print("Random Forest Error was: ", rfError)
print("Random Forest Score was: ", rfScore)

knnRegressor = KNeighborsRegressor()
knnRegressor.fit(x_train, y_train)

knnPredictions = knnRegressor.predict(x_test)
knnError = mean_squared_error(y_test, knnPredictions)
knnScore = r2_score(y_test, knnPredictions)
print("KNN Error was: ", knnError)
print("KNN Score was: ", knnScore)