import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

data_test = pd.read_csv('./Data/conjunto_de_teste.csv')
testId = data_test["id_solicitante"]
data = pd.read_csv('./Data/conjunto_de_treinamento.csv')

###########################################################################
## Arrumando os dados de ajuste do classificador
###########################################################################
## Eliminando colunas que julguei pouco eficientes
###########################################################################
x = data.drop(columns = 
              ["inadimplente",
               "id_solicitante",
               "estado_onde_nasceu",
               "possui_email",
               "possui_telefone_trabalho",
               "codigo_area_telefone_trabalho",
               'tipo_endereco',
               "grau_instrucao",
               "possui_telefone_celular",
               "qtde_contas_bancarias_especiais",
               "meses_no_trabalho",
               "estado_onde_reside"])

y = data["inadimplente"]


###########################################################################
## Transformando colunas com valores do tipo string em int
###########################################################################
x_encoded = x.copy()
# Iterando pelas colunas de x
for col in x.columns:
    # Verifica se a coluna possui valores de string
    if x[col].dtype == 'object':

        # Cria um objeto LabelEncoder
        label_encoder = LabelEncoder()

        # Aplica o LabelEncoder na coluna
        x_encoded[col] = label_encoder.fit_transform(x[col])

###########################################################################
## Arrumando colunas com valores vazios
###########################################################################
imputer = SimpleImputer(strategy='most_frequent')
x_imputed = imputer.fit_transform(x_encoded)

###########################################################################
## Arrumando a escala dos valores
###########################################################################
StdSc = StandardScaler()
StdSc = StdSc.fit(x_imputed)
x_transformed = StdSc.transform(x_imputed)

###########################################################################
## Repetindo o mesmo processamento de dados para os dados de teste
###########################################################################
x_test = data_test.drop(columns = 
              ["id_solicitante",
               "estado_onde_nasceu",
               "possui_email",
               "possui_telefone_trabalho",
               "codigo_area_telefone_trabalho",
               'tipo_endereco',
               "grau_instrucao",
               "possui_telefone_celular",
               "qtde_contas_bancarias_especiais",
               "meses_no_trabalho",
               "estado_onde_reside"])

x_encoded_test = x_test.copy()

for col in x_test.columns:
    if x_test[col].dtype == 'object':

        label_encoder = LabelEncoder()

        x_encoded_test[col] = label_encoder.fit_transform(x_test[col])

imputer = SimpleImputer(strategy='most_frequent')
x_imputed_test = imputer.fit_transform(x_encoded_test)

StdSc = StandardScaler()
StdSc = StdSc.fit(x_imputed_test)
x_transformed_test = StdSc.transform(x_imputed_test)

###########################################################################
## Fazendo testes de desempenho
###########################################################################
x_train2, x_test2, y_train2, y_test2 = train_test_split(x_transformed, y, test_size=0.7)

#testClassifier = RandomForestClassifier()
#testClassifier.fit(x_train2, y_train2)

#testPredictions = testClassifier.predict(x_test2)
#testScore = accuracy_score(y_test2, testPredictions)
#print("testScore was: ", testScore)

#parameters = {
    #"max_depth":[8],
    #"n_estimators": [500],
    #"min_samples_leaf":[10]
    #}

#gridRF = GridSearchCV(RandomForestClassifier(), parameters, cv=10, verbose=2, n_jobs = -1)
#gridRF.fit(x_train2, y_train2)

#bestRF = gridRF.best_estimator_
#bestRF.fit(x_train2,y_train2)
#respostaRF = bestRF.predict(x_test2)
#testScore2 = accuracy_score(y_test2, respostaRF)
#print("SecondTestScore was: ", testScore2)
###########################################################################
## Treinando o classificador 
###########################################################################
parameters = {
    "max_depth":[8],
    "n_estimators": [500],
    "min_samples_leaf":[10]
    }

gridRF = GridSearchCV(RandomForestClassifier(), parameters, n_jobs = -1)
gridRF.fit(x_train2, y_train2)

bestRF = gridRF.best_estimator_
bestRF.fit(x_transformed, y)

###########################################################################
## Prevendo os resultados e colocando eles no arquivo de resultados
###########################################################################
predictions = bestRF.predict(x_transformed_test)

prediction_file = pd.DataFrame(predictions, columns=['inadimplente'])
prediction_file = pd.concat([testId, prediction_file], axis=1)
prediction_file = prediction_file.to_csv('./Data/Resultados.csv', index=False)


prediction_file = pd.read_csv('./Data/Resultados.csv')
prediction_file.shape
