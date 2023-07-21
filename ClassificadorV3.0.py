import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

data_teste = pd.read_csv('./Data/conjunto_de_teste.csv')
testId = data_teste["id_solicitante"]
data = pd.read_csv('./Data/conjunto_de_treinamento.csv')

print(data.groupby(["inadimplente"]).mean().T)
###########################################################################
## Arrumando os dados de ajuste do classificador
###########################################################################
### Avaliando o que fazer com cada coluna

# Id --> Remover, não agrega informação útil
# Grau de instrução --> Remover, todas as celulas tem o mesmo valor
# Estado onde nassceu --> Remover, cardinalidade alta
# Estado onde reside --> Remover, cardinalidade alta
# codigo de area telefone residencial --> Remover, cardinalidade alta
# qqtd contas bancarias especiais --> Remover, pouca informação extra
# estado onde trabalha --> Remover, muitos campos em branco
# codigo de area telefone trabalho --> Remover, cardinalidade alta
# grau de instrução do companheiro --> Remover, muitos campos em branco
# Profissão companheiro --> Remover, muitos campos em branco
# Local onde trabalha --> Remover, valores identicos ao local onde reside

# Forma de envio --> OneHotEncoding

# Sexo -- binarizar
# Possui telefone --> binarizar
# Possui telefone celular --> binarizar
# vinculo formal com a empresa --> binarizar
# Possui telefone de trabalho --> binarizar


# Profissao --> preencehr campos vazios

# Ocupação -->
# tipo_endereco --> Pouca informação util
# nacionalidade
# possui_email
# possui_cartao_visa
###########################################################################
## Removendo Variaveis
###########################################################################

#data = data.drop([18548, 4406], axis = 0)

x = data.drop(columns = 
              ["inadimplente",
               "id_solicitante",
               "grau_instrucao",
               "estado_onde_nasceu",
               "estado_onde_reside",
               "codigo_area_telefone_residencial",
               "qtde_contas_bancarias_especiais",
               "estado_onde_trabalha",
               "codigo_area_telefone_trabalho",
               "grau_instrucao_companheiro",
               "profissao_companheiro",
               "local_onde_trabalha",
               "tipo_endereco",
               "nacionalidade",
               "possui_email",
               "possui_cartao_visa",
               "possui_cartao_mastercard",
               "possui_cartao_diners",
               "possui_cartao_amex",
               "possui_outros_cartoes",
               "ocupacao"], axis=1)

y = data["inadimplente"]

###########################################################################
### Aplicandoo One Hot Encoding
###########################################################################

x = pd.get_dummies(x, columns = ["forma_envio_solicitacao"])
print(x.T)

###########################################################################
### Aplicando binarização
###########################################################################

for coluna in ["possui_telefone_residencial", 
          "possui_telefone_celular", 
          "vinculo_formal_com_empresa", 
          "possui_telefone_trabalho",
          "sexo"]:
    binarizador = LabelEncoder()
    x[coluna] = binarizador.fit_transform(x[coluna])

print( x.T)

###########################################################################
### Analiando os dados
###########################################################################

#print(x.groupby(["inadimplente"]).mean().T)

#cores = ["red" if b==1 else "blue" for b in x["inadimplente"]]

#grafico = x.plot.scatter(
 #   "idade",
  #  "renda_extra",
    #"valor_patrimonio_pessoal",
    #"meses_na_residencia",
   # c = cores,
    #s = 5,
    #marker = "o",
    #alpha = 0.5,
    #figsize = (10,5)
    #)
###########################################################################
## Arrumando colunas com valores vazios
###########################################################################

imputer = SimpleImputer(strategy='most_frequent')
x = imputer.fit_transform(x)

###########################################################################
## Arrumando a escala dos valores
###########################################################################
StdSc = StandardScaler()
StdSc = StdSc.fit(x)
x = StdSc.transform(x)

###########################################################################
## Repetindo o mesmo processamento de dados para os dados de teste
###########################################################################
#data_teste = data_teste.drop([18548, 4406], axis = 0)

x_teste = data_teste.drop(columns = 
              ["id_solicitante",
               "grau_instrucao",
               "estado_onde_nasceu",
               "estado_onde_reside",
               "codigo_area_telefone_residencial",
               "qtde_contas_bancarias_especiais",
               "estado_onde_trabalha",
               "codigo_area_telefone_trabalho",
               "grau_instrucao_companheiro",
               "profissao_companheiro",
               "local_onde_trabalha",
               "tipo_endereco",
               "nacionalidade",
               "possui_email",
               "possui_cartao_visa",
               "possui_cartao_mastercard",
               "possui_cartao_diners",
               "possui_cartao_amex",
               "possui_outros_cartoes",
               "ocupacao"], axis=1)

x_teste = pd.get_dummies(x_teste, columns = ["forma_envio_solicitacao"])

for coluna in ["possui_telefone_residencial", 
          "possui_telefone_celular", 
          "vinculo_formal_com_empresa", 
          "possui_telefone_trabalho",
          "sexo"]:
    binarizador = LabelEncoder()
    x_teste[coluna] = binarizador.fit_transform(x_teste[coluna])
    
imputer = SimpleImputer(strategy='most_frequent')
x_teste = imputer.fit_transform(x_teste)

StdSc = StandardScaler()
StdSc = StdSc.fit(x_teste)
x_teste = StdSc.transform(x_teste)

###########################################################################
## Fazendo testes de desempenho
###########################################################################
#x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.7)

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
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.7)

#clfGB = GradientBoostingClassifier(n_estimators=80, max_depth=6, random_state=1500)
#pred = clfGB.fit(x_train2, y_train2).predict(x_test2)
#print(accuracy_score(y_test2,pred))

parametersGB = {
    "max_depth":[6],
    "n_estimators": [80],
    "random_state":[1500]
    }

gridGB = GridSearchCV(GradientBoostingClassifier(), parametersGB, cv=10, n_jobs = -1)
gridGB.fit(x_train2, y_train2)

bestGB = gridGB.best_estimator_
bestGB.fit(x, y)

####

#parametersRF = {
    #"max_depth":[8],
    #"n_estimators": [500],
    #"min_samples_leaf":[10]
    #}

#gridRF = GridSearchCV(RandomForestClassifier(), parametersRF, cv=10, n_jobs = -1)
#gridRF.fit(x_train2, y_train2)

#bestRF = gridRF.best_estimator_
#bestRF.fit(x, y)

###########################################################################
## Prevendo os resultados e colocando eles no arquivo de resultados
###########################################################################
predictions = bestGB.predict(x_teste)

prediction_file = pd.DataFrame(predictions, columns=['inadimplente'])
prediction_file = pd.concat([testId, prediction_file], axis=1)
prediction_file = prediction_file.to_csv('./Data/ResultadosV3.csv', index=False)


#prediction_file = pd.read_csv('./Data/Resultados.csv')
#prediction_file.shape
