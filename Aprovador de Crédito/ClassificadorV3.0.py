import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer

data_test = pd.read_csv('./Data/conjunto_de_teste.csv')
testId = data_test["id_solicitante"]
data = pd.read_csv('./Data/conjunto_de_treinamento.csv')

print(data.iloc[18548])
print(data.iloc[4406])
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
# Ocupação --> preencher campos vazios

###########################################################################
## Removendo Variaveis
###########################################################################

data = data.drop([18548, 4406], axis = 0)

x = data.drop(columns = 
              [#"inadimplente",
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
               "local_onde_trabalha"], axis=1)

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

print(x.groupby(["inadimplente"]).mean().T)

cores = ["red" if b==1 else "blue" for b in x["inadimplente"]]

grafico = x.plot.scatter(
    "idade",
    "renda_extra",
    #"valor_patrimonio_pessoal",
    #"meses_na_residencia",
    c = cores,
    s = 5,
    marker = "o",
    alpha = 0.5,
    figsize = (10,5)
    )
###########################################################################
## Arrumando colunas com valores vazios
###########################################################################
#imputer = SimpleImputer(strategy='most_frequent')
#x = imputer.fit_transform(x)

#print(x.T)

###########################################################################
## Arrumando a escala dos valores
###########################################################################
#StdSc = StandardScaler()
#StdSc = StdSc.fit(x_imputed)
#x_transformed = StdSc.transform(x_imputed)

#print("\n dados transformados:")
#print(x_transformed)
###########################################################################
## Repetindo o mesmo processamento de dados para os dados de teste
###########################################################################
#x_test = data_test.drop(columns = 
 #             ["id_solicitante",
  #             "estado_onde_nasceu",
   #            "possui_email",
    #           "possui_telefone_trabalho",
     #          "codigo_area_telefone_trabalho",
      #         'tipo_endereco',
       #        "grau_instrucao",
        #       "possui_telefone_celular",
         #      "qtde_contas_bancarias_especiais",
          #     "meses_no_trabalho",
           #    "estado_onde_reside"])

#x_encoded_test = x_test.copy()

#for col in x_test.columns:
 #   if x_test[col].dtype == 'object':

  #      label_encoder = LabelEncoder()

   #     x_encoded_test[col] = label_encoder.fit_transform(x_test[col])

#imputer = SimpleImputer(strategy='most_frequent')
#x_imputed_test = imputer.fit_transform(x_encoded_test)

##StdSc = StandardScaler()
#StdSc = StdSc.fit(x_imputed_test)
#x_transformed_test = StdSc.transform(x_imputed_test)

###########################################################################
## Fazendo testes de desempenho
###########################################################################
#x_train2, x_test2, y_train2, y_test2 = train_test_split(x_transformed, y, test_size=0.7)

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
#parameters = {
 #   "max_depth":[8],
  #  "n_estimators": [500],
   # "min_samples_leaf":[10]
    #}

#gridRF = GridSearchCV(RandomForestClassifier(), parameters, n_jobs = -1)
#gridRF.fit(x_train2, y_train2)

#bestRF = gridRF.best_estimator_
#bestRF.fit(x_transformed, y)

###########################################################################
## Prevendo os resultados e colocando eles no arquivo de resultados
###########################################################################
#predictions = bestRF.predict(x_transformed_test)

#prediction_file = pd.DataFrame(predictions, columns=['inadimplente'])
#prediction_file = pd.concat([testId, prediction_file], axis=1)
#prediction_file = prediction_file.to_csv('./Data/Resultados.csv', index=False)


#prediction_file = pd.read_csv('./Data/Resultados.csv')
#prediction_file.shape
