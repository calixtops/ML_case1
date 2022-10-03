#!/usr/bin/env python3
import pandas as pd 
# from src.utils_geofusion import rf_regressor
pd.options.mode.chained_assignment = None
import numpy as np
import src.utils as gf
from sklearn.impute import SimpleImputer


# Leitura e pre-processamento do dataset
file = 'data/DadosDesafioCientista.xlsx'
dataset = pd.read_excel(file,na_values = '-')

## SimpleImputer para preencher valores NaN
s_imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
s_imputer = s_imputer.fit(dataset.iloc[:,4:-2])
dataset.iloc[:,4:-2] = s_imputer.transform(dataset.iloc[:,4:-2])

###Escolhendo hiperparametros utilizando GridSearchCV

uf_pred = 'RJ'

param_grid_RFR = { 'n_estimators'     : [10,50,100,200,300, 400, 500],
    			   'criterion'        : ['mse', 'mae'],
                   'max_features'     : ['auto', 'sqrt'],
                   'max_depth'        : [10, 35, 60, 85, 110],
                   'min_samples_split': [2, 5, 10],
                   'min_samples_leaf' : [1, 2, 4],
                   'bootstrap'        : [True, False]}


best_params_RFR = gf.train_rfr_param(dataset, uf_train = 'RJ', type_search = 'Random', param_grid = param_grid_RFR)



# Após encontrar os melhores hiper parametros para nosso modelo vamos aplicar o Random Forest Regressor

dataset = gf.rf_regressor(dataset,uf_train='RJ',uf_pred='SP', best_params = best_params_RFR)

####### Estimando hiperparametros para a Classificação

param_grid_RFC =  {'n_estimators'  : [50, 100, 200, 300, 400, 500],
                   'criterion'        : ["gini", "entropy"],
                   'max_depth'        : [10, 35, 60, 85, 110],
                   'min_samples_split': [2, 5, 10],
                   'min_samples_leaf' : [1, 2, 4],
                   'bootstrap'        : [True, False]}



best_params_rfc = gf.train_rfc_param(dataset, uf_train = 'RJ', type_search = 'Random', param_grid = param_grid_RFC)


#### Classificando o potencial utilizando random forest classifier

dataset = rf_classifier(dataset,uf_train='RJ',uf_pred='SP', best_params = best_params_rfc)

#####
# Salvando dataset preenchido
dataset.to_csv('data/DadosDesafioCientista_full.csv', index = False)
# dataset.to_csv('../dashboard/src/DadosDesafioCientista_full.csv', index = False)


