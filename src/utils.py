#!/usr/bin/env python3
import math
import scipy as sp
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from babel.numbers import format_currency, format_decimal
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, accuracy_score 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore")
sns.set()


def train_rfr_param(dataset, uf_train, type_search = 'Random', 
					param_grid = { 'n_estimators'     : [300, 400, 500],
				    			   'criterion'        : ['mse', 'mae'],
				                   'max_features'     : ['auto', 'sqrt'],
				                   'max_depth'        : [10, 35, 60, 85, 110],
				                   'min_samples_split': [2, 5, 10],
				                   'min_samples_leaf' : [1, 2, 4],
				                   'bootstrap'        : [True, False]}):

	'''

	train_rfr_param

	Função que utiliza o GridSearchCV para encontrar os melhores hiperparametros 
	para utilizar no Random Forest Regressor.


	Parametros
	----------

	dataset : dataset, default = none

	uf_train: str, default = 'RJ'

	    Escolha o estado que vai serveir para treinar o modelo.
	    Estado com faturamento já conhecido.

	type_search : str, default = 'Random'

	Define entres os metodos GridSearchCV e RandomizedSearchCV


	param_grid: dict, default = { 'n_estimators'     : 400,
								   'criterion'        : 'mse',
					               'max_features'     : 'sqrt',
					               'max_depth'        : 110,
					               'min_samples_split': 2,
					               'min_samples_leaf' : 1,
					               'bootstrap'        : False}

	dicionario com os parametros que voce deseja testar no GridSearchCV

	Returns
	-------
	grid_search.best_params_
		Dicionario com os hiperparametros otimos para utilização.

	'''


	df_train = dataset[dataset['estado'] == uf_train]


	X_train = df_train.iloc[:,4:-2].values
	y_train = df_train.iloc[:,-2].values

	X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.25, random_state = 1)


	rf = RandomForestRegressor(random_state=11)

	if type_search == 'Grid':
		search = GridSearchCV(estimator = rf, param_grid = param_grid, 
									cv = KFold(n_splits = 5), n_jobs = -1, verbose = 2, scoring = 'r2')
		search.fit(X_train, y_train)
	if type_search == 'Random':
		search = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, n_iter = 100, 
										cv = KFold(n_splits = 5), verbose=2, random_state=11, n_jobs = -1, scoring = 'r2')
		search.fit(X_train, y_train)

	RFR = RandomForestRegressor(**search.best_params_, random_state=11)
	RFR.fit(X_train,y_train)

	y_train_pred = RFR.predict(X_train)
	y_test_pred = RFR.predict(X_test)

	r2_train = r2_score(y_train, y_train_pred)
	r2_test =r2_score(y_test, y_test_pred)

	# print('------- Métricas ------------- ')
	print('R2 Train Score : {:.1%}'.format(r2_train))
	print('R2 Test Score : {:.1%}'.format(r2_test))
	print('r2 Score Search : {:.1%}'.format(search.best_score_))

	print('-------------------------------')

	return search.best_params_


def rf_regressor(dataset,uf_train='RJ',uf_pred='SP', 
				best_params = { 'n_estimators'  : [50, 100, 200, 300, 400, 500],
					               'criterion'        : ["gini", "entropy"],
					               'max_depth'        : [10, 35, 60, 85, 110],
					               'min_samples_split': [2, 5, 10],
					               'min_samples_leaf' : [1, 2, 4],
					               'bootstrap'        : [True, False]}):

	'''
	rf_regressor

	Função que utiliza Random Forest Regressor para estimar faturamento e potencial 
	de um municipio com base em outro. 
	Desenvolvida para estimar o faturamento dos bairros da cidade de São Paulo 
	a partir do faturamento conhecido dos bairros do Rio de Janeiro. 


	Parametros
	----------

	dataset : dataset, default = none

	uf_train: str, default = 'RJ'

	    Escolha o estado que vai serveir para treinar o modelo.
	    Estado com faturamento já conhecido.

	uf_pred: str, default = 'SP'

	    Estado que que vai ser estimado o faturamento a partir de uf_train

	n_estimators : int, default=100
	    O numero de arvores da floresta.

	Returns
	-------
	dataset
		dataset com os valores estimados para uf_pred.

	'''
	df_train = dataset[dataset['estado'] == uf_train]

	X_train = df_train.iloc[:,4:-2].values
	y_train = df_train.iloc[:, -2].values

	print('Treinando modelo Random Forest Regressor...')
	regressor = RandomForestRegressor(**best_params)
	regressor.fit(X_train, y_train)

	df_pred = dataset[dataset['estado'] == uf_pred]

	X_pred = df_pred.iloc[:,4:-2].values
	y_pred = regressor.predict(X_pred)

	df_pred.iloc[:,-2] = y_pred
	dataset = pd.concat([df_train,df_pred])

	dataset[dataset.select_dtypes('float64').columns].applymap(int)
	print('Regressão realizada!!')


	return dataset



def train_rfc_param(dataset, uf_train = 'RJ', type_search = 'Random', 
					  param_grid ={'n_estimators'  : [50, 100, 200, 300, 400, 500],
					               'criterion'        : ["gini", "entropy"],
					               'max_depth'        : [10, 35, 60, 85, 110],
					               'min_samples_split': [2, 5, 10],
					               'min_samples_leaf' : [1, 2, 4],
					               'bootstrap'        : [True, False]}):


	'''
	train_rfc_param

	Função que utiliza o GridSearchCV ou Randomized Search para encontrar os melhores hiperparametros 
	para utilizar no Random Forest Classifier.


	Parametros
	----------

	dataset : dataset, default = none

	uf_train: str, default = 'RJ'

	    Escolha o estado que vai serveir para treinar o modelo.
	    Estado com faturamento já conhecido.

	param_grid: dict, default  = {'n_estimators'      : [50, 100, 200, 300, 400, 500],
				               'criterion'        : ["gini", "entropy"],
				               'max_depth'        : [10, 35, 60, 85, 110],
				               'min_samples_split': [2, 5, 10],
				               'min_samples_leaf' : [1, 2, 4],
				               'bootstrap'        : [True, False]}

	dicionario com os parametros que voce deseja testar no GridSearchCV

	Returns
	-------
	best_params
		Dicionario com os hiperparametros otimos para utilização.

	'''

	df_train = dataset[dataset['estado'] == uf_train]
	df_train = dataset[dataset['estado'] == uf_train]
	df_train = pd.get_dummies(df_train, columns=['potencial'])

	X_train = df_train.iloc[:,4:].values
	y_train = df_train.iloc[:, -3::].values

	X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.25, random_state = 1)

	RFC = RandomForestClassifier(random_state=11)

	if type_search == 'Grid':
	    search = GridSearchCV(estimator = RFC, param_grid = param_grid, cv = KFold(n_splits = 5), n_jobs = -1, verbose = 2)
	    search.fit(X_train, y_train)
	if type_search == 'Random':
	    search = RandomizedSearchCV(estimator = RFC, param_distributions = param_grid, n_iter = 100, cv = KFold(n_splits = 5), verbose=2, random_state=11, n_jobs = -1)
	    search.fit(X_train, y_train)

	RFC = RandomForestClassifier(**search.best_params_,random_state=11)
	RFC.fit(X_train,y_train)

	y_train_pred = RFC.predict(X_train)
	y_test_pred = RFC.predict(X_test)

	a_score_train = accuracy_score(y_train, y_train_pred)
	a_score_test =accuracy_score(y_test, y_test_pred)

	print('------- Métricas ------------- ')
	print('Train Accuracy Score : {:.1%}'.format(a_score_train))
	print('Test  Accuracy Score : {:.1%}'.format(a_score_test))
	print('r2 score search : {:.1%}'.format(search.best_score_))

	return search.best_params_


def rf_classifier(dataset,uf_train='RJ',uf_pred='SP', 
				best_params = {'n_estimators': 500,
								 'min_samples_split': 2,
								 'min_samples_leaf': 1,
								 'max_depth': 110,
								 'criterion': 'gini',
								 'bootstrap': True}
								):




	'''
	rf_regressor

	Função que utiliza Random Forest Regressor para estimar faturamento e potencial 
	de um municipio com base em outro. 
	Desenvolvida para estimar o faturamento dos bairros da cidade de São Paulo 
	a partir do faturamento conhecido dos bairros do Rio de Janeiro. 


	Parametros
	----------

	dataset : dataset, default = none

	uf_train: str, default = 'RJ'

	    Escolha o estado que vai serveir para treinar o modelo.
	    Estado com faturamento já conhecido.

	uf_pred: str, default = 'SP'

	    Estado que que vai ser estimado o faturamento a partir de uf_train

	n_estimators : int, default=100
	    O numero de arvores da floresta.

	Returns
	-------
	dataset
		dataset com os valores estimados para uf_pred.

	'''


	df_train = dataset[dataset['estado'] == uf_train]
	df_train = pd.get_dummies(df_train, columns=['potencial'])

	X_train = df_train.iloc[:,4:-3].values
	y_train = df_train.iloc[:, -3::].values

	print('Treinando modelo Random Forest Classifier...')
	RFC = RandomForestClassifier(**best_params_rfc)
	RFC.fit(X_train, y_train)

	df_pred = dataset[dataset['estado'] == uf_pred]

	X_pred = df_pred.iloc[:,4:-1].values
	y_pred = pd.DataFrame(RFC.predict(X_pred), columns = df_train.iloc[:, -3::].columns, index = df_pred.index)


	df_pred = pd.concat([df_pred, y_pred], axis = 1).drop('potencial', axis =1)

	dataset = pd.concat([df_train,df_pred])

	dataset['potencial'] = dataset.iloc[:,-3::].idxmax(axis=1)

	dataset.potencial = dataset.potencial.apply(lambda x: x.replace('potencial_',''))
	dataset = dataset.drop(['potencial_Alto', 'potencial_Baixo', 'potencial_Médio'], axis = 1)

	return dataset
















































def displot_UF(dataset, x="popDe25a49", 
				xlabel='População de 25 a 49 anos', 
				figname='distplot_pop', 
				bg_color = '#d3dbfe'):


	'''
	displot_UF

	Plot de distribuição. 


	Parametros
	----------

	dataset : dataset, default = none

	x: str, default = 'popDe25a49'

	    Variavel eixo X

	xlabel: str, default = 'População de 25 a 49 anos'
	    
		Defina rotulo do eixo X

	figname: str, default = 'histplot'

		Caminho de onde será salvo a figura


	bg_color: str, default = '#d3dbfe'

	    Cor de fundo do gráfico


	'''

	fig, ax1 = plt.subplots(figsize=(16,10))

	# fig.set_facecolor('#d3dbfe')
	hue_order = ['Alto', 'Médio', 'Baixo']
	g = sns.displot(
			    data=dataset, x=x, 
			    hue="potencial",
			    kind="hist", hue_order=hue_order, 
			    kde=True,
				multiple = 'stack',
				palette=['#8080ff','#66ff66','#ff8080'],

			)

	g._legend.set_title('Potencial')
	plt.xlabel(xlabel,fontsize=14,weight='bold')
	plt.ylabel('Ocorrência',fontsize=14,weight='bold')
	plt.xticks(fontsize=12)
	plt.savefig('figures/{0}'.format(figname),dpi=600, facecolor=bg_color)



def relplot(dataset, x="popDe25a49", 
			y = 'domiciliosAB',
			hue = 'potencial',
			size = 'faturamento',
			xlabel='População de 25 a 49 anos', 
			ylabel = 'Domicílios Classe A e B',
			figname='relplot', bg_color = '#d3dbfe'):


	'''
	relplot

	Constroi plots relacionais em um FacetGrid

	Parametros
	----------

	dataset : dataset, default = none

	x,y: Vetor ou chaves em "dataset", default = (x = ['popDe25a49'], y = ['domiciliosAB']

	    Variáveis que especificam as posições nos eixos x e y.

	hue: Vetor ou chave em "dataset", default = ['potencial']

		Variável de agrupamento que irá produzir elementos com cores diferentes. 
		Pode ser categórica ou numérica, embora o mapeamento de cores comportar-se 
		de forma diferente neste último caso.

	size: Vetor ou chave em "dataset", default = ['faturamento']

		Variável de agrupamento que produzirá elementos com tamanhos diferentes.
		Pode ser categórica ou numérica, embora o mapeamento de tamanhos
		comportar-se de forma diferente neste último caso.

	xlabel: str, default = 'População de 25 a 49 anos'
	    
		Defina rotulo do eixo X

	figname: str, default = 'relplot'

		Caminho de onde será salvo a figura


	bg_color: str, default = '#d3dbfe'

	    Cor de fundo do gráfico


	'''

	fig, ax1 = plt.subplots(figsize=(16,10))

	g = sns.relplot(
	    data=dataset,
	    x=x, y=y,
	    hue=hue, size=size,
	    hue_order=['Alto','Médio','Baixo'],
	    palette=['#8080ff','#66ff66','#ff8080'],
	    sizes=(10, 200),
	)

	g.set(xscale="log", yscale="log")
	g.ax.xaxis.grid(True, "minor", linewidth=.25)
	g.ax.yaxis.grid(True, "minor", linewidth=.25)
	g.despine(left=True, bottom=True)
	plt.xlabel(xlabel,fontsize=14,weight='bold')
	plt.ylabel(ylabel,fontsize=14,weight='bold')
	plt.savefig('figures/{0}'.format(figname),dpi=600, facecolor=bg_color)





def ratio_dataset(dataset,pop_interesse = ['popDe25a34','popDe35a49'], 
				dom_interesse=['domiciliosA1', 'domiciliosA2','domiciliosB1', 'domiciliosB2']):


	'''
	ratio_dataset

	Soma campos de interesse e os converte de valores absolutos para porcentagem


	Parametros
	----------

	dataset : dataset, default = none

	X: str, default = 'popDe25a49'

	    Variavel eixo X

	xlabel: str, default = 'População de 25 a 49 anos'
	    
		Defina rotulo do eixo X

	figname: str, default = 'histplot'

		Caminho de onde será salvo a figura


	bg_color: str, default = '#d3dbfe'

	    Cor de fundo do gráfico


	Returns
	-------
	r_dataset
		dataset com valores em porcentagem
	dataset
		dataset com valores absolutos

	'''

	domicilios = [col for col in dataset if col.startswith('domicili')]
	pop = [col for col in dataset if col.startswith('popD')]

	dataset['popDe25a49'] = dataset[pop_interesse].sum(axis=1)
	dataset['domiciliosAB'] = dataset[dom_interesse].sum(axis=1)
	dataset['domiciliosTotal'] = dataset[domicilios].sum(axis=1)

	r_dataset = dataset.copy()
	r_dataset[pop] = (dataset[pop].div(dataset['população'],axis=0) * 100).round()

	r_dataset['popDe25a49'] = ((dataset['popDe25a49'] / dataset['população']) *100).round()
	r_dataset['domiciliosAB'] = ((dataset['domiciliosAB'] / dataset['domiciliosTotal']) *100).round()


	return r_dataset, dataset

def format_dataset(dataset, percent_cols = None, currency_cols = None, decimal_cols = None):


	'''
	format_dataset

	Formata os campo de porcentagem, monetário e decimal


	Parametros
	----------

	dataset : dataset, default = none

		dataset a ser formatado 

	percent_cols: list(str), default = Nome

	    lista de campos com valores percentuais

	currency_cols: list(str), default = Nome

	    lista de campos com valores monetarios

	decimal_cols: list(str), default = Nome

	    lista de campos com valores decimais

	Returns
	-------
	dataset
		dataset com campos formatados

	'''


	if percent_cols:
		for pcol in percent_cols:
			dataset[pcol] = dataset[pcol].apply(lambda x: '{}%'.format(int(x)))
	if currency_cols:
		for ccol in currency_cols:
			dataset[ccol] = dataset[ccol].apply(lambda x: format_currency(x, currency="BRL", locale="PT_BR"))
	if decimal_cols:
		for dcol in decimal_cols:
			dataset[dcol] = dataset[dcol].apply(lambda x: format_decimal(x, locale="PT_BR"))

	return dataset


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):



    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax.get_figure(), ax