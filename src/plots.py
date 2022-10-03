import pandas as pd
import src.utils as gf

# Importando dataset
file = 'data/DadosDesafioCientista_full.csv'
dataset = pd.read_csv(file)

# Agrupando o Publico Alvo e construindo um novo dataset com 
# a porcentagem de população e domicilios.
r_dataset, dataset = gf.ratio_dataset(dataset)

# Dividindo dataset para os dois estados
df_rj = dataset[dataset['estado'] == 'RJ']
df_sp = dataset[dataset['estado'] == 'SP']

df_rj_r = r_dataset[r_dataset['estado'] == 'RJ']
df_sp_r = r_dataset[r_dataset['estado'] == 'SP']

###### Construindo os plots relacionais
gf.relplot(df_rj, x="popDe25a49", 
            y = 'domiciliosAB',
            hue = 'potencial',
            size = 'faturamento',
            xlabel='População de 25 a 49 anos', 
            ylabel = 'Domicílios Classe A e B',
            figname='relplot_RJ', bg_color = '#d3dbfe')



gf.relplot(df_sp, x="popDe25a49", 
            y = 'domiciliosAB',
            hue = 'potencial',
            size = 'faturamento',
            xlabel='População de 25 a 49 anos', 
            ylabel = 'Domicílios Classe A e B',
            figname='relplot_sp', bg_color = '#d3dbfe')



## Construindo plots de distribuição


gf.displot_UF(df_rj_r, x = 'popDe25a49', 
                xlabel= 'População de 25 a 49 anos [%]', 
                figname='distplot_pop_rj.png')
gf.displot_UF(df_sp_r, x = 'popDe25a49', 
                xlabel= 'População de 25 a 49 anos [%]', 
                figname='distplot_pop_sp.png')


gf.displot_UF(df_rj_r, x = 'domiciliosAB', 
                xlabel = 'Domicilios Classe A e B [%]',
                figname = 'distplot_dom_rj.png')
gf.displot_UF(df_sp_r, x = 'domiciliosAB', 
                xlabel = 'Domicilios Classe A e B [%]',
                figname = 'distplot_dom_sp.png')


##### Salvando as tabelas em png

selec_col = ['nome', 'cidade', 'população', 'popDe25a49', 'domiciliosAB', 'rendaMedia', 'faturamento', 'potencial']

get_first = 10

df_sp_r = df_sp_r[df_sp_r['potencial'] == 'Alto'][selec_col].sort_values('faturamento',ascending= False).head(get_first)
df_rj_r = df_rj_r[df_rj_r['potencial'] == 'Alto'][selec_col].sort_values('faturamento',ascending= False).head(get_first)

df_sp_r['Faturamento X Pop'] = df_sp_r['faturamento'] / df_sp_r['população']
df_rj_r['Faturamento X Pop'] = df_rj_r['faturamento'] / df_rj_r['população']

df_sp_r = df_sp_r.append((df_sp_r.mean()),ignore_index = True)
df_sp_r['nome'].loc[get_first] = 'Média'
df_sp_r = df_sp_r.fillna('')

df_rj_r = df_rj_r.append((df_rj_r.mean()),ignore_index = True)
df_rj_r['nome'].loc[get_first] = 'Média'
df_rj_r = df_rj_r.fillna('')


df_sp_r = gf.format_dataset(df_sp_r, currency_cols = ["faturamento","rendaMedia",'Faturamento X Pop'],
                 percent_cols = ["popDe25a49","domiciliosAB"], decimal_cols = ["população"])

df_rj_r = gf.format_dataset(df_rj_r, currency_cols = ["faturamento","rendaMedia",'Faturamento X Pop'],
                 percent_cols = ["popDe25a49","domiciliosAB"], decimal_cols = ["população"])



fig,ax = gf.render_mpl_table(df_rj_r, header_columns=0, col_width=4.0)
fig.savefig("figures/table_RJ.png",facecolor='#d3dbfe')

fig,ax = gf.render_mpl_table(df_sp_r, header_columns=0, col_width=4.0)
fig.savefig("figures/table_SP.png",facecolor='#d3dbfe')


