#!/usr/bin/env python
# coding: utf-8

# # Projeto Airbnb Rio - Ferramenta de Previsão de Preço de Imóvel para pessoas comuns 

# ### Introdução
# 
# No Airbnb, qualquer pessoa que tenha um quarto ou um imóvel de qualquer tipo (apartamento, casa, chalé, pousada, etc.) pode ofertar o seu imóvel para ser alugado por diária.
# 
# Você cria o seu perfil de host (pessoa que disponibiliza um imóvel para aluguel por diária) e cria o anúncio do seu imóvel.
# 
# No anúncio, o host deve descrever as características do imóvel da forma mais completa possível, de forma a tornar o anúncio mais amigável para os locadores/viajantes que estão em busca de um imóvel.
# 
# Existem dezenas de personalizações possíveis no seu anúncio, desde quantidade mínima de diária, de preço, de quantidade de quartos, até regras de cancelamento, taxa extra para hóspedes extras, exigência de verificação de identidade do locador, etc.
# 
# ### Objetivo
# 
# Construir um modelo de previsão de preço que permita uma pessoa comum que possui um imóvel possa saber quanto deve cobrar pela diária do seu imóvel.
# 
# Ou ainda, para o locador comum, dado o imóvel que ele está buscando, ajudar a saber se aquele imóvel está com preço atrativo (abaixo da média para imóveis com as mesmas características) ou não.
# 
# ### Base de dados
# 
# - As bases de dados foram retiradas do site kaggle: https://www.kaggle.com/allanbruno/airbnb-rio-de-janeiro
# - As bases de dados são os preços dos imóveis obtidos e suas respectivas características em cada mês.
# - Os preços são dados em reais (R$)
# - Temos bases de abril de 2018 a maio de 2020, com exceção de junho de 2018 que não possui base de dados
# 
# ### Expectativas Iniciais
# 
# - Acredito que a sazonalidade pode ser um fator importante, visto que meses como dezembro costumam ser bem caros no RJ
# - A localização do imóvel deve fazer muita diferença no preço, já que no Rio de Janeiro a localização pode mudar completamente as características do lugar (segurança, beleza natural, pontos turísticos)
# - Adicionais/Comodidades podem ter um impacto significativo, visto que temos muitos prédios e casas antigos no Rio de Janeiro
# 
# Vamos descobrir o quanto esses fatores impactam e se temos outros fatores não tão intuitivos que são extremamente importantes.

# ### Importação de Bibliotecas e da Bases de Dados

# In[2]:


# Bibliotecas
import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split


# - Os dados estão separados em várias planilhas de acordo com o mês e o ano daquele respectivo conjunto de dados. Dessa forma, precisaremos unificá-los para realizar a análise.
# <br>
# <br>
# - No processo de importação dos dados, realizaremos a unificação. Para saber de qual mês e de qual ano veio cada conjunto de dados, após a importação iremos adicionar uma coluna com o mês e com o ano de cada informação. 
# 
# 

# In[4]:


# Unificando os dados
meses = {'jan': 1, 'fev':2, 'mar':3, 'abr': 4, 'mai':5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

caminho_bases = pathlib.Path('dataset')

base_airbnb = pd.DataFrame()

for arquivo in caminho_bases.iterdir():
    nome_mes = arquivo.name[:3]
    mes = meses[nome_mes]
    
    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv', ''))
    
    df = pd.read_csv(caminho_bases / arquivo.name, low_memory=False)
    df['ano'] = ano
    df['mes'] = mes
    base_airbnb = base_airbnb.append(df)

display(base_airbnb)


# ### Tratamentos

# - Como temos muitas colunas, nosso modelo pode se tornar muito lento. Uma análise rápida permite ver que várias colunas não são necessárias para o nosso modelo de previsão. Neste caso, vamos excluir algumas colunas da nossa base. Os tipos de colunas que vamos excluir serão:
# 
#     1. IDs, Links e informações irrelevantes para o nosso modelo;
#     2. Colunas repetidas ou extremamente parecidas com outras colunas (Ex: Data x Ano/Mês);
#     3. Colunas preenchidas com texto livre -> Não rodaremos nenhuma análise de palavras;
#     4. Colunas em que todos ou quase todos os valores são iguais.
#     <br>
#     <br>
# - Para avaliar quais colunas serão excluídas, vamos criar um arquivo em excel com os 1.000 primeiros registros e fazer uma análise qualitativa.

# In[5]:


# Analisando as colunas da base de dados
print(list(base_airbnb.columns))
base_airbnb.head(1000).to_csv('primeiros_registros.csv', sep=';')


# In[6]:


# Excluindo as colunas selecionadas
colunas = ['host_response_time','host_response_rate','host_is_superhost','host_listings_count','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','ano','mes']

base_airbnb = base_airbnb.loc[:, colunas]
print(list(base_airbnb.columns))
display(base_airbnb)


# ### Tratando valores que estão faltando

# - Observando os dados, percebemos que existe uma grande quantidade de dados vazios. As colunas com mais de 300.000 valores NaN (valores vazios) serão excluídas da análise.
# <br>
# <br>
# - Para as outras colunas que possuem valores NaN, excluiremos as linhas que contém esses valores, uma vez que, por conta da nossa grande quantidade de dados, a quantidade de linhas que serão excluídas não resultará em perdas significativas para a análise.

# In[7]:


# Excluindo as colunas com dados vazios
for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() > 300000:
        base_airbnb = base_airbnb.drop(coluna, axis=1)
print(base_airbnb.isnull().sum())


# In[8]:


# Excluindo as linhas com dados vazios
base_airbnb = base_airbnb.dropna()

print(base_airbnb.shape)
print(base_airbnb.isnull().sum())


# ### Verificando os tipos de dados de cada coluna

# - Precisamos conferir se os dados de cada coluna estão em um formato de acordo com o que aquela variável representa, já que podem ocorrer casos em que, por exemplo, uma coluna de números estar definida como uma coluna de texto. Esse tipo de problema levará a erros no futuro, logo, precisamos corrigir. 

# In[9]:


print(base_airbnb.dtypes)
print('-'*60)
print(base_airbnb.iloc[0])


# - As colunas preço e extra people estão sendo reconhecidas como variaveis do tipo 'object' (texto) ao invés do tipo numérico (int ou float). Portanto, temos que convertê-las para o tipo que as representa corretamente.

# In[10]:


#price
base_airbnb['price'] = base_airbnb['price'].str.replace('$', '')
base_airbnb['price'] = base_airbnb['price'].str.replace(',', '')
base_airbnb['price'] = base_airbnb['price'].astype(np.float32, copy=False)
#extra people
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace(',', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].astype(np.float32, copy=False)
#verificando os tipos
print(base_airbnb.dtypes)


# ### Análise exploratória e tratamento de outliers

# - Agora iremos analisar as colunas atuais da base de dados. Nesta fase, iremos:
#     1. Analisar a correlação entre as colunas. Se existirem colunas com correlação muito forte, analisaremos a possibilidade de excluir algumas delas.
#     2. Excluir outliers. Como regra, usaremos valores abaixo de 'Q1 - 1.5xAmplitude' e valores acima de 'Q3 + 1.5x Amplitude' para definir outliers. *Após a identificação de outliers, não necessariamente eles serão excluídos, pois pode acontecer de alguns desses valores discrepantes serem importantes para a análise.
#     3. Confirmar se todas as features que temos fazem realmente sentido para o nosso modelo ou se alguma delas não irão nos ajudar e deverão ser excluidas.
# <br>
# <br>
# - Nossa análise será feita na seguinte ordem:
#     
#     1. Vamos começar pelas colunas de preço e de extra_people, que são valores numéricos contínuos.
# 
#     2. Vamos analisar as colunas de valores numéricos discretos (accomodates, bedrooms, guests_included, etc.)
# 
#     3. Por fim, vamos avaliar as colunas de texto e definir quais categorias fazem sentido mantermos ou não.

# In[11]:


plt.figure(figsize=(15, 10))
sns.heatmap(base_airbnb.corr(), annot=True, cmap='Greens')
#print(base_airbnb.corr())


# - Aqui definiremos algumas funções para auxiliar a análise de outliers

# In[12]:


def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return q1 - 1.5 * amplitude, q3 + 1.5 * amplitude
def excluir_outliers(df, nome_coluna):
    qtde_linhas = df.shape[0]
    lim_inf, lim_sup = limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup), :]
    linhas_removidas = qtde_linhas - df.shape[0]
    return df,  linhas_removidas


# In[13]:


def diagrama_caixa(coluna):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    sns.boxplot(x=coluna, ax=ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)
    
def histograma(coluna):
    plt.figure(figsize=(15, 5))
    sns.distplot(coluna, hist=True)

def grafico_barra(coluna):  
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limites(coluna))


# ### Analise das colunas

# - Price

# In[14]:


diagrama_caixa(base_airbnb['price'])
histograma(base_airbnb['price'])


# Como estamos construindo um modelo para imóveis comuns, provavelmente os valores acima do limite superior serão apenas de apartamentos de luxo, que não são o nosso objetivo principal. Logo, podemos excluir esses outliers.

# In[15]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'price')
print('{} linhas removidas'.format(linhas_removidas))


# In[16]:


histograma(base_airbnb['price'])
print(base_airbnb.shape)


# - extra_people

# In[17]:


diagrama_caixa(base_airbnb['extra_people'])
histograma(base_airbnb['extra_people'])


# In[18]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'extra_people')
print('{} linhas removidas'.format(linhas_removidas))


# In[19]:


histograma(base_airbnb['extra_people'])
print(base_airbnb.shape)


# - host_listings_count

# In[20]:


diagrama_caixa(base_airbnb['host_listings_count'])
grafico_barra(base_airbnb['host_listings_count'])


# - Neste caso podemos excluir os outliers, pois, para o objetivo do nosso projeto, hosts com mais de 6 imóveis no airbnb não são o público alvo (provavelmente sejam imobiliárias ou profissionais que gerenciam imóveis no airbnb).

# In[21]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'host_listings_count')
print('{} linhas removidas'.format(linhas_removidas))


# - accommodates

# In[22]:


diagrama_caixa(base_airbnb['accommodates'])
grafico_barra(base_airbnb['accommodates'])


# - Mais uma vez, excluiremos os outliers dessa coluna pois apartamentos que acomodam mais de 9 pessoas não são o nosso foco, nosso objetivo é a precificação de imóveis comuns.

# In[23]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'accommodates')
print('{} linhas removidas'.format(linhas_removidas))


# - bathrooms 

# In[24]:


diagrama_caixa(base_airbnb['bathrooms'])
plt.figure(figsize=(15, 5))
sns.barplot(x=base_airbnb['bathrooms'].value_counts().index, y=base_airbnb['bathrooms'].value_counts())


# - Pelo mesmo motivo dos anteriores, vamos excluir os outliers nos banheiros

# In[25]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bathrooms')
print('{} linhas removidas'.format(linhas_removidas))


# - bedrooms

# In[26]:


diagrama_caixa(base_airbnb['bedrooms'])
grafico_barra(base_airbnb['bedrooms'])


# - Pelo mesmo motivo dos anteriores, vamos excluir outliers em quantidade de quartos

# In[27]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bedrooms')
print('{} linhas removidas'.format(linhas_removidas))


# - beds

# In[28]:


diagrama_caixa(base_airbnb['beds'])
grafico_barra(base_airbnb['beds'])


# - Pelo mesmo motivo dos anteriores, vamos excluir outliers em quantidade de camas

# In[29]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'beds')
print('{} linhas removidas'.format(linhas_removidas))


# - guests_included

# In[30]:


#diagrama_caixa(base_airbnb['guests_included'])
#grafico_barra(base_airbnb['guests_included'])
print(limites(base_airbnb['guests_included']))
plt.figure(figsize=(15, 5))
sns.barplot(x=base_airbnb['guests_included'].value_counts().index, y=base_airbnb['guests_included'].value_counts())


# - Neste caso, escolhemos remover essa coluna da análise. Observamos que, aparentemente, os usuários do airbnb usam muito o valor padrão do airbnb como 1 guest included. Isso pode levar o nosso modelo a considerar uma feature que na verdade não é essencial para a definição do preço. Logo, a removeremos da análise.

# In[31]:


base_airbnb = base_airbnb.drop('guests_included', axis=1)
base_airbnb.shape


# - minimum_nights

# In[32]:


diagrama_caixa(base_airbnb['minimum_nights'])
grafico_barra(base_airbnb['minimum_nights'])


# - Apartamentos com mais de 8 noites como o "mínimo de noites" podem ser apartamentos de temporada. Mais uma vez, como este não é nosso alvo, excluiremos esses outliers.

# In[33]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'minimum_nights')
print('{} linhas removidas'.format(linhas_removidas))


# - maximum_nights

# In[34]:


diagrama_caixa(base_airbnb['maximum_nights'])
grafico_barra(base_airbnb['maximum_nights'])


# - Parece que quase todos os hosts não preenchem o campo de maximum nights, logo, ele não parece ser um fator que será relevante. Nesse caso, excluiremos essa coluna da análise.

# In[35]:


base_airbnb = base_airbnb.drop('maximum_nights', axis=1)
base_airbnb.shape


# - number_of_reviews            

# In[36]:


diagrama_caixa(base_airbnb['number_of_reviews'])
grafico_barra(base_airbnb['number_of_reviews'])


# - Analisando essa coluna, chegamos à conclusão que talvez seja melhor para o nosso modelo excluí-la por dois principais motivos:<br>
#     1. Se excluirmos os outliers, vamos excluir as pessoas que tem a maior quantidade de reviews (o que normalmente são os hosts que têm mais aluguel). Isso pode impactar negativamente o nosso modelo. 
#     2. Pensando no nosso objetivo, se uma pessoa tem um imóvel parado e quer disponibilizá-lo no Airbnb, provavelmente ela terá poucas ou nenhuma review.

# In[37]:


base_airbnb = base_airbnb.drop('number_of_reviews', axis=1)
base_airbnb.shape


# ### Tratamento de colunas de valores de texto

# - property_type 

# In[38]:


print(base_airbnb['property_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('property_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# - Aqui a nossa ação não é "excluir outliers", mas sim agrupar valores que são muito pequenos para simplificar o modelo.
# 
# - Todos os tipos de imóveis que têm menos de 2.000 propriedades na base de dados serão agrupadas em uma categoria chamada "outros".

# In[39]:


tabela_tipos_casa = base_airbnb['property_type'].value_counts()
colunas_agrupar = []

for tipo in tabela_tipos_casa.index:
    if tabela_tipos_casa[tipo] < 2000:
        colunas_agrupar.append(tipo)
print(colunas_agrupar)

for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['property_type']==tipo, 'property_type'] = 'Outros'

print(base_airbnb['property_type'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot('property_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# - room_type 

# In[40]:


print(base_airbnb['room_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('room_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# Em tipo de quarto, não precisamos fazer nada, ele já parece estar bem distribuído

# - bed_type 

# In[41]:


print(base_airbnb['bed_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('bed_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

# agrupando categorias de bed_type
tabela_bed = base_airbnb['bed_type'].value_counts()
colunas_agrupar = []

for tipo in tabela_bed.index:
    if tabela_bed[tipo] < 10000:
        colunas_agrupar.append(tipo)
print(colunas_agrupar)

for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['bed_type']==tipo, 'bed_type'] = 'Outros'

print(base_airbnb['bed_type'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot('bed_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# - Aqui, novamente, a nossa ação não é "excluir outliers", mas sim agrupar valores que são muito pequenos para simplificar o modelo.
# 
# - Como temos 1 valor visivelmente muito maior do que todos os outros, criaremos apenas 2 grupos de camas: "Real Bed" e "outros"

# - cancellation_policy 

# In[42]:


print(base_airbnb['cancellation_policy'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('cancellation_policy', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

# agrupando categorias de cancellation_pollicy
tabela_cancellation = base_airbnb['cancellation_policy'].value_counts()
colunas_agrupar = []

for tipo in tabela_cancellation.index:
    if tabela_cancellation[tipo] < 10000:
        colunas_agrupar.append(tipo)
print(colunas_agrupar)

for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['cancellation_policy']==tipo, 'cancellation_policy'] = 'strict'

print(base_airbnb['cancellation_policy'].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot('cancellation_policy', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)


# - amenities 

# - Como temos uma diversidade muito grande de amenities e o cadastro desses amenities não são padronizados (um mesmo amenitie pode ser descrito de diferentes formas), vamos avaliar a <b>quantidade</b> de amenities como o parâmetro para o nosso modelo.

# In[43]:


print(base_airbnb['amenities'].iloc[1].split(','))
print(len(base_airbnb['amenities'].iloc[1].split(',')))

base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)


# In[44]:


base_airbnb = base_airbnb.drop('amenities', axis=1)
base_airbnb.shape


# In[45]:


diagrama_caixa(base_airbnb['n_amenities'])
grafico_barra(base_airbnb['n_amenities'])


# - Com essa adaptação da coluna de amenities, ela se tornou uma coluna de valor numérico. Dessa forma, ela também precisará passar pela verificação e exclusão dos outliers.

# In[46]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'n_amenities')
print('{} linhas removidas'.format(linhas_removidas))


# ### Visualização de mapa das propriedades

# - Criaremos um mapa que exibirá uma parte aleatória da nossa base de dados (50.000 propriedades) para verificar como as propriedades estão distribuídas pela cidade e também para identificar as regiões de maior preço.

# In[47]:


amostra = base_airbnb.sample(n=50000)
centro_mapa = {'lat':amostra.latitude.mean(), 'lon':amostra.longitude.mean()}
mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude',z='price', radius=2.5,
                        center=centro_mapa, zoom=10,
                        mapbox_style='stamen-terrain')
mapa.show()


# ### Encoding
# 
# - Uma vez que os algoritmos de machine learning que serão utilizados neste projeto não trabalham com variáveis textuais, precisaremos de alguma forma transformar essas variáveis textuais em variáveis numéricas. Este processo é chamado de Encoding. Features de Valores True ou False, substituiremos True por 1 e False por 0. Features de Categoria (features em que os valores da coluna são textos) vamos utilizar o método de encoding one hot encoding.

# In[48]:


colunas_tf = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready']
base_airbnb_cod = base_airbnb.copy()
for coluna in colunas_tf:
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=='t', coluna] = 1
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=='f', coluna] = 0


# In[49]:


colunas_categorias = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
base_airbnb_cod = pd.get_dummies(data=base_airbnb_cod, columns=colunas_categorias)
display(base_airbnb_cod.head())


# ### Modelo de Previsão

# #### Métricas de Avaliação
# - Nosso problema é um problema de regressão. Dois dos principais parâmetros de avaliação de desempenho de modelos de regressão são o R² e o Erro Quadrático Médio. Eles serão os parâmetros utilizados para a avaliação do nosso modelo. O R², basicamente, irá nos dizer o quão bem o nosso modelo consegue explicar o preço. Quanto mais próximo de 100%, melhor o modelo. Já o Erro Quadrático Médio, basicamente, irá nos mostrar o quanto o nosso modelo está errando. Quanto menor for o erro, melhor o modelo.

# In[50]:


# função que retornará os parametros de avaliação do modelo
def avaliar_modelo(nome_modelo, y_teste, previsao):
    r2 = r2_score(y_teste, previsao)
    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo {nome_modelo}:\nR²:{r2:.2%}\nRSME:{RSME:.2f}'


# - Modelos de regressão escolhidos para serem testados:<br>
#     1. RandomForest
#     2. LinearRegression
#     3. Extra Tree

# In[51]:


modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()

modelos = {'RandomForest': modelo_rf,
          'LinearRegression': modelo_lr,
          'ExtraTrees': modelo_et,
          }
y = base_airbnb_cod['price']
X = base_airbnb_cod.drop('price', axis=1)


# - Separação dos dados em treino e teste e treinamento do Modelo

# In[52]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

for nome_modelo, modelo in modelos.items():
    #treinar
    modelo.fit(X_train, y_train)
    #testar
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# - Análise do Melhor Modelo

# In[53]:


for nome_modelo, modelo in modelos.items():
    #testar
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# - Modelo avaliado como melhor modelo: ExtraTressRegressor
# 
#     Esse foi o modelo com maior valor de R² e ao mesmo tempo o menor valor de RSME. Como não tivemos uma grande diferença de velocidade de treino e de previsão desse modelo com o modelo de RandomForest (que teve resultados próximos de R² e RSME), vamos escolher o Modelo ExtraTrees.
#     
#     O modelo de regressão linear não obteve um resultado satisfatório, com valores de R² e RSME muito piores do que os outros 2 modelos.<br><br>
#     
# - Resultados das Métricas de Avaliação do Modelo Vencedor:<br>
# Modelo ExtraTrees: <b>R²</b>:97.49% ; <b>RSME</b>:41.99

# ### Ajustes e Melhorias no Melhor Modelo

# In[54]:


#print(modelo_et.feature_importances_)
#print(X_train.columns)
importancia_features = pd.DataFrame(modelo_et.feature_importances_, X_train.columns)
importancia_features = importancia_features.sort_values(by=0, ascending=False)
display(importancia_features)
plt.figure(figsize=(15, 5))
ax = sns.barplot(x=importancia_features.index, y=importancia_features[0])
ax.tick_params(axis='x', rotation=90)


# ### Ajustes Finais no Modelo
# 
# - is_business_travel ready não parece ter muito impacto no nosso modelo. Para chegar em um modelo mais simples, vamos excluir essa feature e testar o modelo sem ela.

# In[55]:


base_airbnb_cod = base_airbnb_cod.drop('is_business_travel_ready', axis=1)

y = base_airbnb_cod['price']
X = base_airbnb_cod.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

modelo_et.fit(X_train, y_train)
previsao = modelo_et.predict(X_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))


# In[56]:


base_teste = base_airbnb_cod.copy()
for coluna in base_teste:
    if 'bed_type' in coluna:    
        base_teste = base_teste.drop(coluna, axis=1)
print(base_teste.columns)
y = base_teste['price']
X = base_teste.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

modelo_et.fit(X_train, y_train)
previsao = modelo_et.predict(X_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))


# In[57]:


print(previsao)


# In[ ]:


X['price'] = y
X.to_csv('dados.csv')

