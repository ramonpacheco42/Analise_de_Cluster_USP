# %%
# Semelhança vs Dissimilaridade
# Importando as bibliotecas
import rpy2.robjects as robjects
import pandas as pd
import numpy as np
import plotly.express as px 
import plotly.io as pio
import statsmodels.api as sm
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from rpy2.robjects import pandas2ri
from statsmodels.formula.api import ols
pandas2ri.activate()
# Carregando o arquivo RData
robjects.r['load']('Regional Varejista.RData')
# %%
# Lendo o dataframe carregado acima
pandas2ri.activate()
dados = robjects.r['RegionalVarejista']
df = pd.DataFrame.from_records(dados)
# %%
# Transpor o dataframe
df_transposed = df.transpose()

# Resetar o índice do dataframe transposto
RegionalVarejista = df_transposed.reset_index(drop=True)
# Redefinindo os nomes das colunas
RegionalVarejista.columns = ['loja', 'regional', 'atendimento', 'sortimento', 'organização']
# Imprimindo o DataFrame
RegionalVarejista
# %%
# Plotando um gráfico 3D
pio.renderers.default='browser'

fig = px.scatter_3d(RegionalVarejista, 
                    x='atendimento', 
                    y='sortimento', 
                    z='organização',
                    color='regional')
fig.show()
# %%
# Estatística Descirtivas
RegionalVarejista.describe()
# %%
print(RegionalVarejista['organização'].mean())
print(f"{RegionalVarejista['organização'].median()}\n")
print(RegionalVarejista['sortimento'].mean())
print(f"{RegionalVarejista['sortimento'].median()}\n")
print(RegionalVarejista['atendimento'].mean())
print(RegionalVarejista['atendimento'].median())
# %%
# Organizando dtype
df = RegionalVarejista.iloc[:,2:].astype('float64')
df
# %%
# Criando uma matriz de dissimilaridades utilizando euclidiana
dist_matrix = pdist(df, metric='euclidean')
dist_matrix_vw = squareform(dist_matrix)
df_vw = pd.DataFrame(dist_matrix_vw)
df_vw
# %%
# Elaboração da clusterização hierárquica
cluster_single = linkage(dist_matrix, method='single')
cluster_single_vw = pd.DataFrame(cluster_single, columns=['col1', 'col2', 'col3', 'col4'])
coeficientes = cluster_single_vw['col3']
coeficientes
# %%
# Gerando o dendograma para a definição dos cluster
plt.figure(figsize=(16,8))

dendrogram = sch.dendrogram(sch.linkage(df, method = 'single', metric = 'euclidean'), labels = list(RegionalVarejista.loja))
plt.title('Dendrograma', fontsize=16)
plt.xlabel('Lojas', fontsize=16)
plt.ylabel('Distância Euclidiana', fontsize=16)
plt.axhline(y = 40, color = 'red', linestyle = '--')
plt.show()
# %%
# Criando o cluster e definindo k=3
cluster_sing = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'single')
indica_cluster_sing = cluster_sing.fit_predict(df)
# Adicionando a variável cluster_H (Qualitativa) ao dataframe
RegionalVarejista['cluster_H'] = indica_cluster_sing
RegionalVarejista['cluster_H'] = RegionalVarejista['cluster_H'].astype(int)
RegionalVarejista['atendimento'] = RegionalVarejista['atendimento'].astype(int)
RegionalVarejista['sortimento'] = RegionalVarejista['sortimento'].astype(int)
RegionalVarejista['organização'] = RegionalVarejista['organização'].astype(int)
RegionalVarejista
# %%
# Gerando análise descritiva da variável atendimento
desc_atendimento = RegionalVarejista.groupby('cluster_H')['atendimento'].agg(['mean', 'std', 'min', 'max']).reset_index()
desc_atendimento
# %%
# Gerando análise descritiva da variável sortimento
desc_sortimento = RegionalVarejista.groupby('cluster_H')['sortimento'].agg(['mean', 'std', 'min', 'max']).reset_index()
desc_sortimento
# %%
# Gerando análise descritiva da variável organização
desc_organizacao = RegionalVarejista.groupby('cluster_H')['organização'].agg(['mean', 'std', 'min', 'max']).reset_index()
desc_organizacao
# %%
# Criando ANOVA para variável atendimentp
model = ols('atendimento ~ C(cluster_H)', data=RegionalVarejista).fit()
# gerar a tabela ANOVA
anova_atendimento = sm.stats.anova_lm(model, typ=2)
print(anova_organizacao['PR(>F)'][0] <= 0.05)
print(anova_atendimento)
# %%
# Criando ANOVA para variável soritmento
model = ols('sortimento ~ C(cluster_H)', data=RegionalVarejista).fit()
# gerar a tabela ANOVA
anova_sortimento = sm.stats.anova_lm(model, typ=2)
print(anova_organizacao['PR(>F)'][0] <= 0.05)
print(anova_sortimento)
# %%
# Criando ANOVA para variável organização
model = ols('organização ~ C(cluster_H)', data=RegionalVarejista).fit()
# gerar a tabela ANOVA
anova_organizacao = sm.stats.anova_lm(model, typ=2)
print(anova_organizacao['PR(>F)'][0] <= 0.05)
print(anova_organizacao)
# %%
# Elaboração da matriz de destâncias com a distância de Manhattan.
dist_man = pdist(df, metric='cityblock')
dist_man
# %%
# Clusterização hierárquica com método complete
cluster_complete = linkage(dist_man, method='complete')
cluster_complete
# %%
# Gerando o dendograma para a clusterização complete manhattan
plt.figure(figsize=(16,8))

dendrogram = sch.dendrogram(sch.linkage(df, method = 'complete', metric = 'cityblock'), labels = list(RegionalVarejista.loja))
plt.title('Dendrograma', fontsize=16)
plt.xlabel('Lojas', fontsize=16)
plt.ylabel('Distância Euclidiana', fontsize=16)
plt.axhline(y = 100, color = 'red', linestyle = '--')
plt.show()
# %%
# Criando o cluster e definindo k=3
cluster_complete = AgglomerativeClustering(n_clusters = 3, affinity = 'cityblock', linkage = 'complete')
indica_cluster_complete = cluster_sing.fit_predict(df)
# Adicionando a variável cluster_H_man (Qualitativa) ao dataframe
RegionalVarejista['cluster_H_man'] = indica_cluster_complete
RegionalVarejista
# %%
# Gerando a clusterização não hierárquico K-MEANS
kmeans = KMeans(n_clusters = 3, init = 'random').fit(df)
kmenas_cluster = kmeans.labels_
print(kmenas_cluster)
RegionalVarejista['cluster_kmeans'] = kmenas_cluster
RegionalVarejista
# %%

# %%
# Método Elbow para identificação do nº de clusters

X = df.values
sum_of_squared_distances = []

# Definir o número de clusters que queremos testar
K = range(1, 11)

# Calcular a soma das distâncias quadráticas intra-cluster para cada valor de K
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    sum_of_squared_distances.append(km.inertia_)

# Plotar o gráfico
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Soma das distâncias quadráticas intra-cluster')
plt.title('Método Elbow para determinar o número de clusters')
plt.show()
# %%
RegionalVarejista
# %%
# Criando ANOVA para variável atendimento - Kmeans
model = ols('atendimento ~ C(cluster_kmeans)', data=RegionalVarejista).fit()
# gerar a tabela ANOVA
anova_atendimento = sm.stats.anova_lm(model, typ=2)
print(anova_organizacao['PR(>F)'][0] <= 0.05)
print(anova_atendimento)
# %%
# Criando ANOVA para variável soritmento - Kmeans
model = ols('sortimento ~ C(cluster_kmeans)', data=RegionalVarejista).fit()
# gerar a tabela ANOVA
anova_sortimento = sm.stats.anova_lm(model, typ=2)
print(anova_organizacao['PR(>F)'][0] <= 0.05)
print(anova_sortimento)
# %%
# Criando ANOVA para variável organização - Kmeans
model = ols('organização ~ C(cluster_kmeans)', data=RegionalVarejista).fit()
# gerar a tabela ANOVA
anova_organizacao = sm.stats.anova_lm(model, typ=2)
print(anova_organizacao['PR(>F)'][0] <= 0.05)
print(anova_organizacao)
# %%
