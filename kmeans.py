# %%
# Importando as biblíotecas do projeto
import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from scipy.stats import zscore
# %%
# Carregando o dataframe
paises = pd.read_csv('dados_paises.csv')
paises.head()
# %%
# Gerando as estatística descritivas do dataframe
paises.describe()
# %%
# normalizando os dados com z-score
# Seleciona as colunas que serão normalizadas
cols_to_normalize = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']
# Normaliza as colunas selecionadas usando z-score
paises_normalizado = paises[cols_to_normalize].apply(zscore, axis=0)
paises_normalizado.insert(0,'country', paises['country'])
df = paises_normalizado.iloc[:,1:]
df
# %%
# Checando a média da variável gdpp
paises_normalizado.gdpp.mean().round()
# %%
# Checando a desvio padrão da variável gdpp
paises_normalizado.gdpp.std().round()
#%%
dist_matrix = pdist(df, metric='euclidean')
dist_matrix
#%%
# Gerando o single linkage (vizinhos mais próximos)
cluster_single = linkage(dist_matrix, method='single')
# Gerando o dendograma
dendrogram(cluster_single, labels=paises_normalizado['country'].values)
# %%
# Gerando o complete linkage (vizinhos mais distantes)
cluster_complete = linkage(dist_matrix, method='complete')
plt.figure(figsize=(20,10))
dendrogram(cluster_complete, labels=paises_normalizado['country'].values)
# Traça a linha horizontal no eixo y = 5.5
plt.axhline(y=5.5, color='black', linestyle='--')
# Mostra o plot
plt.show()
# %%
# Gerando o average linkage (vizinhos médios)
cluster_average = linkage(dist_matrix, method='average')
dendrogram(cluster_average, labels=paises_normalizado['country'].values)

# %%
from scipy.cluster import hierarchy
# Complete linkage com dendograma e cluster
Z = hierarchy.linkage(cluster_complete, 'complete')
plt.figure()
dn = hierarchy.dendrogram(Z)
# %%
cluster = fcluster(cluster_complete, 12, criterion='maxclust')
paises['Cluster_H'] = cluster
paises
# %%
# Tabela ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
# Crie o modelo ANOVA - variável child_mort
formula = "child_mort ~ C(Cluster)" 
modelo = ols(formula, paises).fit()
anova_tabela = sm.stats.anova_lm(modelo, typ=2)
anova_tabela
# %%
# Crie o modelo ANOVA - variável exports
formula = "exports ~ C(Cluster)" 
modelo = ols(formula, paises).fit()
anova_tabela = sm.stats.anova_lm(modelo, typ=2)
anova_tabela
# %%
# Crie o modelo ANOVA - variável health
formula = "health ~ C(Cluster)" 
modelo = ols(formula, paises).fit()
anova_tabela = sm.stats.anova_lm(modelo, typ=2)
anova_tabela
# %%
# Crie o modelo ANOVA - variável imports
formula = "imports ~ C(Cluster)" 
modelo = ols(formula, paises).fit()
anova_tabela = sm.stats.anova_lm(modelo, typ=2)
anova_tabela
# %%
# Crie o modelo ANOVA - variável income
formula = "income ~ C(Cluster)" 
modelo = ols(formula, paises).fit()
anova_tabela = sm.stats.anova_lm(modelo, typ=2)
anova_tabela
# %%
# Crie o modelo ANOVA - variável inflation
formula = "inflation ~ C(Cluster)" 
modelo = ols(formula, paises).fit()
anova_tabela = sm.stats.anova_lm(modelo, typ=2)
anova_tabela
# %%
# Crie o modelo ANOVA - variável life_expec
formula = "life_expec ~ C(Cluster)" 
modelo = ols(formula, paises).fit()
anova_tabela = sm.stats.anova_lm(modelo, typ=2)
anova_tabela
# %%
# Crie o modelo ANOVA - variável total_fer
formula = "total_fer ~ C(Cluster)" 
modelo = ols(formula, paises).fit()
anova_tabela = sm.stats.anova_lm(modelo, typ=2)
anova_tabela
# %%
# Crie o modelo ANOVA - variável gdpp
formula = "gdpp ~ C(Cluster)" 
modelo = ols(formula, paises).fit()
anova_tabela = sm.stats.anova_lm(modelo, typ=2)
anova_tabela
# %%
# Gerando o K-Means
X = df.iloc[:, 1:].values
sum_of_squared_distances = []
# Definir o número de clusters que queremos testar
K = range(1, 12)
# Calcular a soma das distâncias quadráticas intra-cluster para cada valor de K
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    sum_of_squared_distances.append(km.inertia_)
# %%
# Definir o número de clusters
k = 12
# Treinar o modelo K-means
kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
# %%
paises['Cluster_H'] = kmeans.labels_
paises
