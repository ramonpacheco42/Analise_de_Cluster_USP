# %%
# Importando as bibliotecas
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.io as pio
import seaborn as sns
from scipy.stats import zscore
import matplotlib.pyplot as plt
# %%
# Importando dataset
dados_uso = pd.read_csv('cartao_credito.csv')
dados_uso
# %%
# Normalizando os dados com o método Z-Score
dados_padronizado = dados_uso.iloc[:,2:7].apply(zscore, axis=0)
dados_padronizado
# %%
# Visualização 3D das viariáveis
pio.renderers.default='browser'

fig = px.scatter_3d(dados_padronizado, 
                    x=dados_padronizado.Avg_Credit_Limit, 
                    y=dados_padronizado.Total_Credit_Cards, 
                    z=dados_padronizado.Total_visits_bank,
                    color=dados_uso.Sl_No
                    )
fig.show()
# %%
# Método Elbow para identificação do nº de clusters

X = dados_padronizado.values
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
# Gerando a clusterização não hierárquico K-MEANS
kmeans = KMeans(n_clusters = 4, init = 'random').fit(dados_padronizado)
kmenas_cluster = kmeans.labels_
dados_padronizado['cluster_kmeans'] = kmenas_cluster
dados_uso['cluster_kmeans'] = kmenas_cluster
# %%
dados_padronizado
# %%
dados_uso
# %%
# Análisando os cluster formados pela variável número de cartões de crédito e total de visitas ao banco
sns.scatterplot(data=dados_padronizado, 
                x='Avg_Credit_Limit', 
                y='Total_Credit_Cards', 
                hue='cluster_kmeans')

plt.xlabel("Limite Médio")
plt.ylabel("Quantidade de Cartões")

plt.show()
# %%
# Análisando os cluster formados pelas variáveis limite médio e qtde de visitas
sns.scatterplot(data=dados_padronizado, 
                x='Avg_Credit_Limit', 
                y='Total_visits_bank', 
                hue='cluster_kmeans')

plt.xlabel("Limite Médio")
plt.ylabel("Quantidade de Visitas ao Banco")

plt.show()
# %%
# Análisando os cluster formados pelas variáveis média do limite de crédito e total de visitas online
sns.scatterplot(data=dados_padronizado, 
                x='Avg_Credit_Limit', 
                y='Total_visits_online', 
                hue='cluster_kmeans')

plt.xlabel("Limite Médio")
plt.ylabel("Quantidade de Visitas ao Banco Online")

plt.show()
# %%
# Análise Descritiva
analise_desc = dados_uso.groupby('cluster_kmeans').agg(limite=('Avg_Credit_Limit', 'mean'),
                                             q_cartoes=('Total_Credit_Cards', 'mean'),
                                             q_visitas=('Total_visits_bank', 'mean'),
                                             q_online=('Total_visits_online', 'mean'),
                                             q_liga=('Total_calls_made', 'mean'))

analise_desc
# %%
# Criando ANOVA para variável Avg_Credit_Limit
model = ols('Avg_Credit_Limit ~ C(cluster_kmeans)', data=dados_padronizado).fit()
# gerar a tabela ANOVA
anova_Avg_Credit_Limit = sm.stats.anova_lm(model, typ=2)
print(anova_Avg_Credit_Limit['PR(>F)'][0] <= 0.05)
print(anova_Avg_Credit_Limit)
# %%
# Criando ANOVA para variável Total_Credit_Cards
model = ols('Total_Credit_Cards ~ C(cluster_kmeans)', data=dados_padronizado).fit()
# gerar a tabela ANOVA
anova_Total_Credit_Cards = sm.stats.anova_lm(model, typ=2)
print(anova_Total_Credit_Cards['PR(>F)'][0] <= 0.05)
print(anova_Total_Credit_Cards)
# %%
# Criando ANOVA para variável Total_visits_bank
model = ols('Total_visits_bank ~ C(cluster_kmeans)', data=dados_padronizado).fit()
# gerar a tabela ANOVA
anova_Total_visits_bank = sm.stats.anova_lm(model, typ=2)
print(anova_Total_visits_bank['PR(>F)'][0] <= 0.05)
print(anova_Total_visits_bank)
# %%
# Criando ANOVA para variável Total_visits_online
model = ols('Total_visits_online ~ C(cluster_kmeans)', data=dados_padronizado).fit()
# gerar a tabela ANOVA
anova_Total_visits_online = sm.stats.anova_lm(model, typ=2)
print(anova_Total_visits_online['PR(>F)'][0] <= 0.05)
print(anova_Total_visits_online)
# %%
# Criando ANOVA para variável Total_calls_made
model = ols('Total_calls_made ~ C(cluster_kmeans)', data=dados_padronizado).fit()
# gerar a tabela ANOVA
anova_Total_calls_made = sm.stats.anova_lm(model, typ=2)
print(anova_Total_calls_made['PR(>F)'][0] <= 0.05)
print(anova_Total_calls_made)
# %%
