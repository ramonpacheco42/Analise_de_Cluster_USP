# %%
# Importando as biblíotecas do projeto
import pandas as pd
import numpy as np
import math
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
from scipy.stats import f_oneway
# %%
# Criando as colunas do DataFrame
estudantes = pd.Series(['Gabriela','Luiz Felipe','Patricia','Ovídio','Leonor'], name='Estudantes')
matematica = pd.Series([3.70,7.80,8.90,7.00,3.40], name='Matematica')
fisica = pd.Series([2.70,8.00,1.00,1.00,2.00], name='Fisica')
quimica = pd.Series([9.10,1.50,2.70,9.00,5.00], name='Quimica')
# %%
# Criando o DataFrame
df = pd.DataFrame(estudantes)
df = df.join(matematica)
df = df.join(fisica)
df = df.join(quimica)
df
# %%
# Criando novo dataframe contendo somente as notas dos alunos
notas = df[['Matematica', 'Fisica', 'Quimica']]
notas
# %%
# Criando a função que cácula a distancia euclidiana
def dist_euclidiana(row1, row2):
    soma = 0
    for i in range(len(row1)):
        soma += (row1[i] - row2[i])**2
    return math.sqrt(soma)
# %%
# Criando a função que cálcula a distancia euclidiana quadratica
def dist_euclidiana_quadratica(row1, row2):
    soma = 0
    for i in range(len(row1)):
        soma += (row1[i] - row2[i])**2
    return soma
# %%
def dist_manhattan(row1, row2):
    soma = 0
    for i in range(len(row1)):
        soma += abs(row1[i] - row2[i])
    return soma
# %%
def dist_chebyshev(row1, row2):
    max_dist = 0
    for i in range(len(row1)):
        dist = abs( row1[i] - row2[i])
        if dist > max_dist:
            max_dist = dist
    return max_dist

# %%
def dist_canberra(row1, row2):
    return sum(abs(x1 - x2) / (abs(x1) + abs(x2)) for x1, x2 in zip(row1, row2))
# %%
# Realizando o cálculo e motando um DataFrame com os resultados
distancias = []
for i in range(len(notas)):
    for j in range(i+1, len(notas)):
        dist = dist_euclidiana(notas.iloc[i], notas.iloc[j])
        dist_quad = dist_euclidiana_quadratica(notas.iloc[i], notas.iloc[j])
        dist_man = dist_manhattan(notas.iloc[i], notas.iloc[j])
        dist_cheb = dist_chebyshev(notas.iloc[i], notas.iloc[j])
        dist_can = dist_canberra(notas.iloc[i], notas.iloc[j])
        corr = notas.iloc[i].corr(notas.iloc[j])
        distancias.append([df['Estudantes'][i], df['Estudantes'][j], dist, dist_quad, dist_man, dist_cheb, dist_can, corr])
df_distancias = pd.DataFrame(distancias, columns=['Aluno 01', 'Aluno 02', 'Distância Euclidiana', 'Euclidiana Quadratica', 'Distância Manhattan', 'Distância de Chebyshev', 'Distância de Canberra', 'Correlação'])
df_distancias
# %%
# Metódo de Encadeamento
# Single Linkage (Os vizinhos mais próximos)
# Avg Complete
# Definindo que vamos trabalhar com a distância euclidiana
# Criando o dataframe somente com a distância Euclidiana
df_euclidiana = df_distancias['Distância Euclidiana']
df_euclidiana
# %%
# Cálculando a distância euclidiama de uma maneira mais rápida usando a biblíoteca Scipy
#dist_matrix = squareform(pdist(df.iloc[:, 1:], metric='euclidean'))
# Realiza a clusterização utilizando o método single linkage
Z = linkage(df_euclidiana, method='single')
# Percorre a matriz de linkage e exibe as informações de cada estágio
for i, row in enumerate(Z):
    print(f"Estágio {i}: elementos {int(row[0])} e {int(row[1])}, distância {row[2]}")

# Exibe o dendrograma
dendrogram(Z, labels=df['Estudantes'].values)
# %%
# Corta o dendrograma em três grupos
grupos = fcluster(Z, 3, criterion='maxclust')

# Adiciona os grupos ao DataFrame original
df['Grupo'] = grupos

# Exibe o resultado
print(df)
# %%
# Tabela Anova
# Definindo o p valor e f valor para variável matemática
group1 = df[df['Grupo'] == 1]['Matematica']
group2 = df[df['Grupo'] == 2]['Matematica']
group3 = df[df['Grupo'] == 3]['Matematica']
fvalue, pvalue = f_oneway(group1, group2, group3)
print(f'Para a variável Matemática o fvalor é de: {fvalue}')
print(f'Para a variável Matemática o fvalor é de {pvalue} ')
# %%
# Tabela Anova
# Definindo o p valor e f valor para variável fisica
group1 = df[df['Grupo'] == 1]['Fisica']
group2 = df[df['Grupo'] == 2]['Fisica']
group3 = df[df['Grupo'] == 3]['Fisica']
fvalue, pvalue = f_oneway(group1, group2, group3)
print(f'Para a variável Matemática o fvalor é de: {fvalue}')
print(f'Para a variável Matemática o fvalor é de {pvalue} ')
# %%
# Tabela Anova
# Definindo o p valor e f valor para variável quimica
group1 = df[df['Grupo'] == 1]['Quimica']
group2 = df[df['Grupo'] == 2]['Quimica']
group3 = df[df['Grupo'] == 3]['Quimica']
fvalue, pvalue = f_oneway(group1, group2, group3)
print(f'Para a variável Matemática o fvalor é de: {fvalue}')
print(f'Para a variável Matemática o fvalor é de {pvalue} ')

# %%
# Fazendo a mesma coisa que feito acima só que ao invés de utilizar o método
# Single Linkage vamos usar o método Avarege Linkege
# Realiza a clusterização utilizando o método average linkage
Z = linkage(df_euclidiana, method='average')
# Exibe o dendrograma
dendrogram(Z, labels=df['Estudantes'].values)
# %%
# Corta o dendrograma em três grupos
grupos = fcluster(Z, 3, criterion='maxclust')

# Adiciona os grupos ao DataFrame original
df['Grupo'] = grupos

# Exibe o resultado
print(df)
# %%
# Rodando com o método complete
Z = linkage(df_euclidiana, method='complete')
# Exibe o dendrograma
dendrogram(Z, labels=df['Estudantes'].values)
# %%
# Motando o cluster para complete
grupos = fcluster(Z, 2, criterion='maxclust')

# Adiciona os grupos ao DataFrame original
df['Grupo'] = grupos

# Exibe o resultado
print(df)