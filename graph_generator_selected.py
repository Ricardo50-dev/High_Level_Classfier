import numpy as np
import igraph as ig
import networkx as nx
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import pairwise
from joblib import Parallel, delayed


def rede_eng_Graph(raio, X, y, med, dist):
    # pega o num de classes
    numclasses = len(np.unique(y))

    if dist != 'cosine':
        dist = DistanceMetric.get_metric(dist)
        euclidean_dist = dist.pairwise(X)
    elif dist == 'cosine':
        euclidean_dist = pairwise.cosine_similarity(X)

    # preenche a diagonal principal com valor infinito
    np.fill_diagonal(euclidean_dist, np.inf)

    # ordena as distancias por objeto em ordem ascendente
    ind_ranking = np.argsort(euclidean_dist, axis=1)[:, :]
    # print(euclidean_dist)
    # inicializa uma mascara com False, num de objetos x k
    mask = np.zeros((len(X), len(X))).astype(int)

    # preenche os campos True da mascara, vizinho mais proximos que sao da mesma clase
    lista_grafos = []
    for i in range(len(ind_ranking)):
        for j in range(len(ind_ranking)):
            if y[ind_ranking[i][j]] == y[i] and euclidean_dist[i][j] <= raio:
                mask[i][j] = True

    # captura os indices True da mascara, linha e coluna
    links = mask.nonzero()

    # atribui os indices das linhas
    sources = links[0]

    # atribui o indice real dos objetos das colunas
    targets = ind_ranking[links]

    # inicializa funcao para mapeamento dos vertices
    map_vertices = np.zeros(len(ind_ranking)).astype(int) - 1

    # inicializa variavel para guardar o calculo das medidas
    measures = np.zeros(numclasses)
    for l in range(numclasses):
        # captura todos os objetos em sources que pertencem a classe l
        lsources = np.where(y[sources] == l)[0]

        # captura todos os objetos da base que pertencem a classe l
        all_vertices = np.where(y == l)[0]

        # recebe todos os objetos da classe l que estao conectados
        unique_vertices = np.unique(
            np.append(sources[lsources], targets[lsources])).astype(int)

        # recebe os demais objetos da classe l que nao estao conectados
        unique_vertices = np.unique(
            np.append(unique_vertices, all_vertices)).astype(int)

        # popula a funcao de mapeamento dos vertices com os objetos da classe l
        map_vertices[unique_vertices] = np.arange(
            len(unique_vertices)).astype(int)

        # print(len(unique_vertices), unique_vertices)

        # cria o grafo para classe l
        subg = ig.Graph(len(unique_vertices), list(zip(map_vertices[sources[lsources]].astype(
            int), map_vertices[targets[lsources]].astype(int))))

        # print(subg.vcount())

        # popula a lista de grafos
        lista_grafos.append(subg)

        medida_ind = 0
        # calcula as medidas de rede associadas ao grafo l
        if 'A' in med:
            measures[medida_ind][l] = Assortativity(subg)
            medida_ind += 1
        if 'B' in med:
            measures[medida_ind][l] = ClustCoefficient(subg)
            medida_ind += 1
        if 'C' in med:
            measures[medida_ind][l] = AvgDegree(subg)
            medida_ind += 1
        if 'D' in med:
            measures[medida_ind][l] = Betweenness(subg)
            medida_ind += 1
        if 'E' in med:
            measures[medida_ind][l] = AvgPathLength(subg)
            medida_ind += 1
        if 'F' in med:
            measures[medida_ind][l] = Closeness(subg)
            medida_ind += 1

    return lista_grafos, map_vertices, measures


def S_kNN_Graph(k, X, y, med, dist):
    numclasses = len(np.unique(y))
    lista_grafos = []
    # inicializa variavel para guardar o calculo das medidas
    measures = np.zeros(numclasses)
    # inicializa funcao para mapeamento dos vertices
    map_vertices = np.zeros(len(X)).astype(int) - 1
    for l in range(0, numclasses):

        classes = X[np.where(y == l)[0]]

        if dist != 'cosine':
            dist = DistanceMetric.get_metric(dist)
            euclidean_dist = dist.pairwise(classes)
        elif dist == 'cosine':
            euclidean_dist = pairwise.cosine_similarity(classes)

        # preenche a diagonal principal com valor infinito
        np.fill_diagonal(euclidean_dist, np.inf)

        # ordena as distancias por objeto em ordem ascendente
        ind_ranking = np.argsort(euclidean_dist, axis=1)[:, :k]

        # inicializa uma mascara com False, num de objetos x k
        mask = np.zeros((len(classes), k)).astype(int)

        # preenche os campos True da mascara, vizinho mais proximos que sao da mesma clase

        for i in range(len(ind_ranking)):
            mask[i] = True

        # captura os indices True da mascara, linha e coluna
        links = mask.nonzero()

        # atribui os indices das linhas
        sources = links[0]

        # atribui o indice real dos objetos das colunas
        targets = ind_ranking[links]

        # captura todos os objetos em sources que pertencem a classe l
        lsources = np.where(y[sources] == l)[0]

        # captura todos os objetos da base que pertencem a classe l
        all_vertices = np.where(y == l)[0]

        # recebe todos os objetos da classe l que estao conectados
        unique_vertices = np.unique(
            np.append(sources[lsources], targets[lsources])).astype(int)

        # recebe os demais objetos da classe l que nao estao conectados
        unique_vertices = np.unique(
            np.append(unique_vertices, all_vertices)).astype(int)

        # popula a funcao de mapeamento dos vertices com os objetos da classe l
        map_vertices[unique_vertices] = np.arange(
            len(unique_vertices)).astype(int)
        # print(len(unique_vertices), unique_vertices)

        # cria o grafo para classe l
        subg = ig.Graph(len(unique_vertices), list(zip(map_vertices[sources[lsources]].astype(
            int), map_vertices[targets[lsources]].astype(int))))

        # print(subg.vcount())

        # popula a lista de grafos
        lista_grafos.append(subg)

        medida_ind = 0
        # calcula as medidas de rede associadas ao grafo l
        if 'A' in med:
            measures[medida_ind][l] = Assortativity(subg)
            medida_ind += 1
        if 'B' in med:
            measures[medida_ind][l] = ClustCoefficient(subg)
            medida_ind += 1
        if 'C' in med:
            measures[medida_ind][l] = AvgDegree(subg)
            medida_ind += 1
        if 'D' in med:
            measures[medida_ind][l] = Betweenness(subg)
            medida_ind += 1
        if 'E' in med:
            measures[medida_ind][l] = AvgPathLength(subg)
            medida_ind += 1
        if 'F' in med:
            measures[medida_ind][l] = Closeness(subg)
            medida_ind += 1

    return lista_grafos, map_vertices, measures


def M_kNN_Graph(k, X, y, med, dist):

    # pega o num de classes
    numclasses = len(np.unique(y))

    if dist != 'cosine':
        dist = DistanceMetric.get_metric(dist)
        euclidean_dist = dist.pairwise(X)
    elif dist == 'cosine':
        euclidean_dist = pairwise.cosine_similarity(X)

    # preenche a diagonal principal com valor infinito
    np.fill_diagonal(euclidean_dist, np.inf)

    # ordena as distancias por objeto em ordem ascendente
    ind_ranking = np.argsort(euclidean_dist, axis=1)[:, :k]

    # inicializa uma mascara com False, num de objetos x k
    mask = np.zeros((len(X), k)).astype(int)

    # preenche os campos True da mascara, vizinho mais proximos que sao da mesma clase
    lista_grafos = []
    for i in range(len(ind_ranking)):
        for j in range(0, k):
            if ind_ranking[i][j] in ind_ranking[ind_ranking[i][j] - 1] and y[ind_ranking[i][j]] == y[i]:
                mask[i][j] = True
            else:
                mask[i][j] = False

    # captura os indices True da mascara, linha e coluna
    links = mask.nonzero()

    # atribui os indices das linhas
    sources = links[0]

    # atribui o indice real dos objetos das colunas
    targets = ind_ranking[links]

    # inicializa funcao para mapeamento dos vertices
    map_vertices = np.zeros(len(ind_ranking)).astype(int) - 1

    # inicializa variavel para guardar o calculo das medidas
    measures = np.zeros(numclasses)
    for l in range(numclasses):

        # captura todos os objetos em sources que pertencem a classe l
        lsources = np.where(y[sources] == l)[0]

        # captura todos os objetos da base que pertencem a classe l
        all_vertices = np.where(y == l)[0]

        # recebe todos os objetos da classe l que estao conectados
        unique_vertices = np.unique(
            np.append(sources[lsources], targets[lsources])).astype(int)

        # recebe os demais objetos da classe l que nao estao conectados
        unique_vertices = np.unique(
            np.append(unique_vertices, all_vertices)).astype(int)

        # popula a funcao de mapeamento dos vertices com os objetos da classe l
        map_vertices[unique_vertices] = np.arange(
            len(unique_vertices)).astype(int)

        # print(len(unique_vertices), unique_vertices)

        # cria o grafo para classe l
        subg = ig.Graph(len(unique_vertices), list(zip(map_vertices[sources[lsources]].astype(
            int), map_vertices[targets[lsources]].astype(int))))

        #print(subg.vcount())

        # popula a lista de grafos
        lista_grafos.append(subg)

        medida_ind = 0
        # calcula as medidas de rede associadas ao grafo l
        if 'A' in med:
            measures[medida_ind][l] = Assortativity(subg)
            medida_ind += 1
        if 'B' in med:
            measures[medida_ind][l] = ClustCoefficient(subg)
            medida_ind += 1
        if 'C' in med:
            measures[medida_ind][l] = AvgDegree(subg)
            medida_ind += 1
        if 'D' in med:
            measures[medida_ind][l] = Betweenness(subg)
            medida_ind += 1
        if 'E' in med:
            measures[medida_ind][l] = AvgPathLength(subg)
            medida_ind += 1
        if 'F' in med:
            measures[medida_ind][l] = Closeness(subg)
            medida_ind += 1

    return lista_grafos, map_vertices, measures


def S_M_kNN_Graph(k, X, y, med, dist):
    numclasses = len(np.unique(y))
    lista_grafos = []
    # inicializa variavel para guardar o calculo das medidas
    measures = np.zeros(numclasses)
    # inicializa funcao para mapeamento dos vertices
    map_vertices = np.zeros(len(X)).astype(int) - 1
    for l in range(0, numclasses):

        classes = X[np.where(y == l)[0]]

        if dist != 'cosine':
            dist = DistanceMetric.get_metric(dist)
            euclidean_dist = dist.pairwise(classes)
        elif dist == 'cosine':
            euclidean_dist = pairwise.cosine_similarity(classes)

        # preenche a diagonal principal com valor infinito
        np.fill_diagonal(euclidean_dist, np.inf)

        # ordena as distancias por objeto em ordem ascendente
        ind_ranking = np.argsort(euclidean_dist, axis=1)[:, :k]
        # inicializa uma mascara com False, num de objetos x k
        mask = np.zeros((len(classes), k)).astype(int)

        # preenche os campos True da mascara, vizinho mais proximos que sao da mesma clase

        for i in range(len(ind_ranking)):
            for j in range(0, k):
                if ind_ranking[i][j] in ind_ranking[ind_ranking[i][j] - 1]:
                    mask[i][j] = True

                else:
                    mask[i][j] = False

        # captura os indices True da mascara, linha e coluna
        links = mask.nonzero()

        # atribui os indices das linhas
        sources = links[0]

        # atribui o indice real dos objetos das colunas
        targets = ind_ranking[links]

        # captura todos os objetos em sources que pertencem a classe l
        lsources = np.where(y[sources] == l)[0]

        # captura todos os objetos da base que pertencem a classe l
        all_vertices = np.where(y == l)[0]

        # recebe todos os objetos da classe l que estao conectados
        unique_vertices = np.unique(
            np.append(sources[lsources], targets[lsources])).astype(int)

        # recebe os demais objetos da classe l que nao estao conectados
        unique_vertices = np.unique(
            np.append(unique_vertices, all_vertices)).astype(int)

        # popula a funcao de mapeamento dos vertices com os objetos da classe l
        map_vertices[unique_vertices] = np.arange(
            len(unique_vertices)).astype(int)
        # print(len(unique_vertices), unique_vertices)

        # cria o grafo para classe l
        subg = ig.Graph(len(unique_vertices), list(zip(map_vertices[sources[lsources]].astype(
            int), map_vertices[targets[lsources]].astype(int))))

        # print(subg.vcount())

        # popula a lista de grafos
        lista_grafos.append(subg)

        medida_ind = 0
        # calcula as medidas de rede associadas ao grafo l
        if 'A' in med:
            measures[medida_ind][l] = Assortativity(subg)
            medida_ind += 1
        if 'B' in med:
            measures[medida_ind][l] = ClustCoefficient(subg)
            medida_ind += 1
        if 'C' in med:
            measures[medida_ind][l] = AvgDegree(subg)
            medida_ind += 1
        if 'D' in med:
            measures[medida_ind][l] = Betweenness(subg)
            medida_ind += 1
        if 'E' in med:
            measures[medida_ind][l] = AvgPathLength(subg)
            medida_ind += 1
        if 'F' in med:
            measures[medida_ind][l] = Closeness(subg)
            medida_ind += 1

    return lista_grafos, map_vertices, measures


def kNN_Graph(k, X, y, med, dist):

    # pega o num de classes
    numclasses = len(np.unique(y))

    if dist != 'cosine':
        dist = DistanceMetric.get_metric(dist)
        euclidean_dist = dist.pairwise(X)
    elif dist == 'cosine':
        euclidean_dist = pairwise.cosine_similarity(X)

    # preenche a diagonal principal com valor infinito
    np.fill_diagonal(euclidean_dist, np.inf)

    # ordena as distancias por objeto em ordem ascendente
    ind_ranking = np.argsort(euclidean_dist, axis=1)[:, :k]

    # inicializa uma mascara com False, num de objetos x k
    mask = np.zeros((len(X), k)).astype(int)

    # preenche os campos True da mascara, vizinho mais proximos que sao da mesma clase
    lista_grafos = []
    for i in range(len(ind_ranking)):
        mask[i] = (y[ind_ranking[i]] == y[i])

    # captura os indices True da mascara, linha e coluna
    links = mask.nonzero()

    # atribui os indices das linhas
    sources = links[0]

    # atribui o indice real dos objetos das colunas
    targets = ind_ranking[links]

    # inicializa funcao para mapeamento dos vertices
    map_vertices = np.zeros(len(ind_ranking)).astype(int) - 1

    # inicializa variavel para guardar o calculo das medidas
    measures = np.zeros((len(med), numclasses))
    for l in range(numclasses):

        # captura todos os objetos em sources que pertencem a classe l
        lsources = np.where(y[sources] == l)[0]

        # captura todos os objetos da base que pertencem a classe l
        all_vertices = np.where(y == l)[0]

        # recebe todos os objetos da classe l que estao conectados
        unique_vertices = np.unique(
            np.append(sources[lsources], targets[lsources])).astype(int)

        # recebe os demais objetos da classe l que nao estao conectados
        unique_vertices = np.unique(
            np.append(unique_vertices, all_vertices)).astype(int)

        # popula a funcao de mapeamento dos vertices com os objetos da classe l
        map_vertices[unique_vertices] = np.arange(
            len(unique_vertices)).astype(int)

        # print(len(unique_vertices), unique_vertices)

        # cria o grafo para classe l
        subg = ig.Graph(len(unique_vertices), list(zip(map_vertices[sources[lsources]].astype(
            int), map_vertices[targets[lsources]].astype(int))))

        #print(subg.vcount())

        # popula a lista de grafos
        lista_grafos.append(subg)

        medida_ind = 0
        # calcula as medidas de rede associadas ao grafo l
        if 'A' in med:
            measures[medida_ind][l] = Assortativity(subg)
            medida_ind += 1
        if 'B' in med:
            measures[medida_ind][l] = ClustCoefficient(subg)
            medida_ind += 1
        if 'C' in med:
            measures[medida_ind][l] = AvgDegree(subg)
            medida_ind += 1
        if 'D' in med:
            measures[medida_ind][l] = Betweenness(subg)
            medida_ind += 1
        if 'E' in med:
            measures[medida_ind][l] = AvgPathLength(subg)
            medida_ind += 1
        if 'F' in med:
            measures[medida_ind][l] = Closeness(subg)
            medida_ind += 1

    return lista_grafos, map_vertices, measures


def VS_Graph_vector(k, X, y, med, dist, typ):
    unique_classes = np.unique(y)
    n_samples = len(X)

    # Construção dos grafos
    g = [graphFromSeriesNetworkX(X[i]) for i in range(n_samples)]

    # Cálculo de centralidade
    if typ == 'bet':
        betw = [list(nx.betweenness_centrality(g[i], normalized=False).values()) for i in range(n_samples)]
        centrality = betw
    elif typ == 'degree':
        degree = [dict(nx.degree(g[i])).values() for i in range(n_samples)]
        centrality = degree
    else:
        raise ValueError(f"Tipo '{typ}' não reconhecido. Use 'bet' ou 'degree'.")

    if dist != 'cosine':
        dist = DistanceMetric.get_metric(dist)
        euclidean_dist = dist.pairwise(centrality)
    elif dist == 'cosine':
        euclidean_dist = pairwise.cosine_similarity(centrality)

    np.fill_diagonal(euclidean_dist, np.inf)
    # ordena as distancias por objeto em ordem ascendente
    ind_ranking = np.argsort(euclidean_dist, axis=1)[:, :k]
    # inicializa uma mascara com False, num de objetos x k
    mask = np.zeros((len(X), k)).astype(int)
    lista_grafos = []
    for i in range(len(ind_ranking)):
        mask[i] = (y[ind_ranking[i]] == y[i])

    # captura os indices True da mascara, linha e coluna
    links = mask.nonzero()

    # atribui os indices das linhas
    sources = links[0]

    # atribui o indice real dos objetos das colunas
    targets = ind_ranking[links]

    # inicializa funcao para mapeamento dos vertices
    map_vertices = np.zeros(len(ind_ranking)).astype(int) - 1

    # inicializa variavel para guardar o calculo das medidas
    measures = np.zeros(numclasses)
    for l in range(numclasses):

        # captura todos os objetos em sources que pertencem a classe l
        lsources = np.where(y[sources] == l)[0]

        # captura todos os objetos da base que pertencem a classe l
        all_vertices = np.where(y == l)[0]

        # recebe todos os objetos da classe l que estao conectados
        unique_vertices = np.unique(
          np.append(sources[lsources], targets[lsources])).astype(int)

        # recebe os demais objetos da classe l que nao estao conectados
        unique_vertices = np.unique(
          np.append(unique_vertices, all_vertices)).astype(int)

        # popula a funcao de mapeamento dos vertices com os objetos da classe l
        map_vertices[unique_vertices] = np.arange(
          len(unique_vertices)).astype(int)

        # print(len(unique_vertices), unique_vertices)

        # cria o grafo para classe l
        subg = ig.Graph(len(unique_vertices), list(zip(map_vertices[sources[lsources]].astype(
          int), map_vertices[targets[lsources]].astype(int))))

        # popula a lista de grafos
        lista_grafos.append(subg)

        medida_ind = 0
        # calcula as medidas de rede associadas ao grafo l
        if 'A' in med:
            measures[medida_ind][l] = Assortativity(subg)
            medida_ind += 1
        if 'B' in med:
            measures[medida_ind][l] = ClustCoefficient(subg)
            medida_ind += 1
        if 'C' in med:
            measures[medida_ind][l] = AvgDegree(subg)
            medida_ind += 1
        if 'D' in med:
            measures[medida_ind][l] = Betweenness(subg)
            medida_ind += 1
        if 'E' in med:
            measures[medida_ind][l] = AvgPathLength(subg)
            medida_ind += 1
        if 'F' in med:
            measures[medida_ind][l] = Closeness(subg)
            medida_ind += 1

    return lista_grafos, map_vertices, measures, centrality


def VS_Graph_similarity(k, X, y, med):
    numclasses = len(np.unique(y))

    g = [graphFromSeriesNetworkX(x) for x in X]

    matriz_simi = matriz_similaridade_jaccard_nx(g)

    np.fill_diagonal(matriz_simi, 0)
    # ordena as distancias por objeto em ordem ascendente
    ind_ranking = np.argsort(matriz_simi, axis=1)[:, -k:]
    # inicializa uma mascara com False, num de objetos x k
    mask = np.zeros((len(X), k)).astype(int)
    lista_grafos = []
    for i in range(len(ind_ranking)):
        mask[i] = (y[ind_ranking[i]] == y[i])

    # captura os indices True da mascara, linha e coluna
    links = mask.nonzero()

    # atribui os indices das linhas
    sources = links[0]

    # atribui o indice real dos objetos das colunas
    targets = ind_ranking[links]

    # inicializa funcao para mapeamento dos vertices
    map_vertices = np.zeros(len(ind_ranking)).astype(int) - 1

    # inicializa variavel para guardar o calculo das medidas
    measures = np.zeros(numclasses)
    for l in range(numclasses):

        # captura todos os objetos em sources que pertencem a classe l
        lsources = np.where(y[sources] == l)[0]

        # captura todos os objetos da base que pertencem a classe l
        all_vertices = np.where(y == l)[0]

        # recebe todos os objetos da classe l que estao conectados
        unique_vertices = np.unique(
            np.append(sources[lsources], targets[lsources])).astype(int)

        # recebe os demais objetos da classe l que nao estao conectados
        unique_vertices = np.unique(
            np.append(unique_vertices, all_vertices)).astype(int)

        # popula a funcao de mapeamento dos vertices com os objetos da classe l
        map_vertices[unique_vertices] = np.arange(
            len(unique_vertices)).astype(int)

        # print(len(unique_vertices), unique_vertices)

        # cria o grafo para classe l
        subg = ig.Graph(len(unique_vertices), list(zip(map_vertices[sources[lsources]].astype(
            int), map_vertices[targets[lsources]].astype(int))))

        # ig.plot(subg)
        # print(subg.vcount())

        # popula a lista de grafos
        lista_grafos.append(subg)

        medida_ind = 0
        # calcula as medidas de rede associadas ao grafo l
        if 'A' in med:
            measures[medida_ind][l] = Assortativity(subg)
            medida_ind += 1
        if 'B' in med:
            measures[medida_ind][l] = ClustCoefficient(subg)
            medida_ind += 1
        if 'C' in med:
            measures[medida_ind][l] = AvgDegree(subg)
            medida_ind += 1
        if 'D' in med:
            measures[medida_ind][l] = Betweenness(subg)
            medida_ind += 1
        if 'E' in med:
            measures[medida_ind][l] = AvgPathLength(subg)
            medida_ind += 1
        if 'F' in med:
            measures[medida_ind][l] = Closeness(subg)
            medida_ind += 1

    return lista_grafos, map_vertices, measures, g


def graphFromSeriesNetworkX(serie, n=13):
    """
    Constrói um grafo a partir de uma série temporal usando NetworkX.

    Parâmetros:
        serie (list ou array): Sequência de valores da série temporal.
        n (int): Número máximo de conexões para cada nó.

    Retorna:
        nx.Graph: Grafo gerado.
    """
    g = nx.Graph()
    num_series = len(serie)

    # Adiciona nós ao grafo
    g.add_nodes_from(range(num_series))

    # Ajusta n para não ultrapassar o limite
    n = min(n, num_series)

    # Conectar os nós
    for i in range(num_series):
        ya = serie[i]

        for j in range(i + 1, min(i + n, num_series)):  # Evita autoconexão
            yb = serie[j]
            ta, tb = i, j

            # Verificar se há interseção no caminho
            if not any(serie[tc] > yb + (ya - yb) * ((tb - tc) / (tb - ta)) for tc in range(ta + 1, tb)):
                g.add_edge(ta, tb)

    return g


def jaccard_similarity(grafo1, grafo2):
    """Calcula a similaridade Jaccard média entre os nós de dois grafos."""
    if len(grafo1.nodes) != len(grafo2.nodes):
        raise ValueError("Todos os grafos devem ter o mesmo número de vértices para calcular a similaridade de Jaccard.")

    # Pré-computa as vizinhanças dos nós
    vizinhancas1 = {n: set(grafo1.neighbors(n)) for n in grafo1.nodes}
    vizinhancas2 = {n: set(grafo2.neighbors(n)) for n in grafo2.nodes}

    # Calcula Jaccard para cada nó e retorna a média
    return np.mean([
        len(vizinhancas1[v] & vizinhancas2[v]) / len(vizinhancas1[v] | vizinhancas2[v])
        if vizinhancas1[v] | vizinhancas2[v] else 0
        for v in grafo1.nodes
    ])


def matriz_similaridade_jaccard_nx(lista_grafos):
    """
    Calcula a matriz de similaridade Jaccard entre múltiplos grafos.

    Parâmetros:
        lista_grafos (list): Lista de networkx.Graph.

    Retorna:
        np.ndarray: Matriz de similaridade entre os grafos.
    """
    num_grafos = len(lista_grafos)
    matriz_similaridade = np.zeros((num_grafos, num_grafos))

    # Pré-computa as vizinhanças dos nós para cada grafo
    vizinhancas = [{n: set(g.neighbors(n)) for n in g.nodes} for g in lista_grafos]

    for i in range(num_grafos):
        for j in range(i, num_grafos):  # Apenas a metade superior da matriz
            if len(lista_grafos[i].nodes) != len(lista_grafos[j].nodes):
                raise ValueError("Todos os grafos devem ter o mesmo número de vértices para calcular a similaridade de Jaccard.")

            # Calcula similaridade Jaccard diretamente com NumPy
            similaridades = np.array([
                len(vizinhancas[i][v] & vizinhancas[j][v]) / len(vizinhancas[i][v] | vizinhancas[j][v])
                if vizinhancas[i][v] | vizinhancas[j][v] else 0
                for v in lista_grafos[i].nodes
            ])

            # Média das similaridades
            similaridade_media = np.mean(similaridades)

            # Plotar as redes com muita similaridade
            # if similaridade_media >= 0.95 and similaridade_media != 1:
            #     print(similaridade_media)
            #     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            #     pos1 = nx.spring_layout(lista_grafos[i], seed=42, k=0.3)
            #     pos2 = nx.spring_layout(lista_grafos[j], seed=42, k=0.3)
            #     plt.sca(axes[0])
            #     nx.draw(lista_grafos[i], pos=pos1,  with_labels=False, node_size=15, node_color='skyblue', edge_color='gray', width=0.5)
            #     plt.sca(axes[1])
            #     nx.draw(lista_grafos[j], pos=pos2,  with_labels=False, node_size=15, node_color='skyblue', edge_color='gray', width=0.5)
            #     plt.show()
            matriz_similaridade[i, j] = matriz_similaridade[j, i] = similaridade_media  # Matriz simétrica

    return matriz_similaridade


def matriz_similaridade_jaccard_2_nx(X, Y, n_jobs=-1):
    """
    Calcula a matriz de similaridade Jaccard entre dois conjuntos de grafos.

    Parâmetros:
        X (list): Lista de networkx.Graph (linhas da matriz).
        Y (list): Lista de networkx.Graph (colunas da matriz).
        n_jobs (int): Número de threads paralelas (-1 usa todas as disponíveis).

    Retorna:
        np.ndarray: Matriz de similaridade entre os grafos de X e Y.
    """
    num_grafos_X, num_grafos_Y = len(X), len(Y)
    matriz_similaridade = np.zeros((num_grafos_X, num_grafos_Y))

    resultados = Parallel(n_jobs=n_jobs)(
        delayed(jaccard_similarity)(X[i], Y[j]) for i in range(num_grafos_X) for j in range(num_grafos_Y)
    )

    # Preenche a matriz com os resultados calculados em paralelo
    for idx, (i, j) in enumerate([(i, j) for i in range(num_grafos_X) for j in range(num_grafos_Y)]):
        matriz_similaridade[i, j] = resultados[idx]

    return matriz_similaridade


def Assortativity(graph):
    return ig.Graph.assortativity_degree(graph)


def ClustCoefficient(graph):
    return ig.Graph.transitivity_avglocal_undirected(graph)


def AvgDegree(graph):
    return np.mean(ig.Graph.degree(graph))


def Betweenness(graph):
    return np.mean(ig.Graph.betweenness(graph))


def AvgPathLength(graph):
    return ig.Graph.average_path_length(graph)


def Closeness(graph):
    return np.mean(ig.Graph.closeness(graph))


def deep_copy_graphs(lista_grafos):
    newlist = []
    for i in range(len(lista_grafos)):
        newlist.append(ig.Graph.copy(lista_grafos[i]))

    return newlist