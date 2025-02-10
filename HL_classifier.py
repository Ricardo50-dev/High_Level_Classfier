import numpy as np
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import pairwise
import graph_generator as gn
import graph_generator_selected as gns
import networkx as nx
import math as mt


class HLNB_PC:

    """
        High Level Network-Based by pattern conformation (HLNB_PC) - Hybrid Classifier
        Alto nível baseado em redes por conformidade padrão - Classificador Híbrido
        Args:
            neighbors: Quantidade de vértices que cada amostra será conectada
            dst: Medida de distância entre amostras
            nameMesure: Nome da medida de rede ('Assortativity', 'ClustCoefficient', 'AvgDegree', 'Betweenness', 'AvgPathLength', 'Closeness')
            heuristic: Nome da técnica de construção de rede ('visibility', 'visibility_vector', 'knn', 's_knn', 'm_knn', 's_m_knn')
            weight: Peso da medida (Sempre 1, pois não há combinação de medidas)
            typ: Tipo do vetor para a técnica de construção de rede: 'visibility_vector' ('bet', 'degree')
            lambda_HC: Valor da influência da classificação de alto nível
    """

    def __init__(self, neighbors=3, dst='euclidean', nameMesure='Assortativity', heuristic='visibility', weight=1, typ='degree', lambda_HC=0.8):
        self.fitted = False
        self.lista_grafos = None
        self.map_vertices = None
        self.measures = None
        self.nameOfMesure = nameMesure
        self.heuristic = heuristic
        self.X = None
        self.X_visibility = None
        self.X_vector_measures = None
        self.y = None
        self.k = neighbors
        self.dist = dst
        self.weight = weight
        self.typ = typ
        self.lambda_HC = lambda_HC

    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        if self.heuristic == 'visibility':
            self.lista_grafos, self.map_vertices, self.measures, self.X_visibility = gn.VS_Graph_similarity(self.k, X_train, y_train, self.nameOfMesure)
            self.fitted = True
        elif self.heuristic == 'visibility_vector':
            if self.typ == 'degree':
                self.lista_grafos, self.map_vertices, self.measures, self.X_vector_measures = gn.VS_Graph_vector(self.k, X_train, y_train, self.nameOfMesure, self.dist, self.typ)
                self.fitted = True
            elif self.typ == 'bet':
                self.lista_grafos, self.map_vertices, self.measures, self.X_vector_measures = gn.VS_Graph_vector(self.k, X_train, y_train, self.nameOfMesure, self.dist, self.typ)
                self.fitted = True
        elif self.heuristic == 'knn':
            self.lista_grafos, self.map_vertices, self.measures = gn.kNN_Graph(self.k, X_train, y_train, self.nameOfMesure, self.dist)
            self.fitted = True
        elif self.heuristic == 's_knn':
            self.lista_grafos, self.map_vertices, self.measures = gn.S_kNN_Graph(self.k, X_train, y_train, self.nameOfMesure, self.dist)
            self.fitted = True
        elif self.heuristic == 'm_knn':
            self.lista_grafos, self.map_vertices, self.measures = gn.M_kNN_Graph(self.k, X_train, y_train, self.nameOfMesure, self.dist)
            self.fitted = True
        elif self.heuristic == 's_m_knn':
            self.lista_grafos, self.map_vertices, self.measures = gn.S_M_kNN_Graph(self.k, X_train, y_train, self.nameOfMesure, self.dist)
            self.fitted = True

    def predict_proba(self, X_test):
        if not self.fitted:
            raise NoTrainException("To classify a new sample it is necessary to train the model")

        n_classes = len(np.unique(self.y))
        n_samples = len(X_test)

        # Inicializações eficientes
        proba_high_level = np.full((n_samples, n_classes), np.inf)
        den_final = np.zeros(n_samples)
        proportion = np.array([np.sum(self.y == c) / self.X.shape[0] for c in range(n_classes)], dtype=np.float32)
        predicted = np.full((n_samples, n_classes), np.inf)
        prob = np.full((n_samples, n_classes), np.inf)
        den = np.zeros(n_samples)

        # Cálculo das distâncias baseado na heurística
        if self.heuristic in ['visibility', 'visibility_vector']:
            X_test_visibility = [gn.graphFromSeriesNetworkX(x) for x in X_test]

            if self.heuristic == 'visibility':
                dist = gn.matriz_similaridade_jaccard_2_nx(X_test_visibility, self.X_visibility)
            else:
                X_test_features = []
                for g in X_test_visibility:
                    if self.heuristic == 'bet':
                        X_test_features.append(list(nx.betweenness_centrality(g, normalized=False).values()))
                    else:  # visibility_degree
                        X_test_features.append([deg for _, deg in g.degree()])

                X_test_features = np.array(X_test_features)

                if self.dist == 'cosine':
                    dist = pairwise.cosine_similarity(X_test_features, self.X_vector_measures)
                else:
                    dist = DistanceMetric.get_metric(self.dist).pairwise(X_test_features, self.X_vector_measures)
        else:
            dist = (
                pairwise.cosine_similarity(X_test, self.X)
                if self.dist == 'cosine'
                else DistanceMetric.get_metric(self.dist).pairwise(X_test, self.X)
            )

        # Processamento dos vizinhos
        for i in range(n_samples):
            ids = np.argsort(dist[i])[-self.k:] if self.heuristic == 'visibility' else np.argsort(dist[i])[:self.k]

            # newgraph recebe uma copia de lista_grafos
            newgraph = gn.deep_copy_graphs(self.lista_grafos)
            # j eh o indice do objeto de treino e c a classe dele
            for j, c in enumerate(self.y[ids]):
                newid = self.lista_grafos[c].vcount()
                newgraph[c].add_vertex(newid)

                # verifica se o vertice em que o novo objeto sera conectado ja esta no grafo
                if self.map_vertices[ids[j]] > newid:
                    newgraph[c].add_vertex(self.map_vertices[ids[j]])

                newgraph[c].add_edge(newid, self.map_vertices[ids[j]])

            # para cada uma das classes MODIFICADO
            for c in np.unique(self.y):
                subg = newgraph[c]
                # calcula a variação das medidas
                if self.nameOfMesure == 'Assortativity':
                    predicted[i, c] = abs(gn.Assortativity(subg) - self.measures[c])  # Será o próprio (Numerador)
                    # Modificado7
                    if mt.isnan(predicted[i, c]):
                        predicted[i, c] = np.nan_to_num(predicted[i, c])

                    if predicted[i, c] == 0.0:
                        predicted[i, c] = 10000

                    den[i] += predicted[i, c]  # Somatório da diferença das 2 classes, (Denominador)
                elif self.nameOfMesure == 'ClustCoefficient':
                    predicted[i, c] = abs(gn.ClustCoefficient(subg) - self.measures[c])  # Será o próprio (Numerador)
                    # Modificado
                    if mt.isnan(predicted[i, c]):
                        predicted[i, c] = np.nan_to_num(predicted[i, c])

                    if predicted[i, c] == 0.0:
                        predicted[i, c] = 10000

                    den[i] += predicted[i, c]  # Somatório da diferença das 2 classes, (Denominador)
                elif self.nameOfMesure == 'AvgDegree':
                    predicted[i, c] = abs(gn.AvgDegree(subg) - self.measures[c])  # Será o próprio (Numerador)
                    # Modificado
                    if mt.isnan(predicted[i, c]):
                        predicted[i, c] = np.nan_to_num(predicted[i, c])

                    if predicted[i, c] == 0.0:
                        predicted[i, c] = 10000

                    den[i] += predicted[i, c]  # Somatório da diferença das 2 classes, (Denominador)
                elif self.nameOfMesure == 'Betweenness':
                    predicted[i, c] = abs(gn.Betweenness(subg) - self.measures[c])  # Será o próprio (Numerador)
                    # Modificado
                    if mt.isnan(predicted[i, c]):
                        predicted[i, c] = np.nan_to_num(predicted[i, c])

                    if predicted[i, c] == 0.0:
                        predicted[i, c] = 10000

                    den[i] += predicted[i, c]  # Somatório da diferença das 2 classes, (Denominador)
                elif self.nameOfMesure == 'AvgPathLength':
                    predicted[i, c] = abs(gn.AvgPathLength(subg) - self.measures[c])  # Será o próprio (Numerador)
                    # Modificado
                    if mt.isnan(predicted[i, c]):
                        predicted[i, c] = np.nan_to_num(predicted[i, c])

                    if predicted[i, c] == 0.0:
                        predicted[i, c] = 10000

                    den[i] += predicted[i, c]  # Somatório da diferença das 2 classes, (Denominador)
                elif self.nameOfMesure == 'Closeness':
                    predicted[i, c] = abs(gn.Closeness(subg) - self.measures[c])  # Será o próprio (Numerador)
                    # Modificado
                    if mt.isnan(predicted[i, c]):
                        predicted[i, c] = np.nan_to_num(predicted[i, c])

                    if predicted[i, c] == 0.0:
                        predicted[i, c] = 10000

                    den[i] += predicted[i, c]  # Somatório da diferença das 2 classes, (Denominador)

            # para cada uma das classes MODIFICADO
            for c in np.unique(self.y):
                prob[i, c] = (predicted[i, c] / den[i]) * proportion[c]
                den_final[i] += self.weight * (1 - prob[i, c])

            for c in np.unique(self.y):
                proba_high_level[i, c] = (self.weight * (1 - prob[i, c])) / den_final[i]

            # print(proba_high_level)
            # MODIFICADO
            # target = np.argmin(proba_high_level, axis=1)
        return proba_high_level

    def predict(self, X_test):
        target = np.argmax(self.predict_proba(X_test), axis=1)
        return target

    def predict_hybrid(self, X_test, proba_LL):
        proba_HL = self.predict_proba(X_test)
        ind_linha = proba_HL.shape[0]
        ind_col = proba_HL.shape[1]
        proba_hybrid = np.zeros((ind_linha, ind_col))

        for i in range(ind_linha):
            for c in range(ind_col):
                proba_hybrid[i][c] = ((1 - self.lambda_HC) * proba_LL[i][c]) + (self.lambda_HC * proba_HL[i][c])

        target = np.argmax(proba_hybrid, axis=1)

        return target


class HLNB_PCMC:

    """
        High Level Network-Based by pattern conformation with measure combination (HLNB_PCMC) - Hybrid Classifier
        Alto nível baseado em redes por conformidade padrão com combinação de medidas - Classificador Híbrido
        Args:
            neighbors: Quantidade de vértices que cada amostra será conectada
            dst: Medida de distância entre amostras
            heuristic: Nome da técnica de construção de rede ('visibility', 'visibility_vector', 'knn', 's_knn', 'm_knn', 's_m_knn')
            typ: Tipo do vetor para a técnica de construção de rede: 'visibility_vector' ('bet', 'degree')
            lambda_HC: Valor da influência da classificação de alto nível
            weight: Peso das medidas de rede (o somatório dos pesos deve ser 1),
                    => A quantidade e ordem de pesos deve ser a mesma fornecida pelas medidas a serem combinadas
            nameOfMesures: A => Assortativity, B => ClustCoefficient, C => AvgDegree
                           D => Betweenness, E => AvgPathLength, F => Closeness
                           => A sequência de letras indicam quais medidas serão consideradas
                           => A sequência de letras também indica o valor dos pesos dado como entrada
                           => !!! A sequência deve estar na ordem alfabética:
                                Ex (Erro): CBA
                                Ex (Correto): ABC
    """

    def __init__(self, neighbors=3, dst='euclidean', nameOfMesures='ABC', heuristic='visibility', weight=[0.4, 0.3, 0.3], typ='degree', lambda_HC=0.8):
        self.fitted = False
        self.lista_grafos = None
        self.map_vertices = None
        self.measures = None
        self.nameOfMesures = nameOfMesures
        self.heuristic = heuristic
        self.X = None
        self.X_visibility = None
        self.X_vector_measures = None
        self.y = None
        self.k = neighbors
        self.dist = dst
        self.weight = weight
        self.typ = typ
        self.lambda_HC = lambda_HC

    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        if self.heuristic == 'visibility':
            self.lista_grafos, self.map_vertices, self.measures, self.X_visibility = gns.VS_Graph_similarity(self.k,
                                                                                                            X_train,
                                                                                                            y_train,
                                                                                                            self.nameOfMesures)
            self.fitted = True
        elif self.heuristic == 'visibility_vector':
            if self.typ == 'degree':
                self.lista_grafos, self.map_vertices, self.measures, self.X_visibility = gns.VS_Graph_vector(self.k,
                                                                                                            X_train,
                                                                                                            y_train,
                                                                                                            self.nameOfMesures,
                                                                                                            self.dist,
                                                                                                            self.typ)
                self.fitted = True
            elif self.typ == 'bet':
                self.lista_grafos, self.map_vertices, self.measures, self.X_visibility = gns.VS_Graph_vector(self.k,
                                                                                                            X_train,
                                                                                                            y_train,
                                                                                                            self.nameOfMesures,
                                                                                                            self.dist,
                                                                                                            self.typ)
                self.fitted = True
        elif self.heuristic == 'knn':
            self.lista_grafos, self.map_vertices, self.measures = gns.kNN_Graph(self.k, X_train, y_train,
                                                                               self.nameOfMesures, self.dist)
            self.fitted = True
        elif self.heuristic == 's_knn':
            self.lista_grafos, self.map_vertices, self.measures = gns.S_kNN_Graph(self.k, X_train, y_train,
                                                                                 self.nameOfMesures, self.dist)
            self.fitted = True
        elif self.heuristic == 'm_knn':
            self.lista_grafos, self.map_vertices, self.measures = gns.M_kNN_Graph(self.k, X_train, y_train,
                                                                                 self.nameOfMesures, self.dist)
            self.fitted = True
        elif self.heuristic == 's_m_knn':
            self.lista_grafos, self.map_vertices, self.measures = gns.S_M_kNN_Graph(self.k, X_train, y_train,
                                                                                   self.nameOfMesures, self.dist)
            self.fitted = True

    def predict_proba(self, X_test):
        if not self.fitted:
            raise NoTrainException("To classify a new sample it is necessary to train the model")

        n_classes = len(np.unique(self.y))
        n_samples = len(X_test)

        # Inicializações eficientes
        proba_high_level = np.full((n_samples, n_classes), np.inf)
        den_final = np.zeros(n_samples)
        proportion = np.array([np.sum(self.y == c) / self.X.shape[0] for c in range(n_classes)], dtype=np.float32)
        prob = np.zeros((n_samples, n_classes))
        variation = np.zeros((len(self.nameOfMesures), n_classes))
        variation_den = np.zeros((len(self.nameOfMesures), n_samples))

        # Cálculo das distâncias baseado na heurística
        if self.heuristic in ['visibility', 'visibility_vector']:
            X_test_visibility = [gn.graphFromSeriesNetworkX(x) for x in X_test]

            if self.heuristic == 'visibility':
                dist = gn.matriz_similaridade_jaccard_2_nx(X_test_visibility, self.X_visibility)
            else:
                X_test_features = []
                for g in X_test_visibility:
                    if self.typ == 'bet':
                        X_test_features.append(list(nx.betweenness_centrality(g, normalized=False).values()))
                    else:  # visibility_degree
                        X_test_features.append([deg for _, deg in g.degree()])

                X_test_features = np.array(X_test_features)

                if self.dist == 'cosine':
                    dist = pairwise.cosine_similarity(X_test_features, self.X_vector_measures)
                else:
                    dist = DistanceMetric.get_metric(self.dist).pairwise(X_test_features, self.X_vector_measures)
        else:
            dist = (
                pairwise.cosine_similarity(X_test, self.X)
                if self.dist == 'cosine'
                else DistanceMetric.get_metric(self.dist).pairwise(X_test, self.X)
            )

        # Processamento dos vizinhos
        for i in range(n_samples):
            ids = np.argsort(dist[i])[-self.k:] if self.heuristic == 'visibility' else np.argsort(dist[i])[:self.k]

            # Cópia otimizada dos grafos
            newgraph = gn.deep_copy_graphs(self.lista_grafos)

            for j, c in enumerate(self.y[ids]):
                newid = self.lista_grafos[c].vcount()
                newgraph[c].add_vertex(newid)

                mapped_vertex = self.map_vertices[ids[j]]
                if mapped_vertex > newid:
                    newgraph[c].add_vertex(mapped_vertex)

                newgraph[c].add_edge(newid, mapped_vertex)

            if len(self.nameOfMesures) != len(self.weight):
                raise ValueError("The number of weights must match the number of measurements selected.")

            # para cada uma das classes MODIFICADO
            for c in np.unique(self.y):
                subg = newgraph[c]
                ind_med = 0
                # calcula a variação das medidas
                if 'A' in self.nameOfMesures:
                    variation[ind_med][c] = abs(gn.Assortativity(subg) - self.measures[ind_med][c])  # Será o próprio (Numerador)
                    # Modificado7
                    if mt.isnan(variation[ind_med][c]):
                        variation[ind_med][c] = np.nan_to_num(variation[ind_med][c])

                    if variation[ind_med][c] == 0.0:
                        variation[ind_med][c] = 10000

                    variation_den[ind_med][i] += variation[ind_med][c]  # Somatório da diferença das 2 classes, (Denominador)
                    ind_med += 1
                if 'B' in self.nameOfMesures:
                    variation[ind_med][c] = abs(gn.ClustCoefficient(subg) - self.measures[ind_med][c])  # Será o próprio (Numerador)
                    # Modificado
                    if mt.isnan(variation[ind_med][c]):
                        variation[ind_med][c] = np.nan_to_num(variation[ind_med][c])

                    if variation[ind_med][c] == 0.0:
                        variation[ind_med][c] = 10000

                    variation_den[ind_med][i] += variation[ind_med][c]  # Somatório da diferença das 2 classes, (Denominador)
                    ind_med += 1
                if 'C' in self.nameOfMesures:
                    variation[ind_med][c] = abs(gn.AvgDegree(subg) - self.measures[ind_med][c])  # Será o próprio (Numerador)
                    # Modificado
                    if mt.isnan(variation[ind_med][c]):
                        variation[ind_med][c] = np.nan_to_num(variation[ind_med][c])

                    if variation[ind_med][c] == 0.0:
                        variation[ind_med][c] = 10000

                    variation_den[ind_med][i] += variation[ind_med][c]  # Somatório da diferença das 2 classes, (Denominador)
                    ind_med += 1
                if 'D' in self.nameOfMesures:
                    variation[ind_med][c] = abs(gn.Betweenness(subg) - self.measures[ind_med][c])  # Será o próprio (Numerador)
                    # Modificado
                    if mt.isnan(variation[ind_med][c]):
                        variation[ind_med][c] = np.nan_to_num(variation[ind_med][c])

                    if variation[ind_med][c] == 0.0:
                        variation[ind_med][c] = 10000

                    variation_den[ind_med][i] += variation[ind_med][c]  # Somatório da diferença das 2 classes, (Denominador)
                    ind_med += 1
                if 'E' in self.nameOfMesures:
                    variation[ind_med][c] = abs(gn.AvgPathLength(subg) - self.measures[ind_med][c])  # Será o próprio (Numerador)
                    # Modificado
                    if mt.isnan(variation[ind_med][c]):
                        variation[ind_med][c] = np.nan_to_num(variation[ind_med][c])

                    if variation[ind_med][c] == 0.0:
                        variation[ind_med][c] = 10000

                    variation_den[ind_med][i] += variation[ind_med][c]  # Somatório da diferença das 2 classes, (Denominador)
                    ind_med += 1
                if 'F' in self.nameOfMesures:
                    variation[ind_med][c] = abs(gn.Closeness(subg) - self.measures[ind_med][c])  # Será o próprio (Numerador)
                    # Modificado
                    if mt.isnan(variation[ind_med][c]):
                        variation[ind_med][c] = np.nan_to_num(variation[ind_med][c])

                    if variation[ind_med][c] == 0.0:
                        variation[ind_med][c] = 10000

                    variation_den[ind_med][i] += variation[ind_med][c]  # Somatório da diferença das 2 classes, (Denominador)
                    ind_med += 1
            # para cada uma das classes MODIFICADO
            for c in np.unique(self.y):
                for ind_medidas in range(ind_med):
                    prob[i, c] = self.weight[ind_medidas] * (1 - ((variation[ind_medidas][c] / variation_den[ind_medidas][i]) * proportion[c]))
                den_final[i] += prob[i, c]

            for c in np.unique(self.y):
                proba_high_level[i, c] = prob[i, c] / den_final[i]

        return proba_high_level

    def predict(self, X_test):
        target = np.argmax(self.predict_proba(X_test), axis=1)
        return target

    def predict_hybrid(self, X_test, proba_LL):
        proba_HL = self.predict_proba(X_test)
        ind_linha = proba_HL.shape[0]
        ind_col = proba_HL.shape[1]
        proba_hybrid = np.zeros((ind_linha, ind_col))

        for i in range(ind_linha):
            for c in range(ind_col):
                proba_hybrid[i][c] = ((1 - self.lambda_HC) * proba_LL[i][c]) + (self.lambda_HC * proba_HL[i][c])

        target = np.argmax(proba_hybrid, axis=1)

        return target


class NoTrainException(Exception):
    """the model is not trained"""

    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"{self.message}"