# !/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
import networkx as nx
import itertools
import pickle
from lib.feature.util import *


def get_clique_feature(data):
    graph = load_graph()
    clique_len = dict()
    clique = dict()
    for c in nx.find_cliques(graph):
        len_c= len(c)
        id = c[0]
        for item in c:
            clique_len[item] = max(clique_len.get(item,0), len_c)
            clique[item] = id
    clique_values = list(set(clique.values()))
    # q1所属的团的大小
    data['clique_q1_size'] = data.q1.apply(lambda x:clique_len.get(x,-1))
    data['clique_q2_size'] = data.q2.apply(lambda x:clique_len.get(x,-1))
    # q1所属团, 用该团的第一个元素表示
    data['clique_q1'] = data.q1.apply(lambda x: clique_values.index(clique[x]) if x in clique else -1)
    data['clique_q2'] = data.q2.apply(lambda x: clique_values.index(clique[x]) if x in clique else -1)
    return data

def get_connected_feature(data):
    g = load_graph()
    connected_len = dict()
    connected = dict()
    for c in nx.connected_components(g):
        len_c = len(c)
        id = list(c)[0]
        for item in c:
            connected_len[item] = max(connected_len.get(item,0), len_c)
            connected[item] = id
    connected_values = list(set(connected.values()))
    # q所属连通子图的大小
    data['connected_q1_size'] = data.q1.apply(lambda x:connected_len.get(x,-1))
    data['connected_q2_size'] = data.q2.apply(lambda x:connected_len.get(x,-1))
    data['connected_q1'] = data.q1.apply(lambda x: connected_values.index(connected[x])
                                                    if x in connected else -1)
    data['connected_q2'] = data.q2.apply(lambda x: connected_values.index(connected[x])
                                                    if x in connected else -1)
    return data

def get_nodes_features(data):
    graph = load_graph()
    data['connected_q1_neighbors'] = data.q1.apply(lambda x: list(graph.neighbors(x)) if x in graph
                                                   else [])
    data['connected_q2_neighbors'] = data.q2.apply(lambda x: list(graph.neighbors(x)) if x in graph
                                                   else [])
    data['connected_q1_neighbors_size'] = data.connected_q1_neighbors.apply(lambda x: len(x))
    data['connected_q2_neighbors_size'] = data.connected_q2_neighbors.apply(lambda x: len(x))
    q1_neighbors= list(set(list(data['connected_q1_neighbors'].sum())))
    q2_neighbors= list(set(list(data['connected_q2_neighbors'].sum())))
    q1_dict = dict(zip(q1_neighbors,range(len(q1_neighbors))))
    q2_dict = dict(zip(q2_neighbors,range(len(q2_neighbors))))
    data['connected_q1_neighbors'] = data.connected_q1_neighbors.\
        apply(lambda x: ([q1_dict.get(i) for i in x[:50]] if x is not [] else [])+max(0,50-len(x))*[-1])
    data['connected_q2_neighbors'] = data.connected_q2_neighbors.\
        apply(lambda x: ([q2_dict.get(i) for i in x[:50]] if x is not [] else [])+max(0,50-len(x))*[-1])
    return data

def get_edges_features(data):
    def min_len(x,g):
        q1 = x['q1']
        q2 = x['q2']
        if q1 in g and q2 in g:
            prev = set()
            prev.add(q1)
            neighbors = list(zip(list(g.neighbors(q1)), len(list(g.neighbors(q1)))*[1]))
            for n in neighbors:
                if n[0] == q2:
                    return n[1]
                if n[0] not in prev:
                    neighbors.extend(list(zip(list(g.neighbors(n[0])),
                                              len(list(g.neighbors(n[0])))*[n[1]+1])))
                    prev.add(n[0])
        return -1
    graph = load_graph()
    data['min_len'] = data.apply(lambda x: min_len(x,graph), axis=1)
    return data













