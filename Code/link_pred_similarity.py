#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 17:14:28 2023

@author: nathancasanova
"""

import numpy as np
import pandas as pd
import igraph as ig
import cairo
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
from sknetwork.classification import get_accuracy_score
from sknetwork.data import from_graphml
from sknetwork.linkpred import AdamicAdar,CommonNeighbors,HubDepressedIndex, HubPromotedIndex, \
    JaccardIndex, PreferentialAttachment, ResourceAllocation, SaltonIndex, SorensenIndex,  \
    is_edge, whitened_sigmoid
from sknetwork.utils import get_weights
from link_pred_feature_engineer import sum_stats
import random

rng = 2023
random.seed(rng)
#%% Helper Functions
def link_pred(adjacency, models, edges, threshold):
    results = []
    probas = []
    y_true = is_edge(adjacency, edges)
    for model in tqdm(models):
        sim = model
        sim.fit(adjacency)
        scores = sim.predict(edges)
        proba = whitened_sigmoid(scores)
        y_pred = proba > threshold
        accuracy = get_accuracy_score(y_true, y_pred)
        results.append((str(model), accuracy))
        probas.append(proba)
    
    proba_array = np.array(probas).reshape(len(y_true),len(models))
    results_df = pd.DataFrame(results, columns=['models', 'accuracy'])
    probas_df = pd.DataFrame(proba_array, columns=[str(model) for model in models]) 
    return results_df, probas_df

def delete_edge(g, percent_edges: float):
    random.seed(rng)
    g_prime = g.copy()
    edge_list = g_prime.get_edgelist()
    edge_delete = random.sample(edge_list,k=round(len(edge_list)*percent_edges))
    g_prime.delete_edges(edge_delete)
    return g_prime, edge_delete

def linkpred_adamacar(g, percent_edge: float, mode='all'):
    g_prime, edge_delete = delete_edge(g, percent_edge)
    sim = np.array(g_prime.similarity_inverse_log_weighted(mode = mode))
    edge_score = []
    for idx, score in np.ndenumerate(sim):
            edge_score.append([idx, score])
    edge_score = np.array(edge_score, dtype=object)
    edge_predict = edge_score[(-edge_score[:, 1]).argsort()][:len(g.es.indices)]
    top_edge = edge_predict[:,0].tolist()
    matches = [edge for edge in top_edge if edge in edge_delete]
    print(f"Accuracy: {len(matches)/len(edge_delete)}")
    return g_prime, edge_delete, matches

def linkpred_jaccard(g, percent_edge: float, mode='all'):
    g_prime, edge_delete = delete_edge(g, percent_edge)
    sim = np.array(g_prime.similarity_jaccard(mode = mode))
    edge_score = []
    for idx, score in np.ndenumerate(sim):
            edge_score.append([idx, score])
    edge_score = np.array(edge_score, dtype=object)
    edge_predict = edge_score[(-edge_score[:, 1]).argsort()][:len(g.es.indices)]
    top_edge = edge_predict[:,0].tolist()
    matches = [edge for edge in top_edge if edge in edge_delete]
    print(f"Accuracy: {len(matches)/len(edge_delete)}")
    return g_prime, edge_delete, matches

def linkpred_dice(g, percent_edge: float, mode='all'):
    g_prime, edge_delete = delete_edge(g, percent_edge)
    sim = np.array(g_prime.similarity_dice(mode = mode))
    edge_score = []
    for idx, score in np.ndenumerate(sim):
            edge_score.append([idx, score])
    edge_score = np.array(edge_score, dtype=object)
    edge_predict = edge_score[(-edge_score[:, 1]).argsort()][:len(g.es.indices)]
    top_edge = edge_predict[:,0].tolist()
    matches = [edge for edge in top_edge if edge in edge_delete]
    print(f"Accuracy: {len(matches)/len(edge_delete)}")
    return g_prime, edge_delete, matches
#%%
g = ig.read('collabs_fin.pkl', format='pickle')
edge_list = g.get_edgelist()
#%%
edge_df = g.get_edge_dataframe() #get edge attributes dataframe for further checking
node_df = g.get_vertex_dataframe() 

#%%
g_prime, edge_delete, matches = linkpred_adamacar(g, 0.2, mode='all')
#%%
g_prime_j, edge_delete_j, matches_j = linkpred_jaccard(g, 0.2, mode='all')
#%%
g_prime_d, edge_delete_d, matches_d = linkpred_dice(g, 0.2, mode='all')
#%%
# graph = from_graphml('collabs.graphml', weight_key='weight')
# adjacency = graph.adjacency
# names = graph.names
# edges = edge_list
# y_true = is_edge(adjacency, edges)

# models = [AdamicAdar(),CommonNeighbors(),HubDepressedIndex(), HubPromotedIndex(), \
#     JaccardIndex(), PreferentialAttachment(), ResourceAllocation(), SaltonIndex(), SorensenIndex()]
# #%%

# #%%
# results, probas_df = link_pred(adjacency, models, edges, 0.75)
# results2, probas_df2 = link_pred(adjacency, models, edges, 0.45)
#%%

#layouts: "fruchterman_reingold", "kk", "dh", "auto"
def plot(g, layout='auto', filename='graph_plot.pdf'):
    categories = np.unique(np.array(g.vs['category'])).tolist()
    colors = sns.color_palette(palette="bright", n_colors=len(categories))
    vertex_type_dict = {k:v for k, v in zip(categories, colors)}
    visual_style = {}
    visual_style["vertex_size"] = 3
    visual_style["vertex_color"] = [vertex_type_dict[category] for category in g.vs["category"]]
    visual_style["vertex_frame_width"] = 0 
    visual_style["edge_width"] = [float(w)/max(g.es['weight']) for w in g.es['weight']]
    visual_style["edge_arrow_size"] = 0.2
    visual_style["edge_curved"] = False
    visual_style["edge_label"] = None
    visual_style["layout"] = layout
    visual_style["bbox"] = (400, 400)
    visual_style["margin"] = 20
    sns.set_theme(style = 'whitegrid', rc={'figure.dpi': 100})
    fig, ax = plt.subplots(figsize=(7,5))
    ig.plot(g, **visual_style, target=ax)
    plt.show()
    fig.savefig(filename)
        
#%%
plot(g,filename='g.pdf')