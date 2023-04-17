#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 21:02:36 2023

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
from sknetwork.utils import get_weights
from link_pred_feature_engineer import sum_stats
import random
from stellargraph import StellarDiGraph, StellarGraph
import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, HinSAGE, link_classification
from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection
import wandb
from wandb.keras import WandbMetricsLogger
from stellargraph import globalvar

rng = 2023
random.seed(rng)
#%%
def node_attributes(graph):
    nodes = pd.DataFrame(columns=['degree','closeness','page_rank',
                              'eccentricity'])
    for v in graph.vs.indices:
        nodes.loc[v] = [graph.vs[v].degree(),
                        graph.vs[v].closeness(),
                        graph.personalized_pagerank(v),
                        graph.vs[v].eccentricity()]
    return nodes

def sum_stats(g):
    summary = g.summary()
    directed = g.is_directed()
    weighted = g.is_weighted()
    connected = g.is_connected(mode='weak')
    simple = g.is_simple()
    ave_degree = mean(g.degree())
    max_degree = g.maxdegree()
    density = g.density()
    ave_cluster = g.transitivity_avglocal_undirected()
    print(f"Summary:{summary}\nDirected: {directed}\nWeighted: {weighted}\n"
        f"Connected: {connected}\nSimple: {simple}\nAverage Degree: {ave_degree}\nMax Degree: {max_degree}\n"
        f"Density: {density}\nAverage Clustering Coefficient: {ave_cluster}")
    
def add_node_attribs(g, nodes_df):
    attribs = nodes_df.columns[1:].to_list()
    for index in g.vs.indices:
        filter = nodes_df['id'] == g.vs[index]['name']
        attrib_values =  nodes_df[filter].values[0, 1:].tolist()
        for attrib, value in zip(attribs, attrib_values):
            g.vs[index][attrib] = value
#%%
g = ig.read('collabs_fin.pkl', format='pickle')
edge_list = g.get_edgelist()
#%%
# components = g_orig.connected_components(mode="weak")
# n_components = len(components)
# print(f"Number of connected components: {n_components}")
#%%
# Giant component
# g = components.giant()

#%%
comm = g.community_infomap(g.es['weight'])
#%%
clusters = comm.subgraphs()
#%%
big_cluster = []
small_cluster = []
for i, cluster in enumerate(clusters):
    if clusters[i].vcount() >= 20:
        big_cluster.append(cluster)
    else:
        small_cluster.append(cluster)
#%%
big_names1 = big_cluster[0].vs['name']
big_names2 = big_cluster[1].vs['name']
#%%
for index in g.vs.indices:
    if g.vs[index]['name'] in big_names1:
        g.vs[index]["cluster"] = "cluster1"
    elif g.vs[index]["name"] in big_names2:
        g.vs[index]["cluster"] = "cluster2"
    else:
        g.vs[index]["cluster"] = "none"
#%%
node_graph_attributes = node_attributes(g)
#%%
edge_df = g.get_edge_dataframe() #get edge attributes dataframe for further checking
node_df = g.get_vertex_dataframe() 
edge_df.rename(columns={"channel_from": "from", "channel_to": "to"}, inplace=True)
edge_df.drop(labels=["name", "from", "to", "from_category", "to_category"], axis =1, inplace=True)
# node_df_onehot = node_df.drop(labels=["topicIds", "name", 
#                                       'viewCount', 'subscriberCount', 
#                                       'videoCount', 'commentCount', "network",'popularity',
#                                       "cluster"], axis =1)
# node_df_onehot = pd.get_dummies(node_df_onehot, columns=["category"])
node_df_onehot = node_df.drop(labels=["topicIds", "name", 
                                      'viewCount', 'subscriberCount', 
                                      'videoCount', 'commentCount', "network",'popularity'], axis =1)
node_df_onehot = pd.get_dummies(node_df_onehot, columns=["category", "cluster"])

#%%
node_cluster = node_df.drop(labels=["topicIds", "name", 
                                       'viewCount', 'subscriberCount', "category",
                                       'videoCount', 'commentCount', "network",'popularity'], axis =1)
node_cluster = pd.get_dummies(node_cluster, columns=["cluster"])
#node_cluster.drop(labels=["cluster_none"], axis =1, inplace=True)
node_df.drop(labels=["topicIds", "name", "network", "category", "cluster"], axis =1, inplace=True)

#%%
scaler = preprocessing.StandardScaler()
#%%
node_df = pd.DataFrame(scaler.fit_transform(node_df), columns=node_df.columns)
#%%
node_df.reset_index(inplace=True)
node_graph_attributes.reset_index(inplace=True)
node_df_onehot.reset_index(inplace=True)
node_cluster.reset_index(inplace=True)
#%%
node_all = node_df.merge(node_df_onehot, left_on="index", right_on="vertex ID")
node_all.drop(columns='index', inplace=True)
#%%
node_combined = node_graph_attributes.merge(node_df, left_on="index", right_on="index")
node_super =  node_graph_attributes.merge(node_all, left_on="index", right_on="vertex ID")
node_super.drop(columns='index', inplace=True)
attribs_cluster = node_graph_attributes.merge(node_cluster, left_on="index", right_on="vertex ID")
attribs_cluster.drop(columns='index', inplace=True)
#%%
node_graph_attributes.set_index('index', inplace=True)
node_df.set_index('index', inplace=True)
node_combined.set_index('index', inplace=True)
node_all.set_index('vertex ID', inplace=True)
node_super.set_index('vertex ID', inplace=True)
attribs_cluster.set_index('vertex ID', inplace=True)
#%%
collab_graph = StellarGraph(
    {"corner": attribs_cluster}, {"line": edge_df})

# collab_graph = StellarDiGraph(
#     {"corner": node_graph_attributes}, {"line": edge_df}
# )
#print(collab_graph.info())
#%%
# Define an edge splitter on the original graph G:
edge_splitter_test = EdgeSplitter(collab_graph)

# Randomly sample a fraction p=0.2 of all positive links, and same number of negative links, from G, and obtain the
# reduced graph G_test with the sampled links removed:
G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
    p=0.2, method="global", keep_connected=True, seed=rng)

# Define an edge splitter on the reduced graph G_test:
edge_splitter_train = EdgeSplitter(G_test)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
# reduced graph G_train with the sampled links removed:
G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
    p=0.3, method="global", keep_connected=False, seed=rng)

# print(G_train.info())
# print(G_test.info())
#%%
BATCH_SIZE = 20
EPOCHS = 50
NUM_SAMPLES = [20, 10]
LEARNING_RATE = 1e-3
#%%
train_gen = GraphSAGELinkGenerator(G_train, BATCH_SIZE, NUM_SAMPLES, weighted=True)
train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)
test_gen = GraphSAGELinkGenerator(G_test, BATCH_SIZE, NUM_SAMPLES, weighted=True)
test_flow = test_gen.flow(edge_ids_test, edge_labels_test)
initializer = keras.initializers.GlorotUniform(seed=rng)
#%%
layer_sizes = [20, 20]
graphsage = GraphSAGE(
    layer_sizes=layer_sizes, generator=train_gen, bias=True, kernel_initializer=initializer)
#dropout=0.3
# Build the model and expose input and output sockets of graphsage model
# for link prediction
x_inp, x_out = graphsage.in_out_tensors()

prediction = link_classification(
    output_dim=1, output_act="relu", edge_embedding_method="ip")(x_out)
#%% MODEL 
model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=keras.losses.binary_crossentropy,
    metrics=["acc"])
#%%
init_train_metrics = model.evaluate(train_flow)
init_test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_test_metrics):
    print("\t{}: {:0.4f}".format(name, val))
#%%
wandb.login()
wandb.init(
    project="link_prediction",
    group="drop_first",
    config={
        "epochs": EPOCHS,
        "features": "attribs_cluster", #CHANGE THIS
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "num_samples": NUM_SAMPLES,
        "layer_size": layer_sizes
            })
#%% TRAINING LOOP
history = model.fit(train_flow, 
                    epochs=EPOCHS,
                    validation_data=test_flow, 
                    verbose=2,
                    callbacks = [WandbMetricsLogger()])
#%%
sg.utils.plot_history(history)
#%%
train_metrics = model.evaluate(train_flow)
test_metrics = model.evaluate(test_flow)
train_results = {}
print("\nTrain Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, train_metrics):
    train_results[f"train_{name}"] = val
    print("\t{}: {:0.4f}".format(name, val))

wandb.log({**train_results})
test_results = {}
print("\nTest Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, test_metrics):
    test_results[f"test_{name}"] = val
    print("\t{}: {:0.4f}".format(name, val))
wandb.log({**test_results})
wandb.finish()
#%%
#model.save('attribs_cluster_only.h5')

#%%
# import wandb
# api = wandb.Api()

# run = api.run("casanath/link_prediction/47l7hiva")
# run.config["features"] = "orig_channel_attribs_only"
# run.update()

