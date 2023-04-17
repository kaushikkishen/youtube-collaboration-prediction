#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:16:36 2023

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
#%%
g = ig.read('collabs_fin.pkl', format='pickle')
edge_list = g.get_edgelist()
#%%
node_graph_attributes = node_attributes(g)
#%%
edge_df = g.get_edge_dataframe() #get edge attributes dataframe for further checking
node_df = g.get_vertex_dataframe() 
edge_df.rename(columns={"channel_from": "from", "channel_to": "to"}, inplace=True)
edge_df.drop(labels=["name", "from", "to", "from_category", "to_category"], axis =1, inplace=True)
node_df_onehot = node_df.drop(labels=["topicIds", "name", 
                                      'viewCount', 'subscriberCount', 
                                      'videoCount', 'commentCount', "network",'popularity'], axis =1)
node_df_onehot = pd.get_dummies(node_df_onehot, columns=["category"])
node_df.drop(labels=["topicIds", "name", "network", "category"], axis =1, inplace=True)
#%%
scaler = preprocessing.StandardScaler()
#%%
node_df = pd.DataFrame(scaler.fit_transform(node_df), columns=node_df.columns)
#%%
node_df.reset_index(inplace=True)
node_graph_attributes.reset_index(inplace=True)
node_df_onehot.reset_index(inplace=True)
#%%
node_all = node_df.merge(node_df_onehot, left_on="index", right_on="vertex ID")
node_all.drop(columns='index', inplace=True)
#%%
node_combined = node_graph_attributes.merge(node_df, left_on="index", right_on="index")
node_super =  node_graph_attributes.merge(node_all, left_on="index", right_on="vertex ID")
node_super.drop(columns='index', inplace=True)
#%%
node_graph_attributes.set_index('index', inplace=True)
node_df.set_index('index', inplace=True)
node_combined.set_index('index', inplace=True)
node_all.set_index('vertex ID', inplace=True)
node_super.set_index('vertex ID', inplace=True)
#%% Create instance of undirected graph since following code has no implem for directed graph
collab_graph = StellarGraph(
    {"corner": node_super}, {"line": edge_df})

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
    config={
        "epochs": EPOCHS,
        "features": "node_df", #CHANGE THIS
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
# model.save('graphsage_super_final.h5')

# #%%
# import wandb
# api = wandb.Api()

# run = api.run("casanath/link_prediction/fuoze2d3")
# run.config["features"] = "node_super"
# run.update()

#%%

# model_file = 'graphsage_channel_attribs.h5'
# model_reload = keras.models.load_model(model_file, custom_objects=sg.custom_keras_layers)
# #%%
# train_metrics = model_reload.evaluate(train_flow)
# test_metrics = model_reload.evaluate(test_flow)
# train_results = {}
# print("\nTrain Set Metrics of the trained model:")
# for name, val in zip(model_reload.metrics_names, train_metrics):
#     train_results[f"train_{name}"] = val
# #%%
# wandb.init(
#     project="saved_linkpreds",
#     config={
#         "epochs": EPOCHS,
#         "features": "node_df", #CHANGE THIS
#         "model_file": model_file,
#         "batch_size": BATCH_SIZE,
#         "lr": LEARNING_RATE,
#         "optimizer": "Adam",
#         "loss": "CrossEntropyLoss",
#         "num_samples": NUM_SAMPLES,
#         "layer_size": layer_sizes
#             })
# #%%
# train_metrics = model_reload.evaluate(train_flow)
# test_metrics = model_reload.evaluate(test_flow)

# train_results = {}
# print("\nTrain Set Metrics of the trained model:")
# for name, val in zip(model_reload.metrics_names, train_metrics):
#     train_results[f"train_{name}"] = val
#     print("\t{}: {:0.4f}".format(name, val))
# #%%
# wandb.log({**train_results})
# test_results = {}
# print("\nTest Set Metrics of the trained model:")
# for name, val in zip(model_reload.metrics_names, test_metrics):
#     test_results[f"test_{name}"] = val
#     print("\t{}: {:0.4f}".format(name, val))
# wandb.log({**test_results})
# #%%
# wandb.finish()





















