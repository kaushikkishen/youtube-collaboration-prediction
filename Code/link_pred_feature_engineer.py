#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:12:41 2023

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
#%% Helper functions
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
    
def disconnected_nodes_count(g):
    count = 0
    for node in g.vs:
        if any(g.are_connected(node.index, neighbor) and g.es[g.get_eid(node.index, neighbor)].is_loop() for neighbor in node.neighbors()):
            count += 1
    print(count) 

def add_node_attribs(g, nodes_df):
    attribs = nodes_df.columns[1:].to_list()
    for index in g.vs.indices:
        filter = nodes_df['id'] == g.vs[index]['name']
        attrib_values =  nodes_df[filter].values[0, 1:].tolist()
        for attrib, value in zip(attribs, attrib_values):
            g.vs[index][attrib] = value

def node_attributes(graph):
    nodes = pd.DataFrame(columns=['degree','closeness','clustering_coefficient','page_rank',
                              'eccentricity'])
    for v in graph.vs.indices:
        nodes.loc[v] = [graph.vs[v].degree(),
                        graph.vs[v].closeness(),
                        graph.transitivity_local_undirected(v),
                        graph.personalized_pagerank(v),
                        graph.vs[v].eccentricity()]
    
    return nodes
#%%
if __name__ == "__main__":
    collabs_df = pd.read_csv('./df_collabs.txt',
                             sep="\t", header=0, index_col=0) 
    nodes_df = pd.read_csv('./df_channel_statistics_first_day.txt',
                           sep="\t", header=0)
    
    # collabs_network = collabs_df[['from','to','weight', 'cluster', 'from_category', 'from_cluster', 'from_network',
    #     'from_popularity', 'from_topic',  'to_category', 'to_cluster',
    #     'to_network', 'to_popularity', 'to_topic', 'videos', ]].copy()
    
    collabs_df['channel_from'] = collabs_df['from']
    collabs_df['channel_to'] = collabs_df['to']
    collabs_df['name'] = collabs_df.apply(lambda x: x.name, axis=1)
    collabs_network = collabs_df[['from','to','weight', 'name', 'channel_from','channel_to', 
                                  'from_category', 'to_category']].copy()
#%%
if __name__ == "__main__":
    collabs = ig.Graph.DataFrame(collabs_network, directed=True, use_vids=False)
    
    
    #collabs_orig = collabs.get_edge_dataframe()
    add_node_attribs(collabs, nodes_df) #add node attributes from nodes_df
    collabs_new = collabs.get_edge_dataframe() #get edge attributes dataframe for further checking
    collabs_nodes = collabs.get_vertex_dataframe() #get node attributtes dataframe for further checking
    
    #null checks
    null_edges = collabs_new[collabs_new.isna().any(axis=1)] 
    null_nodes = collabs_nodes[collabs_nodes.isna().any(axis=1)]
    
    #get node-level graph attributes
    nodes = node_attributes(collabs)


#%% Export graph
# if __name__ == "__main__":
#     collabs.write_pickle('collabs_fin.pkl')
#     collabs.write_graphml('collabs.graphml')
#     collabs_new.to_csv('collabs_edges.csv')
#     collabs_nodes.to_csv('collabs_nodes.csv')
#%% Visualizations
# categories = nodes_df['category'].unique().tolist()

# colors = sns.color_palette(palette="bright", n_colors=len(categories))
# vertex_type_dict = {k:v for k, v in zip(categories, colors)}
# visual_style = {}
# visual_style["vertex_size"] = 3
# visual_style["vertex_color"] = [vertex_type_dict[category] for category in collabs.vs["category"]]
# visual_style["vertex_frame_width"] = 0 
# visual_style["edge_width"] = [float(w)/max(collabs.es['weight']) for w in collabs.es['weight']]
# #visual_style["edge_width"] = 1
# visual_style["edge_arrow_size"] = 0.2
# visual_style["edge_curved"] = False
# visual_style["edge_label"] = None
# #visual_style["layout"] = "fruchterman_reingold"
# visual_style["layout"] = "dh"
# #visual_style["layout"] = "kk"
# visual_style["bbox"] = (400, 400)
# visual_style["margin"] = 20
# #ig.plot(collabs, "test_plot.pdf", **visual_style)
# sns.histplot(nodes['degree'], log_scale=True, bins=10)

#%%
#sim = similarity.adamic_adar(g)
# components = g.connected_components(mode='weak')
# sns.set_theme(style = 'ticks', rc={'figure.dpi': 100})
# fig, ax = plt.subplots(figsize=(7,5))
# ig.plot(
#     components,
#     target=ax,
#     palette=ig.RainbowPalette(),
#     vertex_frame_width = 0.01, 
#     vertex_size=2.5,
#     layout = "auto",
#     vertex_color=list(map(int, ig.rescale(components.membership, (0, 200), clamp=True))),
#     edge_width=[float(w)/max(g.es['weight']) for w in g.es['weight']]
# )
# plt.show()
# fig.savefig('test.pdf')
















   