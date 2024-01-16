import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def reduce_population_agglomerative_clustering(data_to_cluster, num_clusters):
    labels = AgglomerativeClustering(n_clusters=num_clusters, linkage="average").fit(data_to_cluster).labels_
    reduced_population = []
    # print(type(data_to_cluster))
    # display(data_to_cluster)
    for i in range(max(labels) + 1):
        cluster = np.array(data_to_cluster)[labels == i]
        centroid = np.mean(cluster, axis=0)
        distances_from_centroid = np.linalg.norm(cluster - centroid, axis=1)
        closest_point = cluster[distances_from_centroid.argmin()].tolist()
        reduced_population.append(closest_point)
    # print(type(reduced_population))
    # print(reduced_population)
    reduced_population_df = pd.DataFrame(reduced_population, columns=data_to_cluster.columns)
    # display(reduced_population_df)
    return reduced_population_df
