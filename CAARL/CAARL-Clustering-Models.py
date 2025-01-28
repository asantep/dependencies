import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


class LoadSamples:
    def __init__(self, path_to_data):
        self.samples = self._load_data(path_to_data)

    def _load_data(self, path_to_data):
        with open(path_to_data, 'rb') as file:
            dico = pickle.load(file)
        return dico

class ClusteringProcess:
    def __init__(self, models, co_evolving_series, threshold=0.5, regularization_param=0.1):
        self.models = models
        self.co_evolving_series = co_evolving_series
        self.threshold = threshold
        self.regularization_param = regularization_param

    def cluster_models(self, X):
        # Generate the sub-time-series using the models
        sub_time_series = []
        for i, model in enumerate(self.models):
            with torch.no_grad():
                sub_series = model(X[i])
                sub_time_series.append(sub_series)
        sub_time_series = torch.stack(sub_time_series).numpy()

        # Compute pairwise distances based on MSE between models
        pairwise_distances = pdist(sub_time_series, metric=self._mse_distance)

        # Perform hierarchical clustering
        linkage_matrix = linkage(pairwise_distances, method='average')

        # Determine clusters based on a distance threshold
        clusters = fcluster(linkage_matrix, t=self.threshold, criterion='distance')

        # Refine clustering based on the loss function
        refined_clusters = self._refine_clusters(sub_time_series, clusters)

        return refined_clusters

    def _mse_distance(self, series_a, series_b):
        # Compute the mean-square-error between two time series
        mse_loss = nn.MSELoss()(torch.tensor(series_a), torch.tensor(series_b))
        return mse_loss.item()

    def _refine_clusters(self, sub_time_series, clusters):
        refined_clusters = clusters.copy()
        unique_clusters = np.unique(clusters)
        for cluster_id in unique_clusters:
            indices = np.where(clusters == cluster_id)[0]
            if len(indices) < 2:
                continue  # Skip refinement for single-model clusters

            centroid = torch.mean(torch.tensor(sub_time_series[indices]), dim=0)
            for i in indices:
                current_series = torch.tensor(sub_time_series[i])
                mse_loss = nn.MSELoss()(current_series, centroid).item()
                variance = torch.var(torch.tensor(sub_time_series[indices]), dim=0).mean().item()
                loss = mse_loss + self.regularization_param * variance
                if loss > self.threshold:
                    refined_clusters[i] = max(refined_clusters) + 1
        return refined_clusters

def run(path_to_data):

    # Loading data
    co_evolving_series, models = LoadSamples(path_to_data).samples

    # Perform clustering without specifying the number of clusters
    clustering_process = ClusteringProcess(models, co_evolving_series, threshold=0.5, regularization_param=0.1)
    clusters = clustering_process.cluster_models(co_evolving_series)

    print("Cluster assignments:", clusters)


if __name__ == '__main__':
    path_to_data = '/media/etienne/VERBATIM/Causal-Inference-Graph-Modeling-in-CoEvolving-Time-Sequences/Results/Sampling/ETTh1/2177_120.pkl'
    run(
        path_to_data=path_to_data
    )



