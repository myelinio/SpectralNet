import sys

import numpy as np

from core.util import get_y_preds, get_y_preds_from_cm


class ClusteringAlgorithm:
    def __init__(self, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args={}):
        """
        Using either a newly instantiated ClusterClass or a provided
        cluster_obj, generates cluster assignments based on input data

        cluster_obj:    a pre-fitted instance of a clustering class
        ClusterClass:   a reference to the sklearn clustering class, necessary
                        if instantiating a new clustering class
        n_clusters:     number of clusters in the dataset, necessary
                        if instantiating new clustering class
        init_args:      any initialization arguments passed to ClusterClass

        """
        self.cluster_obj = cluster_obj
        self.n_clusters = n_clusters
        self.ClusterClass = ClusterClass
        self.init_args = init_args
        # if provided_cluster_obj is None, we must have both ClusterClass and n_clusters
        assert not (self.cluster_obj is None and (ClusterClass is None or self.n_clusters is None))
        self.kmeans_to_true_cluster_labels = None

    def fit(self, x, y):
        """
        x:              the points with which to perform clustering
        """
        if self.cluster_obj is None:
            self.cluster_obj = self.ClusterClass(self.n_clusters, **self.init_args)
            for _ in range(10):
                try:
                    self.cluster_obj.fit(x)
                    break
                except:
                    print("Unexpected error:", sys.exc_info())
            else:
                return np.zeros((len(x),))

        cluster_assignments = self.cluster_obj.predict(x)
        _, confusion_matrix, kmeans_to_true_cluster_labels = get_y_preds(cluster_assignments, y, self.n_clusters)
        _, self.kmeans_to_true_cluster_labels = get_y_preds_from_cm(cluster_assignments, self.n_clusters, confusion_matrix)
        return self

    def predict_cluster_assignments(self, x_spectralnet):
        return self.cluster_obj.predict(x_spectralnet)

    def predict(self, x_spectralnet):
        cluster_assignments = self.predict_cluster_assignments(x_spectralnet)
        return self.kmeans_to_true_cluster_labels[cluster_assignments]

