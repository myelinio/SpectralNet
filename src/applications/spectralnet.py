"""
spectralnet.py: contains run function for spectralnet
"""
import os

import numpy as np
import tensorflow as tf
from keras.layers import Input
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.preprocessing import OneHotEncoder

from core import networks
from core.clustering import ClusteringAlgorithm
from core.data import concatenate
from core.util import print_accuracy


def run_net(data, params, train=True):

    # Unpack data
    x_train, y_train, x_val, y_val, x_test, y_test = data['spectral']['train_and_test']
    x_train_unlabeled, y_train_unlabeled, x_train_labeled, y_train_labeled = data['spectral'][
        'train_unlabeled_and_labeled']
    x_val_unlabeled, y_val_unlabeled, x_val_labeled, y_val_labeled = data['spectral']['val_unlabeled_and_labeled']

    x = concatenate([x_train, x_val, x_test])
    y = concatenate([y_train, y_val, y_test])

    if len(x_train_labeled):
        y_train_labeled_onehot = OneHotEncoder().fit_transform(y_train_labeled.reshape(-1, 1)).toarray()
    else:
        y_train_labeled_onehot = np.empty((0, len(np.unique(y))))

    # Set up inputs
    # create true y placeholder (not used in unsupervised training)
    y_true = tf.placeholder(tf.float32, shape=(None, params['n_clusters']), name='y_true')

    batch_sizes = {
        'Unlabeled': params['batch_size'],
        'Labeled': params['batch_size'],
        'Orthonorm': params.get('batch_size_orthonorm', params['batch_size']),
    }

    input_shape = x.shape[1:]

    # spectralnet has three inputs -- they are defined here
    inputs = {
        'Unlabeled': Input(shape=input_shape, name='UnlabeledInput'),
        'Labeled': Input(shape=input_shape, name='LabeledInput'),
        'Orthonorm': Input(shape=input_shape, name='OrthonormInput'),
    }

    # run only if we are using a siamese network
    if params['affinity'] == 'siamese':
        siamese_model_path = params['siamese_model_path']
        siamese_net = networks.SiameseNet(inputs, params['arch'], params.get('siam_reg'), y_true, siamese_model_path)
    else:
        siamese_net = None

    # Define and train spectral net
    spectralnet_model_path = os.path.join(params['model_path'], 'spectral_net')
    spectral_net = networks.SpectralNet(inputs, params['arch'],
                                        params.get('spec_reg'), y_true, y_train_labeled_onehot,
                                        params['n_clusters'], params['affinity'], params['scale_nbr'],
                                        params['n_nbrs'], batch_sizes,
                                        spectralnet_model_path,
                                        siamese_net, True, x_train, len(x_train_labeled),
                                        )
    if train:
        spectral_net.train(
            x_train_unlabeled, x_train_labeled, x_val_unlabeled,
            params['spec_lr'], params['spec_drop'], params['spec_patience'],
            params['spec_ne'])

        print("finished training")
        if not os.path.isdir(spectralnet_model_path):
            os.makedirs(spectralnet_model_path)
        spectral_net.save_model()

        print("finished saving model")

    # Evaluate model
    # get final embeddings
    x_spectralnet = spectral_net.predict(x)

    #########
    # kmeans_assignments, km = get_cluster_sols(x_spectralnet, ClusterClass=KMeans, n_clusters=params['n_clusters'],
    #                                           init_args={'n_init': 10})
    # joblib.dump(km, os.path.join(params['model_path'], 'spectral_net', 'kmeans.sav'))
    #
    # y_spectralnet, confusion_matrix = get_y_preds(kmeans_assignments, y, params['n_clusters'])
    #########

    clustering_algo = ClusteringAlgorithm(ClusterClass=KMeans, n_clusters=params['n_clusters'],
                                              init_args={'n_init': 10})
    clustering_algo.fit(x_spectralnet, y)

    # get accuracy and nmi
    joblib.dump(clustering_algo, os.path.join(params['model_path'], 'spectral_net', 'clustering_aglo.sav'))

    kmeans_assignments = clustering_algo.predict_cluster_assignments(x_spectralnet)
    y_spectralnet = clustering_algo.predict(x_spectralnet)
    print_accuracy(kmeans_assignments, y, params['n_clusters'])
    nmi_score = nmi(kmeans_assignments, y)
    print('NMI: ' + str(np.round(nmi_score, 3)))

    if params['generalization_metrics']:
        x_spectralnet_train = spectral_net.predict(x_train_unlabeled)
        x_spectralnet_test = spectral_net.predict(x_test)
        km_train = KMeans(n_clusters=params['n_clusters']).fit(x_spectralnet_train)
        from scipy.spatial.distance import cdist
        dist_mat = cdist(x_spectralnet_test, km_train.cluster_centers_)
        closest_cluster = np.argmin(dist_mat, axis=1)
        print_accuracy(closest_cluster, y_test, params['n_clusters'], ' generalization')
        nmi_score = nmi(closest_cluster, y_test)
        print('generalization NMI: ' + str(np.round(nmi_score, 3)))

    return x_spectralnet, y_spectralnet
