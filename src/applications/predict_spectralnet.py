"""
Expected run times on a GTX 1080 GPU:
MNIST: 1 hr
Reuters: 2.5 hrs
cc: 15 min
"""

import sys, os
# add directories in src/ to path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import argparse
from collections import defaultdict

from core.data import get_data
from applications.spectralnet import run_net
import keras.backend.tensorflow_backend as ktf
import tensorflow as tf
import os

import numpy as np
import tensorflow as tf
from keras.layers import Input
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

from core import networks
from core.util import print_accuracy, get_cluster_sols, get_y_preds, get_y_preds_from_cm
from sklearn.externals import joblib

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='gpu number to use', default='')
parser.add_argument('--gpu_memory_fraction', type=float, help='gpu percentage to use', default='1.0')
parser.add_argument('--dset', type=str, help='datasett to use', default='mnist')
args = parser.parse_args()

model_path = os.environ.get('MODEL_PATH') or '/tmp/model/'
data_path = os.environ.get('DATA_PATH') or '/tmp/data/'

# SELECT GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

params = defaultdict(lambda: None)

# SET GENERAL HYPERPARAMETERS
general_params = {
        'dset': args.dset,                  # dataset: reuters / mnist
        'val_set_fraction': 0.1,            # fraction of training set to use as validation
        'precomputedKNNPath': '',           # path for precomputed nearest neighbors (with indices and saved as a pickle or h5py file)
        'siam_batch_size': 128,             # minibatch size for siamese net
        }
params.update(general_params)

# SET DATASET SPECIFIC HYPERPARAMETERS
if args.dset == 'mnist':
    mnist_params = {
        'n_clusters': 10,                   # number of clusters in data
        'use_code_space': True,             # enable / disable code space embedding
        'affinity': 'siamese',              # affinity type: siamese / knn
        'n_nbrs': 3,                        # number of nonzero entries (neighbors) to use for graph Laplacian affinity matrix
        'scale_nbr': 2,                     # neighbor used to determine scale of gaussian graph Laplacian; calculated by
                                            # taking median distance of the (scale_nbr)th neighbor, over a set of size batch_size
                                            # sampled from the datset

        'siam_k': 2,                        # threshold where, for all k <= siam_k closest neighbors to x_i, (x_i, k) is considered
                                            # a 'positive' pair by siamese net

        'siam_ne': 400,                     # number of training epochs for siamese net
        'spec_ne': 400,                     # number of training epochs for spectral net
        'siam_lr': 1e-3,                    # initial learning rate for siamese net
        'spec_lr': 1e-3,                    # initial learning rate for spectral net
        'siam_patience': 10,                # early stopping patience for siamese net
        'spec_patience': 20,                # early stopping patience for spectral net
        'siam_drop': 0.1,                   # learning rate scheduler decay for siamese net
        'spec_drop': 0.1,                   # learning rate scheduler decay for spectral net
        'batch_size': 1024,                 # batch size for spectral net
        'siam_reg': None,                   # regularization parameter for siamese net
        'spec_reg': None,                   # regularization parameter for spectral net
        'siam_n': None,                     # subset of the dataset used to construct training pairs for siamese net
        'siamese_tot_pairs': 600000,        # total number of pairs for siamese net
        'arch': [                           # network architecture. if different architectures are desired for siamese net and
                                            #   spectral net, 'siam_arch' and 'spec_arch' keys can be used
            {'type': 'relu', 'size': 1024},
            {'type': 'relu', 'size': 1024},
            {'type': 'relu', 'size': 512},
            {'type': 'relu', 'size': 10},
            ],
        'use_approx': False,                # enable / disable approximate nearest neighbors
        'use_all_data': True,               # enable to use all data for training (no test set)
        }
    params.update(mnist_params)
elif args.dset == 'reuters':
    reuters_params = {
        'n_clusters': 4,
        'use_code_space': True,
        'affinity': 'siamese',
        'n_nbrs': 30,
        'scale_nbr': 10,
        'siam_k': 100,
        'siam_ne': 20,
        'spec_ne': 300,
        'siam_lr': 1e-3,
        'spec_lr': 5e-5,
        'siam_patience': 1,
        'spec_patience': 5,
        'siam_drop': 0.1,
        'spec_drop': 0.1,
        'batch_size': 2048,
        'siam_reg': 1e-2,
        'spec_reg': 5e-1,
        'siam_n': None,
        'siamese_tot_pairs': 400000,
        'arch': [
            {'type': 'relu', 'size': 512},
            {'type': 'relu', 'size': 256},
            {'type': 'relu', 'size': 128},
            ],
        'use_approx': True,
        'use_all_data': True,
        'data_path': data_path,
        }
    params.update(reuters_params)
elif args.dset == 'cc':
    cc_params = {
        # data generation parameters
        'train_set_fraction': 1.,       # fraction of the dataset to use for training
        'noise_sig': 0.1,               # variance of the gaussian noise applied to x
        'n': 1500,                      # number of total points in dataset
        # training parameters
        'n_clusters': 2,
        'use_code_space': False,
        'affinity': 'siamese',
        # 'affinity': 'full',
        'n_nbrs': 2,
        'scale_nbr': 2,
        'spec_ne': 300,
        'spec_lr': 1e-3,
        'spec_patience': 30,
        'spec_drop': 0.1,
        'batch_size': 128,
        'batch_size_orthonorm': 128,
        'spec_reg': None,
        'arch': [
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            ],
        'use_all_data': True,
        }
    params.update(cc_params)
elif args.dset == 'cc_semisup':
    cc_semisup_params = {
        'dset': 'cc',                   # dataset affects data loading in get_data() so we must set back to 'cc'
        # data generation parameters
        'train_set_fraction': .8,
        'noise_sig': 0.175,
        'n': 1900,
        # training parameters
        'train_labeled_fraction': 0.02, # fraction of the training set to provide labels for (in semisupervised experiments)
        'n_clusters': 2,
        'use_code_space': False,
        'affinity': 'knn',
        'n_nbrs': 2,
        'scale_nbr': 2,
        'spec_ne': 10,
        'spec_lr': 1e-3,
        'spec_patience': 30,
        'spec_drop': 0.1,
        'batch_size': 128,
        'batch_size_orthonorm': 256,
        'spec_reg': None,
        'arch': [
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            {'type': 'softplus', 'size': 50},
            {'type': 'BatchNormalization'},
            ],
        'generalization_metrics': True, # enable to check out of set generalization error and nmi
        'use_all_data': False,
        }
    params.update(cc_semisup_params)

params['model_path'] = model_path
# LOAD DATA
data = get_data(params)


def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=False)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


ktf.set_session(get_session(args.gpu_memory_fraction))


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

x_train, y_train, x_val, y_val, x_test, y_test = data['spectral']['train_and_test']
x_train_unlabeled, y_train_unlabeled, x_train_labeled, y_train_labeled = data['spectral'][
    'train_unlabeled_and_labeled']
x_val_unlabeled, y_val_unlabeled, x_val_labeled, y_val_labeled = data['spectral']['val_unlabeled_and_labeled']

if 'siamese' in params['affinity']:
    pairs_train, dist_train, pairs_val, dist_val = data['siamese']['train_and_test']


# x = np.ones((10, 2))
# y = np.ones((10, 1))

x = np.concatenate([x_train, np.ones((10, 2))], axis=0)
np.savetxt('spectralnet_input.txt', x)
batch_sizes = {
    'Unlabeled': x.shape[0],
    'Labeled': x.shape[0],
    'Orthonorm': x.shape[0],
}
def run_predict(params):
    # ktf.set_learning_phase(0)
    input_shape = x.shape[1:]

    y_labeled_onehot = np.empty((0, params['n_clusters']))

    # spectralnet has three inputs -- they are defined here
    inputs = {
        'Unlabeled': Input(shape=input_shape, name='UnlabeledInput'),
        'Labeled': Input(shape=input_shape, name='LabeledInput'),
        'Orthonorm': Input(shape=input_shape, name='OrthonormInput'),
    }

    #
    # DEFINE AND TRAIN SIAMESE NET
    #
    # run only if we are using a siamese network
    if params['affinity'] == 'siamese':
        siamese_model_path = os.path.join(params['model_path'], 'spectral_net')

        siamese_net = networks.SiameseNet(inputs, params['arch'], params.get('siam_reg'), None, siamese_model_path)

        # history = siamese_net.train(pairs_train, dist_train, pairs_val, dist_val,
        #                             params['siam_lr'], params['siam_drop'], params['siam_patience'],
        #                             params['siam_ne'], params['siam_batch_size'])
        if not os.path.isdir(siamese_model_path):
            os.makedirs(siamese_model_path)
        siamese_net.save_model()


    else:
        siamese_net = None

    #
    # DEFINE AND TRAIN SPECTRALNET
    #
    y_true = tf.placeholder(tf.float32, shape=(None, params['n_clusters']), name='y_true')

    spectralnet_model_path = os.path.join(params['model_path'], 'spectral_net')
    spectral_net = networks.SpectralNet(inputs, params['arch'],
                                        params.get('spec_reg'),
                                        y_true, y_labeled_onehot,
                                        params['n_clusters'], params['affinity'], params['scale_nbr'],
                                        params['n_nbrs'], batch_sizes,
                                        spectralnet_model_path,
                                        siamese_net, train=False
                                        )
    # EVALUATE
    #

    # get final embeddings
    x_spectralnet = spectral_net.predict_unlabelled(x)
    print('x_spectralnet', x_spectralnet.shape)
    km = joblib.load(os.path.join(params['model_path'], 'spectral_net', 'kmeans.sav'))
    kmeans_assignments = km.predict(x_spectralnet)

    confusion_matrix = joblib.load(os.path.join(params['model_path'], 'spectral_net', 'confusion_matrix.sav'))

    y_spectralnet = get_y_preds_from_cm(kmeans_assignments, params['n_clusters'], confusion_matrix)

    return x_spectralnet, y_spectralnet

# RUN EXPERIMENT
x_spectralnet, y_spectralnet = run_predict(params)
if args.dset in ['cc', 'cc_semisup']:
    # run plotting script
    import plot_2d
    plot_2d.process(x_spectralnet, y_spectralnet, x, y=None, params=params)
