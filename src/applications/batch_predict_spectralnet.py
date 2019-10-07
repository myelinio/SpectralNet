"""
"""

# add directories in src/ to path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import argparse
import os

import keras.backend.tensorflow_backend as ktf
import numpy as np
import tensorflow as tf
from keras.layers import Input
from sklearn.externals import joblib

from applications.config import get_spectralnet_config, get_siamese_config
from core import networks, costs
from core.data import build_siamese_data, load_spectral_data, decode_data, load_base_data, build_spectral_data, \
    load_data, load_siamese_data

# PARSE ARGUMENTS
from core.util import print_accuracy
from keras import backend as K

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='gpu number to use', default='')
parser.add_argument('--gpu_memory_fraction', type=float, help='gpu percentage to use', default='0.8')
parser.add_argument('--dset', type=str, help='dataset to use', default='mnist')
args = parser.parse_args()

params = get_spectralnet_config(args)
params['train_set_fraction'] = 0.8
data = load_spectral_data(params['data_path'], args.dset)


#
# params_noencode = params.copy()
# params_noencode['use_code_space'] = False
# # base_data = load_data(params_noencode)
# origin_data = build_spectral_data(params_noencode, base_data)


def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=False)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


ktf.set_session(get_session(args.gpu_memory_fraction))


x_train, y_train, x_val, y_val, x_test, y_test = data['spectral']['train_and_test']
x_train_unlabeled, y_train_unlabeled, x_train_labeled, y_train_labeled = data['spectral'][
    'train_unlabeled_and_labeled']
x_val_unlabeled, y_val_unlabeled, x_val_labeled, y_val_labeled = data['spectral']['val_unlabeled_and_labeled']

if 'siamese' in params['affinity']:
    siamese_data = load_siamese_data(get_siamese_config(args)['data_path'], args.dset)
    pairs_train, dist_train, pairs_val, dist_val = siamese_data['train_and_test']


# x = np.ones((10, 2))
# y = np.ones((10, 1))

# x = np.concatenate([x_train, np.ones((10, 2))], axis=0)
x = x_val[:100]
y = y_val[:100]
batch_sizes = {
    'Unlabeled': x.shape[0],
    'Labeled': x.shape[0],
    'Orthonorm': x.shape[0],
}


def run_predict(params):
    K.set_learning_phase(0)
    input_shape = x.shape[1:]

    y_labeled_onehot = np.empty((0, params['n_clusters']))

    # spectralnet has three inputs -- they are defined here
    inputs = {
        'Unlabeled': Input(shape=input_shape, name='UnlabeledInput'),
        'Labeled': Input(shape=input_shape, name='LabeledInput'),
        'Orthonorm': Input(shape=input_shape, name='OrthonormInput'),
    }

    # Load Siamese network
    if params['affinity'] == 'siamese':
        siamese_net = networks.SiameseNet(inputs, params['arch'], params.get('siam_reg'), None, params['siamese_model_path'])

    else:
        siamese_net = None

    # Load Spectral net
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
    # get final embeddings
    W_tensor = costs.knn_affinity(siamese_net.outputs['A'], params['n_nbrs'], scale=None, scale_nbr=params['scale_nbr'])

    x_spectralnet = spectral_net.predict_unlabelled(x)
    W = spectral_net.run_tensor(x, W_tensor)
    print('x_spectralnet', x_spectralnet.shape)
    clustering_algo = joblib.load(os.path.join(params['model_path'], 'spectral_net', 'clustering_aglo.sav'))

    kmeans_assignments = clustering_algo.predict_cluster_assignments(x_spectralnet)
    y_spectralnet = clustering_algo.predict(x_spectralnet)
    print_accuracy(kmeans_assignments, y, params['n_clusters'])
    # x_dec = decode_data(x, params, params['dset'])
    return x_spectralnet, y_spectralnet, x_spectralnet, W

# RUN EXPERIMENT
x_spectralnet, y_spectralnet, x, W = run_predict(params)
if args.dset in ['cc', 'cc_semisup']:
    # run plotting script
    import plot_2d
    plot_2d.process_plot(x_spectralnet, y_spectralnet, x, y=None, params=params, W=W)
