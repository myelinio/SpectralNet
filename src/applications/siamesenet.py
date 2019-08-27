"""
spectralnet.py: contains run function for spectralnet
"""
import os

import numpy as np
import tensorflow as tf
from keras.layers import Input

from core import networks

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def run_net(data, params):
    #
    # UNPACK DATA
    #
    pairs_train, dist_train, pairs_val, dist_val = data['train_and_test']

    #
    # SET UP INPUTS
    #

    # create true y placeholder (not used in unsupervised training)
    y_true = tf.placeholder(tf.float32, shape=(None, params['n_clusters']), name='y_true')

    batch_sizes = {
        'Unlabeled': params['batch_size'],
        'Labeled': params['batch_size'],
        'Orthonorm': params.get('batch_size_orthonorm', params['batch_size']),
    }

    input_shape = pairs_train[0].shape[1:]

    inputs = {
        'Unlabeled': Input(shape=input_shape, name='UnlabeledInput'),
        'Labeled': Input(shape=input_shape, name='LabeledInput'),
    }

    #
    # DEFINE AND TRAIN SIAMESE NET
    #

    siamese_model_path = params['model_path']

    siamese_net = networks.SiameseNet(inputs, params['arch'], params.get('siam_reg'), y_true, siamese_model_path)

    siamese_net.train(pairs_train, dist_train, pairs_val, dist_val,
                      params['siam_lr'], params['siam_drop'], params['siam_patience'],
                      params['siam_ne'], params['siam_batch_size'])
    if not os.path.isdir(siamese_model_path):
        os.makedirs(siamese_model_path)
    siamese_net.save_model()
