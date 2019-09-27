"""
spectralnet.py: contains run function for spectralnet
"""
import os

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

    # from time import time
    # import matplotlib.pyplot as plt
    # from sklearn import manifold
    # from applications.plot_embedding import plot_embedding
    # from core.data import get_common_data, load_base_data
    # y_train, x_train, p_train, \
    # y_test, x_test, \
    # y_val, x_val, p_val, \
    # y_train_labeled, x_train_labeled, \
    # y_val_labeled, x_val_labeled, \
    # y_train_unlabeled, x_train_unlabeled, \
    # y_val_unlabeled, x_val_unlabeled, \
    # train_val_split = get_common_data(params, load_base_data(params, params['dset']))
    #
    # sample_size = 1000
    # x_test = x_val[:sample_size, :]
    # y_test = y_val[:sample_size]
    # x_affinity = siamese_net.predict(x_test, batch_sizes)
    #
    # # ----------------------------------------------------------------------
    # # t-SNE embedding of the digits dataset
    # print("Computing t-SNE embedding")
    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # t0 = time()
    # X_tsne = tsne.fit_transform(x_test)
    # X_affinity_tsne = tsne.fit_transform(x_affinity)
    #
    # plot_embedding(X_tsne,
    #                y_test,
    #                "t-SNE embedding of the digits - original (time %.2fs)" %
    #                (time() - t0))
    # plot_embedding(X_affinity_tsne,
    #                y_test,
    #                "t-SNE embedding of the digits - siamese (time %.2fs)" %
    #                (time() - t0))
    # plt.show()