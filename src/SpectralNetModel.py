import argparse
import collections

import joblib

from applications.config import get_spectralnet_config
from core import networks
from core.data import build_spectral_data, embed_if_needed
from core.util import get_y_preds_from_cm
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import os
from keras.layers import Input
import numpy as np


Args = collections.namedtuple('Args', 'gpu gpu_memory_fraction dset')


def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=False)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


class SpectralNetModel(object):

    def __init__(self):
        args = Args(os.environ['GPU'], float(os.environ['GPU_MEMORY_FRACTION']), os.environ['DATA_SET'])
        global graph
        graph = tf.get_default_graph()

        self.params = get_spectralnet_config(args)

        # LOAD DATA
        data = build_spectral_data(self.params)

        ktf.set_session(get_session(args.gpu_memory_fraction))

        x_train, y_train, x_val, y_val, x_test, y_test = data['spectral']['train_and_test']

        x = x_train[:100]
        np.savetxt('spectralnet_input.txt', x)
        batch_sizes = {
            'Unlabeled': x.shape[0],
            'Labeled': x.shape[0],
            'Orthonorm': x.shape[0],
        }

        input_shape = x.shape[1:]

        y_labeled_onehot = np.empty((0, self.params['n_clusters']))

        # spectralnet has three inputs -- they are defined here
        inputs = {
            'Unlabeled': Input(shape=input_shape, name='UnlabeledInput'),
            'Labeled': Input(shape=input_shape, name='LabeledInput'),
            'Orthonorm': Input(shape=input_shape, name='OrthonormInput'),
        }

        # Load Siamese network
        if self.params['affinity'] == 'siamese':
            siamese_net = networks.SiameseNet(inputs, self.params['arch'], self.params.get('siam_reg'), None,
                                              self.params['siamese_model_path'])

        else:
            siamese_net = None

        # Load Spectral net
        y_true = tf.placeholder(tf.float32, shape=(None, self.params['n_clusters']), name='y_true')

        spectralnet_model_path = os.path.join(self.params['model_path'], 'spectral_net')
        self.spectral_net = networks.SpectralNet(inputs, self.params['arch'],
                                                 self.params.get('spec_reg'),
                                                 y_true, y_labeled_onehot,
                                                 self.params['n_clusters'], self.params['affinity'], self.params['scale_nbr'],
                                                 self.params['n_nbrs'], batch_sizes,
                                                 spectralnet_model_path,
                                                 siamese_net, train=False
                                                 )
        # get final embeddings
        self.km = joblib.load(os.path.join(self.params['model_path'], 'spectral_net', 'kmeans.sav'))

        self.confusion_matrix = joblib.load(os.path.join(self.params['model_path'], 'spectral_net', 'confusion_matrix.sav'))
        self.x_train = x_train

    def predict(self, X, feature_names):
        with graph.as_default():
            x_embedded = embed_if_needed([X], self.self.params)[0]
            x = np.concatenate([self.x_train, x_embedded], axis=0)
            x_spectralnet = self.spectral_net.predict_unlabelled(x)

            kmeans_assignments = self.km.predict(x_spectralnet)
            y_spectralnet = get_y_preds_from_cm(kmeans_assignments, self.self.params['n_clusters'], self.confusion_matrix)
            return y_spectralnet[-X.shape[0]:]

    def send_feedback(self, features, feature_names, reward, truth):
        pass


if __name__ == '__main__':
    os.environ['GPU'] = "0"
    os.environ['GPU_MEMORY_FRACTION'] = "0"
    os.environ['DATA_SET'] = "mnist"
    d = SpectralNetModel()
