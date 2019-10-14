import collections
import os

import joblib
import keras.backend.tensorflow_backend as ktf
import numpy as np
import tensorflow as tf
from keras.layers import Input
from keras import backend as K

from applications.config import get_spectralnet_config
from core import networks
from core.data import embed_if_needed, load_spectral_data
from core.util import print_accuracy, get_session

Args = collections.namedtuple('Args', 'gpu gpu_memory_fraction dset batch_size')


class SpectralNetModel(object):

    def __init__(self):
        K.set_learning_phase(0)
        args = Args(os.environ['GPU'], float(os.environ['GPU_MEMORY_FRACTION']), os.environ['DATA_SET'],
                    int(os.environ['BATCH_SIZE']))
        global graph
        graph = tf.get_default_graph()
        self.params = get_spectralnet_config(args)
        ktf.set_session(get_session(args.gpu_memory_fraction))

        self.batch_size = args.batch_size
        self.batch_sizes = {
            'Unlabeled': self.batch_size,
            'Labeled': self.batch_size,
            'Orthonorm': self.batch_size,
        }
        n_clusters = self.params['n_clusters']
        y_labeled_onehot = np.empty((0, n_clusters))

        # spectralnet has three inputs -- they are defined here
        input_shape = [n_clusters]
        inputs = {
            'Unlabeled': Input(shape=input_shape, name='UnlabeledInput'),
            'Labeled': Input(shape=input_shape, name='LabeledInput'),
            'Orthonorm': Input(shape=input_shape, name='OrthonormInput'),
        }

        # Load Spectral net
        y_true = tf.placeholder(tf.float32, shape=(None, n_clusters), name='y_true')

        spectralnet_model_path = os.path.join(self.params['model_path'], 'spectral_net')
        self.spectral_net = networks.SpectralNet(inputs, self.params['arch'],
                                                 self.params.get('spec_reg'),
                                                 y_true, y_labeled_onehot,
                                                 n_clusters, self.params['affinity'], self.params['scale_nbr'],
                                                 self.params['n_nbrs'], self.batch_sizes,
                                                 spectralnet_model_path,
                                                 siamese_net=None, train=False, x_train=None
                                                 )
        self.clustering_algo = joblib.load(os.path.join(self.params['model_path'], 'spectral_net', 'clustering_aglo.sav'))

    def predict(self, X, feature_names):
        with graph.as_default():
            data = [X]
            embed_if_needed(data, self.params)
            x_embedded = data[0]
            x_spectralnet = self.spectral_net.predict_unlabelled(x_embedded)
            y_pred = self.clustering_algo.predict(x_spectralnet)

            return y_pred[-X.shape[0]:]

    def send_feedback(self, features, feature_names, reward, truth):
        pass


if __name__ == '__main__':
    os.environ['GPU'] = "0"
    os.environ['GPU_MEMORY_FRACTION'] = "0"
    os.environ['DATA_SET'] = "mnist"
    os.environ['BATCH_SIZE'] = "100"
    d = SpectralNetModel()
