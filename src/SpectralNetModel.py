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

Args = collections.namedtuple('Args', 'gpu gpu_memory_fraction dset batch_size')


def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=False)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


class SpectralNetModel(object):

    def __init__(self):
        K.set_learning_phase(0)
        args = Args(os.environ['GPU'], float(os.environ['GPU_MEMORY_FRACTION']), os.environ['DATA_SET'],
                    int(os.environ['BATCH_SIZE']))
        global graph
        graph = tf.get_default_graph()

        self.params = get_spectralnet_config(args)

        # LOAD DATA
        data = load_spectral_data(self.params['data_path'], args.dset)

        ktf.set_session(get_session(args.gpu_memory_fraction))

        x_train, y_train, x_val, y_val, x_test, y_test = data['spectral']['train_and_test']

        x = x_train[:1000]
        y = y_train[:1000]
        batch_size = x.shape[0]
        self.batch_sizes = {
            'Unlabeled': batch_size,
            'Labeled': batch_size,
            'Orthonorm': batch_size,
        }

        input_shape = x.shape[1:]

        y_labeled_onehot = np.empty((0, self.params['n_clusters']))

        # spectralnet has three inputs -- they are defined here
        inputs = {
            'Unlabeled': Input(shape=input_shape, name='UnlabeledInput'),
            'Labeled': Input(shape=input_shape, name='LabeledInput'),
            'Orthonorm': Input(shape=input_shape, name='OrthonormInput'),
        }
        y_true = tf.placeholder(tf.float32, shape=(None, self.params['n_clusters']), name='y_true')

        # # Load Siamese network
        # if self.params['affinity'] == 'siamese':
        #     self.siamese_net = networks.SiameseNet(inputs, self.params['arch'], self.params.get('siam_reg'), y_true,
        #                                       self.params['siamese_model_path'])
        #
        # else:
        #     self.siamese_net = None

        # Load Spectral net
        y_true = tf.placeholder(tf.float32, shape=(None, self.params['n_clusters']), name='y_true')

        spectralnet_model_path = os.path.join(self.params['model_path'], 'spectral_net')
        self.spectral_net = networks.SpectralNet(inputs, self.params['arch'],
                                                 self.params.get('spec_reg'),
                                                 y_true, y_labeled_onehot,
                                                 self.params['n_clusters'], self.params['affinity'], self.params['scale_nbr'],
                                                 self.params['n_nbrs'], self.batch_sizes,
                                                 spectralnet_model_path,
                                                 siamese_net=None, train=False, x_train=None
                                                 )
        self.clustering_algo = joblib.load(os.path.join(self.params['model_path'], 'spectral_net', 'clustering_aglo.sav'))

        x_spectralnet = self.spectral_net.predict_unlabelled(x_train[:1])
        x_spectralnet = self.spectral_net.predict_unlabelled(x_train)
        # kmeans_assignments = self.clustering_algo.predict_cluster_assignments(x_spectralnet)
        # y_spectralnet = self.clustering_algo.predict(x_spectralnet)
        # print_accuracy(kmeans_assignments, y, self.params['n_clusters'])

    def predict(self, X, feature_names):
        with graph.as_default():
            data = [X]
            embed_if_needed(data, self.params)
            x_embedded = data[0]
            x = x_embedded
            x_spectralnet = self.spectral_net.predict(x)
            y_pred = self.clustering_algo.predict(x_spectralnet)

            return y_pred[-X.shape[0]:]

    def send_feedback(self, features, feature_names, reward, truth):
        pass


if __name__ == '__main__':
    os.environ['GPU'] = "0"
    os.environ['GPU_MEMORY_FRACTION'] = "0"
    os.environ['DATA_SET'] = "mnist"
    os.environ['BATCH_SIZE'] = "10"
    d = SpectralNetModel()
