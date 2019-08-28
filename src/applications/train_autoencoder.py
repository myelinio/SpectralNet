"""
"""

import argparse
from keras import backend as K
# import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf

from applications.config import get_siamese_config
# PARSE ARGUMENTS
from core.data import load_spectral_data
from core.networks import AutoEncoder

# add directories in src/ to path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='gpu number to use', default='')
parser.add_argument('--gpu_memory_fraction', type=float, help='gpu percentage to use', default='1.0')
parser.add_argument('--dset', type=str, help='datasett to use', default='mnist')
args = parser.parse_args()

params = get_siamese_config(args)

data = load_spectral_data(params['data_path'], args.dset)


def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=False)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def get_reconstruction_mse(x):
    x_embedded = ae.predict_embedding(x)
    x_recon = ae.predict_reconstruction(x_embedded)
    return np.mean(np.square(x - x_recon))


K.set_session(get_session(args.gpu_memory_fraction))

# RUN Train
x_train = data['spectral']['train_and_test'][0]
x_test = data['spectral']['train_and_test'][2]
json_path = '../pretrain_weights/ae_{}.json'.format(args.dset)
weights_path = '../pretrain_weights/ae_{}_weights.h5'.format(args.dset)

ae = AutoEncoder(x_train.shape[1], params['ae_arch'], params.get('ae_reg'), json_path, weights_path)
ae.train(x_train, x_test, epochs=200)
ae.save()

reconstruction_mse = get_reconstruction_mse(x_train)
print("train total reconstruction error:", np.mean(reconstruction_mse))
reconstruction_mse = get_reconstruction_mse(x_test)
print("test total reconstruction error:", np.mean(reconstruction_mse))
