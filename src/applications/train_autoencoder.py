"""
"""

import argparse

# import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from keras import backend as K

from applications.config import get_autoencoder_config
# PARSE ARGUMENTS
from core.data import load_spectral_data
from core.networks import AutoEncoder
import os

# add directories in src/ to path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='gpu number to use', default='')
parser.add_argument('--gpu_memory_fraction', type=float, help='gpu percentage to use', default='1.0')
parser.add_argument('--dset', type=str, help='dataset to use', default='mnist')
args = parser.parse_args()

params = get_autoencoder_config(args)

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
json_path = 'pretrain_weights/ae_{}.json'.format(args.dset)
weights_path = '{}/ae_{}_weights.h5'.format(params['model_path'], args.dset)
# weights_path = '../pretrain_weights/ae_{}_weights.h5'.format(args.dset)

if not os.path.exists(params['model_path']):
    os.makedirs(params['model_path'])

x_train = x_train.reshape(-1, np.prod(x_train.shape[1:]))
x_test = x_test.reshape(-1, np.prod(x_test.shape[1:]))

ae = AutoEncoder(x_train.shape[1], params['ae_arch'], params.get('ae_reg'), json_path, weights_path)

ae.train(x_train, x_test, epochs=1)

reconstruction_mse = get_reconstruction_mse(x_train)
print("train initial reconstruction error:", np.mean(reconstruction_mse))
reconstruction_mse = get_reconstruction_mse(x_test)
print("test initial reconstruction error:", np.mean(reconstruction_mse))

ae.train(x_train, x_test, epochs=params['spec_ae'])
ae.save()

reconstruction_mse = get_reconstruction_mse(x_train)
print("train final reconstruction error:", np.mean(reconstruction_mse))
reconstruction_mse = get_reconstruction_mse(x_test)
print("test final reconstruction error:", np.mean(reconstruction_mse))

# from modeldb.basic.Structs import Model, ModelConfig, ModelMetrics, Dataset
# from modeldb.basic.ModelDbSyncerBase import Syncer
#
# # Create a syncer using a convenience API
# syncer_obj = Syncer.create_syncer("Spectral Autoencoder %s" % args.dset,
#                                   "test_user",
#                                   "Autoencoder %s" % args.dset,
#                                   host="localhost")
# # create Datasets by specifying their filepaths and optional metadata
# # associate a tag (key) for each Dataset (value) and synch them
# datasets = {
#     "train": Dataset("/path/to/train", {"num_cols": 15, "dist": "random"}),
#     "test": Dataset("/path/to/test", {"num_cols": 15, "dist": "gaussian"})
# }
# syncer_obj.sync_datasets(datasets)
#
# # create the Model, ModelConfig, and ModelMetrics instances and synch them
# for i in range(10):
#     syncer_obj = Syncer.create_syncer("Spectral Autoencoder %s" % args.dset,
#                                       "test_user",
#                                       "Autoencoder %s" % args.dset,
#                                       host="localhost")
#
#     mdb_model1 = Model(model_type="model_obj", model="Spectral Autoencoder", path=params['model_path'], tag="autoencoder:spectralnet")
#     model_config1 = ModelConfig(model_type="model_obj", config={}, tag="autoencoder:spectralnet")
#     model_metrics1 = ModelMetrics({"rmse": 1 + np.random.uniform(0, 1)}, tag="autoencoder:spectralnet")
#     syncer_obj.sync_model("train", model_config1, mdb_model1)
#     syncer_obj.sync_metrics("test", mdb_model1, model_metrics1)
#     # actually write it
#     syncer_obj.sync()
