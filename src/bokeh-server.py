import argparse

import keras.backend.tensorflow_backend as ktf
import numpy as np
import tensorflow as tf
from bokeh.embed import server_document
from bokeh.layouts import column, gridplot
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure
from bokeh.sampledata.sea_surface_temperature import sea_surface_temperature
from bokeh.server.server import Server
from bokeh.themes import Theme
from flask import Flask, render_template
from keras import backend as K
from keras.layers import Input
from tornado.ioloop import IOLoop

from applications.config import get_spectralnet_config, get_siamese_config
from applications.plot_embedding import plot_embedding_bokeh
from core import networks
from core.data import load_spectral_data, load_siamese_data
# PARSE ARGUMENTS
from core.util import get_session
from sklearn import manifold
from core.data import get_common_data, load_base_data
from bokeh.layouts import column
from bokeh.models import Button, CustomJS
from bokeh.plotting import figure

# Example defined here: https://github.com/bokeh/bokeh/blob/1.3.4/examples/howto/server_embed/flask_embed.py
app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='gpu number to use', default='')
parser.add_argument('--gpu_memory_fraction', type=float, help='gpu percentage to use', default='0.8')
parser.add_argument('--dset', type=str, help='dataset to use', default='mnist')
args = parser.parse_args()
ktf.set_session(get_session(args.gpu_memory_fraction))
K.set_learning_phase(0)

params = get_spectralnet_config(args)
params['train_set_fraction'] = 0.8
data = load_spectral_data(params['data_path'], args.dset)

x_train, y_train, x_val, y_val, x_test, y_test = data['spectral']['train_and_test']
x_train_unlabeled, y_train_unlabeled, x_train_labeled, y_train_labeled = data['spectral'][
    'train_unlabeled_and_labeled']
x_val_unlabeled, y_val_unlabeled, x_val_labeled, y_val_labeled = data['spectral']['val_unlabeled_and_labeled']

batch_sizes = {
    'Unlabeled': x_train.shape[0],
    'Labeled': x_train.shape[0],
    'Orthonorm': x_train.shape[0],
}

input_shape = x_train.shape[1:]
inputs = {
    'Unlabeled': Input(shape=input_shape, name='UnlabeledInput'),
    'Labeled': Input(shape=input_shape, name='LabeledInput'),
    'Orthonorm': Input(shape=input_shape, name='OrthonormInput'),
}

y_true = tf.placeholder(tf.float32, shape=(None, params['n_clusters']), name='y_true')

# Load Siamese network
if params['affinity'] == 'siamese':
    siamese_input_shape = [params['n_clusters']]
    siamese_inputs = {
        'Unlabeled': Input(shape=siamese_input_shape, name='UnlabeledInput'),
        'Labeled': Input(shape=siamese_input_shape, name='LabeledInput'),
    }
    siamese_net = networks.SiameseNet(siamese_inputs, params['arch'], params.get('siam_reg'), y_true,
                                      params['siamese_model_path'])
else:
    siamese_net = None


y_train, x_train, p_train, \
y_test, x_test, \
y_val, x_val, p_val, \
y_train_labeled, x_train_labeled, \
y_val_labeled, x_val_labeled, \
y_train_unlabeled, x_train_unlabeled, \
y_val_unlabeled, x_val_unlabeled, \
train_val_split = get_common_data(params, load_base_data(params, params['dset']))


def modify_doc(doc):
    p1 = figure(x_axis_type="datetime", title="t-SNE embedding of the digits - original")
    p1.xaxis.axis_label = 'x1'
    p1.yaxis.axis_label = 'x2'
    p1.legend.location = "top_left"
    p2 = figure(x_axis_type="datetime", title="t-SNE embedding of the digits - siamese")
    p2.xaxis.axis_label = 'x1'
    p2.yaxis.axis_label = 'x2'
    p2.legend.location = "top_left"

    def callback(attr, old, sample_size):
        x_test = x_val[:sample_size, :]
        y_test = y_val[:sample_size]
        x_affinity = siamese_net.predict(x_test, batch_sizes)

        # ----------------------------------------------------------------------
        # t-SNE embedding of the digits dataset
        print("Computing t-SNE embeddings for sample size %s" % sample_size)
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(x_test)
        plot_embedding_bokeh(X_tsne, y_test, p1)

        X_affinity_tsne = tsne.fit_transform(x_affinity)
        plot_embedding_bokeh(X_affinity_tsne, y_test, p2)
        print("Finished Plotting for sample size %s" % sample_size)

    b = Button(label="Reset", button_type="success", width=300)
    b.js_on_click(CustomJS(args=dict(p1=p1, p2=p2), code="""
        console.log("start CustomJS");
        p1.reset.emit();
        p2.reset.emit();
        console.log("finish CustomJS");
    """))
    default_sample_size = 100
    slider = Slider(start=100, end=1000, value=default_sample_size, step=100, title="Sample Size")
    slider.on_change('value', callback)
    callback(None, None, default_sample_size)
    grid = gridplot([[p1, p2]], plot_width=400, plot_height=400)

    doc.add_root(column(slider, grid,b))

    doc.theme = Theme(filename="theme.yaml")


@app.route('/', methods=['GET'])
def bkapp_page():
    script = server_document('http://localhost:5006/bkapp')
    return render_template("embed.html", script=script, template="Flask")


def bk_worker():
    # Can't pass num_procs > 1 in this configuration. If you need to run multiple
    # processes, see e.g. flask_gunicorn_embed.py
    server = Server({'/bkapp': modify_doc}, io_loop=IOLoop(), allow_websocket_origin=["127.0.0.1:8000"])
    server.start()
    server.io_loop.start()


from threading import Thread

Thread(target=bk_worker).start()

if __name__ == '__main__':
    app.run(port=8000)
