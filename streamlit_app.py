# coding: utf-8

import streamlit as st
import os
import numpy as np
import PIL.Image

import tensorflow as tf


'# Deeeeeeep dreeeeeeeeeam ~_~'

model_fn = 'models/tensorflow_inception_graph.pb'

# creating TensorFlow session and loading the model
graph = tf.Graph()
sess = tf.compat.v1.InteractiveSession(graph=graph)

with tf.compat.v1.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

t_input = tf.compat.v1.placeholder(np.float32, name='input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})


def get_tensor(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name('%s:0' % layer)


# start with a gray image with a little noise
img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0


def write_image(dg, arr):
    arr = np.uint8(np.clip(arr/255.0, 0, 1)*255)
    dg.image(arr, use_column_width=True)
    return dg


def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.compat.v1.placeholder, argtypes))

    def wrap(f):
        out = f(*placeholders)

        def wrapper(*args, **kw):
            return out.eval(
                dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap


# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.compat.v1.image.resize_bilinear(img, size)[0, :, :, :]


resize = tffunc(np.float32, np.int32)(resize)


def calc_grad_tiled(img, t_grad, tile_size=512):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over
    multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz), sz):
        for x in range(0, max(w-sz//2, sz), sz):
            sub = img_shift[y:y+sz, x:x+sz]
            g = sess.run(t_grad, {t_input: sub})
            grad[y:y+sz, x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def do_deepdream(
        t_obj, img0=img_noise, iter_n=10, step=1.5, octave_n=4,
        octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]

    # split the image into a number of octaves
    img = img0
    octaves = []
    for i in range(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        hi = img-resize(lo, hw)
        img = lo
        octaves.append(hi)

    image_widget = st.empty()
    text_template = 'Octave: %s\nIteration: %s'
    text_widget = st.sidebar.text(text_template % (0, 0))
    progress_widget = st.sidebar.progress(0)
    p = 0.

    # generate details octave by octave
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2])+hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g*(step / (np.abs(g).mean()+1e-7))
            p += 1
            progress_widget.progress(p / (octave_n * iter_n))

            write_image(image_widget, img)
            text_widget.text(text_template % (octave, i))


layers = [
    op.name for op in graph.get_operations()
    if op.type == 'Conv2D' and 'import/' in op.name
    ]

# Picking some internal layer. Note that we use outputs before applying the
# ReLU nonlinearity to have non-zero gradients for features with negative
# initial activations.
layer_num = st.sidebar.slider('Layer to visualize', 0, len(layers) - 1, 35)
layer = layers[layer_num]
#'**Selected layer:** ', layer

channels = int(get_tensor(layer).get_shape()[-1])

channel = st.sidebar.slider('Channel to visualize', 0, channels - 1, 139)
#'**Selected channel:** ', channel

octaves = st.sidebar.slider('Octaves', 1, 30, 4)
#'**Selected number of octaves:**', octaves

iterations = st.sidebar.slider('Iterations per octave', 1, 30, 10)
#'**Selected number of iterations:**', iterations

filename = st.sidebar.file_uploader('Choose an image:', ('jpg', 'jpeg'))

max_img_width = 600
max_img_height = 400

if filename:
    img0 = PIL.Image.open(filename)
    img0.thumbnail((max_img_width, max_img_height), PIL.Image.ANTIALIAS)
    img0 = np.float32(img0)
else:
    img0 = img_noise

'## Original image'
write_image(st, img0)

'## Output'
out = do_deepdream(
    get_tensor(layer)[:, :, :, channel], img0, octave_n=octaves,
    iter_n=iterations)

