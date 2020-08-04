import PIL.Image
import numpy as np
import os
import requests
import streamlit as st
import subprocess
import tensorflow as tf
import zipfile


'# Deeeeeeep dreeeeeeeeeam ~_~'


# Basic setup

THIS_FILE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(THIS_FILE_DIR, 'models')
MODEL_FILENAME = os.path.join(MODEL_DIR, 'tensorflow_inception_graph.pb')


@st.cache
def download_model_from_web():
    if os.path.isfile(MODEL_FILENAME):
        return

    try:
        os.mkdir(MODEL_DIR)
    except FileExistsError:
        pass

    MODEL_ZIP_URL = (
        'https://storage.googleapis.com/download.tensorflow.org/models/'
        'inception5h.zip')
    ZIP_FILE_NAME = 'inception5h.zip'
    ZIP_FILE_PATH = os.path.join(MODEL_DIR, ZIP_FILE_NAME)
    resp = requests.get(MODEL_ZIP_URL, stream=True)

    with open(ZIP_FILE_PATH, 'wb') as file_desc:
        for chunk in resp.iter_content(chunk_size=5000000):
            file_desc.write(chunk)

    zip_file = zipfile.ZipFile(ZIP_FILE_PATH)
    zip_file.extractall(path=MODEL_DIR)

    os.remove(ZIP_FILE_PATH)


@st.cache(allow_output_mutation=True)
def init_model():
    with tf.compat.v1.gfile.FastGFile(MODEL_FILENAME, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


download_model_from_web()
graph_def = init_model()

graph = tf.Graph()
sess = tf.compat.v1.InteractiveSession(graph=graph)


# Below is the actual logic for this app. It's a bit messy because it's almost a
# straight copy/paste from the original DeepDream example repo, which is also
# messy:
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/deepdream

t_input = tf.compat.v1.placeholder(np.float32, name='input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})


def get_tensor(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name('%s:0' % layer)


# Start with a gray image with a little noise.
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
    multiple iterations.
    '''
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
    p = 0.0

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

# Sidebar controls:

# Temporary config option to remove deprecation warning.
st.set_option('deprecation.showfileUploaderEncoding', False)

file_obj = st.sidebar.file_uploader('Choose an image:', ('jpg', 'jpeg'))

# Picking some internal layer. Note that we use outputs before applying the
# ReLU nonlinearity to have non-zero gradients for features with negative
# initial activations.

max_value = len(layers) - 1
layer_num = st.sidebar.slider('Layer to visualize', 0, max_value, min(35, max_value))
layer = layers[layer_num]

channels = int(get_tensor(layer).get_shape()[-1])
max_value = channels - 1
channel = st.sidebar.slider('Channel to visualize', 0, max_value, min(139, max_value))

octaves = st.sidebar.slider('Octaves', 1, 30, 4)

iterations = st.sidebar.slider('Iterations per octave', 1, 30, 10)


# Show original image and final image, computing DeepDream on it iteratively.

max_img_width = 600
max_img_height = 400

if file_obj:
    img0 = PIL.Image.open(file_obj)
    img0.thumbnail((max_img_width, max_img_height), PIL.Image.ANTIALIAS)
    img0 = np.float32(img0)
else:
    img0 = img_noise


'## Original image'

if file_obj:
    write_image(st, img0)
else:
    st.info("No input image provided. Using random noise.")


'## Output'

out = do_deepdream(
    get_tensor(layer)[:, :, :, channel], img0, octave_n=octaves,
    iter_n=iterations)
