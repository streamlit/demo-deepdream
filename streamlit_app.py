from io import BytesIO

import numpy as np
import PIL.Image
import streamlit as st
import tensorflow as tf

"# Deep Dream :sleeping:"


def download(url, max_dim=None):
    """Download an image from a URL and read it into a NumPy array."""
    name = url.split("/")[-1]
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)


def deprocess(img):
    """Normalize an image"""
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)


def img_to_bytes(img):
    """Convert a PIL image to a byte array so users can download it"""
    with BytesIO() as buf:
        result.save(buf, format="PNG")
        img_bytes = buf.getvalue()
    return img_bytes


def random_roll(img, maxroll):
    """Randomly shift the image to avoid tiled boundaries."""
    shift = tf.random.uniform(
        shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32
    )
    img_rolled = tf.roll(img, shift=shift, axis=[0, 1])
    return shift, img_rolled


@st.cache_resource
def load_base_model():
    return tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")


@st.cache_resource
def load_all_layers(_model):
    return {layer.name: layer.output for layer in _model.layers}


def load_dream_model(base_model, layers):
    return tf.keras.Model(inputs=base_model.input, outputs=layers)


def show(dg, img):
    """Display an image."""
    dg.image(PIL.Image.fromarray(np.array(img)), use_column_width=True)
    return dg


# Below is the actual logic for this app. It's a bit messy because it's almost a
# straight copy/paste from the original DeepDream example repo, which is also
# messy:
# https://github.com/tensorflow/docs/blob/9ae740ab7b5b3f9c32ca060332037b51d95674ae/site/en/tutorials/generative/deepdream.ipynb


def calc_loss(img, model):
    """Pass forward the image through the model to retrieve the activations."""
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return tf.reduce_sum(losses)


class TiledGradients(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[2], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
        )
    )
    def __call__(self, img, img_size, tile_size=512):
        shift, img_rolled = random_roll(img, tile_size)
        gradients = tf.zeros_like(img_rolled)
        xs = tf.range(0, img_size[1], tile_size)[:-1]
        if not tf.cast(len(xs), bool):
            xs = tf.constant([0])
        ys = tf.range(0, img_size[0], tile_size)[:-1]
        if not tf.cast(len(ys), bool):
            ys = tf.constant([0])

        for x in xs:
            for y in ys:
                with tf.GradientTape() as tape:
                    tape.watch(img_rolled)
                    img_tile = img_rolled[y : y + tile_size, x : x + tile_size]
                    loss = calc_loss(img_tile, self.model)
                gradients = gradients + tape.gradient(loss, img_rolled)

        gradients = tf.roll(gradients, shift=-shift, axis=[0, 1])
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        return gradients


def run_deep_dream_with_octaves(
    img, steps_per_octave=100, step_size=0.01, octaves=range(-2, 3), octave_scale=1.3
):
    base_shape = tf.shape(img)
    img = tf.keras.utils.img_to_array(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    initial_shape = img.shape[:-1]
    img = tf.image.resize(img, initial_shape)

    text_template = "Octave: %s\n\nStep: %s"
    progress_widget = st.sidebar.progress(0)
    p = 0.0

    for octave in octaves:
        new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32) * (
            octave_scale**octave
        )
        new_size = tf.cast(new_size, tf.int32)
        img = tf.image.resize(img, new_size)
        for step in range(steps_per_octave):
            gradients = get_tiled_gradients(img, new_size)
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)
            p += 1
            progress_widget.progress(
                p / (steps_per_octave * len(octaves)),
                text_template % (octave, step + 1),
            )
            if step % 10 == 0:
                show(image_widget, deprocess(img))
    result = PIL.Image.fromarray(np.array(deprocess(img)))
    return result


# Load and cache the base model and all layers
base_model = load_base_model()
all_layers = load_all_layers(base_model)

st.sidebar.caption(
    """This Streamlit app is based entirely on
[TensorFlow's DeepDream tutorial](https://github.com/tensorflow/docs/blob/9ae740ab7b5b3f9c32ca060332037b51d95674ae/site/en/tutorials/generative/deepdream.ipynb)."""
)

# Define a dictionary for image sources
image_sources = {
    "Specify image by URL...": None,
    "Upload image from my machine...": None,
    "Grace Hopper example": "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg",
    "Red sunflower example": "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg",
    "Yellow labrador example": "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg",
}

image_source = st.sidebar.selectbox(
    "Input image",
    list(image_sources.keys()),
    help="""
    Select an image to use as a starting point for Deep Dream.\n
    Either enter a URL to an image on the web, upload an image
    from your computer, or select one of the example images.""",
)

if image_source == "Specify image by URL...":
    url = st.sidebar.text_input(
        "Image URL",
        image_sources["Yellow labrador example"],
        help="The URL of the image to use as a starting point for Deep Dream.",
    )
    original_img = download(url, max_dim=500)

elif image_source == "Upload image from my machine...":
    uploaded_file = st.sidebar.file_uploader(
        "Upload your own image...",
        type=["jpg", "png"],
        help="Upload an image from your computer to use as a starting point for Deep Dream.\n\nUses the same image as the 'Yellow labrador example' if no image is uploaded.",
    )
    if uploaded_file is not None:
        original_img = PIL.Image.open(uploaded_file)
        original_img.thumbnail((500, 500))
        original_img = np.array(original_img.copy())
    else:
        original_img = download(image_sources["Yellow labrador example"], max_dim=500)

else:
    original_img = download(image_sources[image_source], max_dim=500)

octaves = st.sidebar.slider(
    "Octaves",
    min_value=-2,
    max_value=3,
    value=(-1, 0),
    step=1,
    help="The number of scales at which to run gradient ascent.",
)
steps_per_octave = st.sidebar.slider(
    "Steps per Octave",
    min_value=10,
    max_value=100,
    value=50,
    step=10,
    help="The number of gradient ascent steps to run at each octave.",
)
step_size = st.sidebar.slider(
    "Step Size",
    min_value=0.01,
    max_value=0.1,
    value=0.01,
    step=0.01,
    help="The step size for gradient ascent.",
)

names = st.sidebar.multiselect(
    "Select layers to visualize",
    list(all_layers.keys()),
    default=["mixed3", "mixed5"],
    help="""
    For DeepDream, the layers of interest are those where the convolutions are concatenated. 
    There are 11 of these layers in InceptionV3, named 'mixed0' though 'mixed10'. \n\n
    Using different layers will result in different dream-like images. Deeper layers respond 
    to higher-level features (such as eyes and faces), while earlier layers respond to simpler 
    features (such as edges, shapes, and textures).
    Source: [TensorFlow DeepDream](https://www.tensorflow.org/tutorials/generative/deepdream?hl=en#prepare_the_feature_extraction_model)""",
)

# Retrieve specific user-chosen layers from multiselect
layers = [all_layers[name] for name in names]
dream_model = load_dream_model(base_model, layers)
get_tiled_gradients = TiledGradients(dream_model)

st.subheader("Original Image")
st.image(original_img, use_column_width=True)

st.subheader("Deep Dream Image")
image_widget = st.empty()
result = run_deep_dream_with_octaves(
    img=original_img,
    steps_per_octave=steps_per_octave,
    step_size=step_size,
    octaves=octaves,
)

img_bytes = img_to_bytes(result)

st.download_button(
    label="Download Deep Dream Image",
    data=img_bytes,
    file_name="deep_dream.png",
    mime="image/png",
)
