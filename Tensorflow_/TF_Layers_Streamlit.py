import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt


st.title("Tensorflow Layers")

img = tf.io.decode_jpeg(tf.io.read_file('D:\digital_enhancement\images\kk\image_170.jpg'))
img = tf.image.resize(img,[72*2,128*2])
img = img[tf.newaxis,...]
img.shape


conv2d = tf.keras.layers.Conv2D(3, 31, activation='relu',input_shape=(None, None, 3))
out = conv2d(img)


st.write("Input Image Shape: ", img.shape)
st.write("Output Image Shape: ", out.shape)

st.pyplot(out[0].numpy(), clamp=True)

# st.image(out[0].numpy(), use_column_width=True)

# plt.imshow(out[0].numpy()*255.)