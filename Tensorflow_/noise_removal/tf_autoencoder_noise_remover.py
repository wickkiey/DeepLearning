# %%
import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist

import matplotlib.pyplot as plt

# %%
(xtrain,_), (xtest,_) = fashion_mnist.load_data()

# %%
xtrain.shape

# %%
# normalize the images

x_train = xtrain.astype('float32') / 255.
x_test = xtest.astype('float32') / 255.

# %%
# adding new dimension to the images

x_train = xtrain[...,tf.newaxis]

x_test = x_test[...,tf.newaxis]

# %%
print(x_train.shape, x_test.shape)

# %%
noise_factor = 0.4
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape) 
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

# %%
x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.) 
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

# %%
n = 5
plt.figure(figsize=(20, 8))
plt.gray()
for i in range(n):
  ax = plt.subplot(2, n, i + 1) 
  plt.title("original", size=20) 
  plt.imshow(tf.squeeze(x_test[i])) 
  plt.gray() 
  bx = plt.subplot(2, n, n+ i + 1) 
  plt.title("original + noise", size=20) 
  plt.imshow(tf.squeeze(x_test_noisy[i])) 
plt.show()

# %%
# Build encoder decoder model using keras functional API

from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose

class NoiseRemover(tf.keras.Model):
    def __init__(self):
        
        super(NoiseRemover,self).__init__()
        
        self.encoder = tf.keras.Sequential([
            Input(shape=(28,28,1)),
            Conv2D(16,3,activation='relu',padding='same',strides=2),
            Conv2D(8,3,activation='relu',padding='same',strides=2)
        ])
        
        self.decoder = tf.keras.Sequential([
            Conv2DTranspose(8,3,activation='relu',padding='same',strides=2),
            Conv2DTranspose(16,3,activation='relu',padding='same',strides=2),
            Conv2D(1,3,activation='sigmoid',padding='same')
            
        ])
        
    def call(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
            
        


# %%
auto_encoder = NoiseRemover()

# %%
auto_encoder.compile(optimizer='adam', loss='mse')

# %%
auto_encoder.fit(x_train_noisy, x_train, epochs=10, validation_data=(x_test_noisy, x_test))

# %%



