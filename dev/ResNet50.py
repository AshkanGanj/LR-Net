
import sys

# git clone https://github.com/wm-bupt/oracle-mnist.git

sys.path.insert(1, './oracle-mnist/src')
import mnist_reader

x_train, y_train = mnist_reader.load_data('oracle-mnist/data/oracle', kind='train')
x_test, y_test = mnist_reader.load_data('oracle-mnist/data/oracle', kind='t10k')

import tensorflow as tf 
import numpy as np 

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data() #just for fashion mnist



# expand new axis, channel axis 
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis = -1)

# [optional]: we may need 3 channel (instead of 1)
x_train = np.repeat(x_train, 3, axis=-1)
x_test = np.repeat(x_test, 3, axis=-1)
# it's always better to normalize 
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# resize the input shape , i.e. old shape: 28, new shape: 32
x_train = tf.image.resize(x_train, [32,32]) # if we want to resize 
x_test = tf.image.resize(x_test, [32,32]) # if we want to resize 

# one hot 
y_train = tf.keras.utils.to_categorical(y_train , num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

input = tf.keras.Input(shape=(32,32,3))
efnet = tf.keras.applications.ResNet50(weights='imagenet',
                                             include_top = False, 
                                             input_tensor = input)
# Now that we apply global max pooling.
gap = tf.keras.layers.GlobalMaxPooling2D()(efnet.output)

# Finally, we add a classification layer.
output = tf.keras.layers.Dense(10, activation='softmax', use_bias=True)(gap)

# bind all
func_model = tf.keras.Model(efnet.input, output)

func_model.compile(
          loss  = tf.keras.losses.CategoricalCrossentropy(),
          metrics = tf.keras.metrics.CategoricalAccuracy(),
          optimizer = tf.keras.optimizers.Adam())
# fit 
func_model.fit(x_train, y_train, batch_size=128, epochs=10, verbose = 2, validation_data=(x_test, y_test))

