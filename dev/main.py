"tensorflow 2.9.1"

from email import generator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, random, time, shutil, math, glob
from pip import main
import seaborn as sns 
# Libraries for TensorFlow
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, BatchNormalization,  Dropout, Concatenate
from tensorflow import keras

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

plt.rcParams['figure.figsize'] = [15, 5]
base_log_dir = "./logs/"
sns.set_style("darkgrid")

if not os.path.exists(base_log_dir):
    os.makedirs(base_log_dir)

print("Importing libraries completed.")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Define a callback funcion
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=30,
    verbose=1,
    mode='min',
    baseline=None,
    restore_best_weights=True
)

# Define Block
def MK_block(input_):
    out_1_3 = Conv2D(48,kernel_size=(3,3),activation="relu",padding="same")(input_)
    out_1_5 = Conv2D(24,kernel_size=(5,5),activation="relu",padding="same")(input_)
    out_1_7 = Conv2D(12,kernel_size=(7,7),activation="relu",padding="same")(input_)

    out_1_3_5 = Concatenate()([input_,out_1_3,out_1_5])
    out_1_5_7 = Concatenate()([input_,out_1_3,out_1_5,out_1_7])

    out_1_b_3 = BatchNormalization()(out_1_3)
    out_1_b_3_5 = BatchNormalization()(out_1_3_5)
    out_1_b_5_7 = BatchNormalization()(out_1_5_7)

    out_2_3 = Conv2D(48,kernel_size=(3,3),activation="relu",padding="same")(out_1_b_3)
    out_2_5_2 = Conv2D(24,kernel_size=(5,5),activation="relu",padding="same")(out_1_b_3)
    out_2_3 = Concatenate()([input_,out_2_3,out_2_5_2,out_1_b_3])

    out_2_5 = Conv2D(36,kernel_size=(5,5),activation="relu",padding="same")(out_1_b_3_5)
    out_2_7 = Conv2D(18,kernel_size=(7,7),activation="relu",padding="same")(out_1_b_5_7)

    out_3_5_7 = Concatenate()([input_,out_2_5,out_2_7])
    out_3_b_5_7 = BatchNormalization()(out_3_5_7)
    out_3_b_3 = BatchNormalization()(out_2_3)

    out_4_3 = Conv2D(72,kernel_size=(3,3),activation="relu",padding="same")(out_3_b_5_7)
    out_4_b_3 = BatchNormalization()(out_4_3)

    out  = Concatenate()([input_,out_3_b_3,out_1_b_3,out_1_b_3_5,out_1_b_5_7,out_3_b_5_7,out_4_b_3])
    out = Conv2D(24,kernel_size=(1,1), activation="relu",padding="same")(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D(pool_size=(2,2))(out)

    return out

# Define Overall Model
def build_model():
    input_ = Input(shape=(28,28,1))
    input_ = tf.keras.layers.Rescaling(1./255)(input_)

    out_d = MK_block(input_)
    out_d = MK_block(out_d)
    out_d = MK_block(out_d)

    flat = Flatten()(out_d)
    dr = 0.3

    out_d = Dense(256,activation="relu")(flat)
    out_d = Dropout(dr)(out_d)
    out_d = Dense(128,activation="relu")(out_d)
    out_d = Dropout(dr)(out_d)
    out_d = Dense(128,activation="relu")(out_d)
    out_d = Dropout(dr)(out_d)

    out_d = Dense(64,activation="relu")(out_d)
    out_d = Dropout(dr)(out_d)

    out_d = Dense(10,activation="softmax")(out_d)

    model = keras.Model(inputs=input_,outputs=out_d)

    keras.utils.plot_model(model,expand_nested=True,show_shapes=True, to_file="model.png",show_dtype=True,show_layer_activations=True)
    plt.show()

    print("Model built successfully.")
    return model

def train_model(model,x_train, y_train,x_test, y_test):
    # Compile and train
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    History = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=128,epochs=1000,verbose=1, callbacks=[callback])

    # Export results as CSV for ploting
    hist_df = pd.DataFrame(History.history) 
    hist_csv_file = 'history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


def load_data(option):
    if option == 'fashion':
        # Load Fashion-MNIST dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
         # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")
        print(x_test.shape[0], "test samples")
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

    elif option == 'digit':
        # Load MNIST-digit
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
         # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")
        print(x_test.shape[0], "test samples")
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

    elif option == 'oracle':
        """
        Download oracle-mnist dataset
        run this commands:
            git clone https://github.com/wm-bupt/oracle-mnist.git
            and then load line 150 and 151
        """
        # Load Oracle-MNIST
        sys.path.insert(1, './oracle-mnist/src')
        import mnist_reader

        x_train, y_train = mnist_reader.load_data('oracle-mnist/data/oracle', kind='train')
        x_test, y_test = mnist_reader.load_data('oracle-mnist/data/oracle', kind='t10k')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
    else:
        pass
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    # select dataset for your training
    """
    Options : 
        1) digit
        2) fashion
        3) oracle
    """
    inp = 'digit'
    x_train, y_train, x_test, y_test = load_data(inp)

    # Pre Processing
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    model = build_model()
    train_model(model,  x_train, y_train, x_test, y_test)

    print('Training : done!')