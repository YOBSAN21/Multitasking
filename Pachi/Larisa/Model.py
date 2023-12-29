import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
np.seterr(divide='ignore', invalid='ignore')

data_path = 'D:/Lari/Larisa/Data_1/'
data_path_ = 'D:/Lari/Larisa/Data_1/Train'

s_1 = os.listdir(data_path_ + "/Bones")
s_2 = os.listdir(data_path_ + "/Brain")
s_3 = os.listdir(data_path_ + "/Normal")

Nos_Train = len(s_1) + len(s_2) + len(s_3)

image_size = 150
BATCH_SIZE = 4
STEPS_PER_EPOCH = int(Nos_Train // BATCH_SIZE)

# def preprocessing_fun(image):
#     WIDTH, HEIGHT, CHANNEL = image.shape
#     image = cv2.resize(image, (WIDTH, HEIGHT))
#     # Applying GaussianBlur.
#     blurred = cv2.blur(image, ksize=(int(WIDTH / 6), int(HEIGHT / 6)))
#     image = cv2.addWeighted(image, 4, blurred, 6, 0)
#     return image

train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True)
                                   # preprocessing_function=preprocessing_fun)


val_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory(data_path + '/Train',
                                                 target_size=(image_size, image_size),
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='categorical', shuffle=True)


Validation_set = val_datagen.flow_from_directory(data_path + '/Val',
                                                 target_size=(image_size, image_size),
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='categorical', shuffle=False)


def display_training_curves(training_accuracy, validation_accuracy, title, subplot):
    if subplot % 10 == 1:
        plt.subplots(figsize=(4, 6))
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.plot(training_accuracy)
    ax.plot(validation_accuracy)
    ax.set_title('Model ' + title)
    ax.set_ylabel(title)
    ax.set_xlabel('Epoch')
    ax.legend(['Train_Accuracy', 'Val_Accuracy'])


def display_training_curves2(training_loss, validation_loss, title, subplot):
    if subplot % 10 == 1:
        plt.subplots(figsize=(4, 6))
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.plot(training_loss)
    ax.plot(validation_loss)
    ax.set_title('Model ' + title)
    ax.set_ylabel(title)
    ax.set_xlabel('Epoch')
    ax.legend(['Train_Loss', 'Val_Loss'])


def categorical_smooth_loss(y_true, y_pred, label_smoothing=0.3):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
    return loss


# training call backs
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, epsilon=0.01, patience=5, verbose=1, mode='min')

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=10, restore_best_weights=True)

counter = Counter(training_set.classes)

max_val = float(max(counter.values()))

class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}


#  Build model -------------------------------------------------------------------------------------------------------
def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))


def activation_block(x):
    x = layers.Activation(gelu)(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x


def wilfried(image_size=150, filters=8, depth=4, kernel_size=5, patch_size=2, num_classes=3):

    inputs = keras.Input((image_size, image_size, 3))

    # Extract patch embeddings.
    x = conv_stem(inputs, filters, patch_size)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    print(x.shape)

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(units, activation=activation)(x)

    return keras.Model(inputs, outputs)


my_model = wilfried()
my_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.2), loss='categorical_crossentropy',
               metrics=['accuracy'])

if __name__ == "__main__":
    history = my_model.fit(training_set, validation_data=Validation_set, callbacks=[lr_reduce, es_callback], epochs=300)

    # display_training_curves2(history.history['loss'], history.history['val_loss'], 'loss', 211)
    # display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
    # plt.show()
    #
    # #Saving the Model
    my_model.save("model_D.h5")
