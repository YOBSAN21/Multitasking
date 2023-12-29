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

image_size = 224
BATCH_SIZE = 4
STEPS_PER_EPOCH = int(Nos_Train // BATCH_SIZE)


def preprocessing_fun(image):
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_1 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_2 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_3 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    # Flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    if p_spatial > .75:
        image = tf.image.transpose(image)

    # Rotates
    if p_rotate > .75:
        image = tf.image.rot90(image, k=3)  # rotate 270º
    elif p_rotate > .5:
        image = tf.image.rot90(image, k=2)  # rotate 180º
    elif p_rotate > .25:
        image = tf.image.rot90(image, k=1)  # rotate 90º

    # Pixel-level transforms
    if p_pixel_1 >= .4:
        image = tf.image.random_saturation(image, lower=.7, upper=1.3)
    if p_pixel_2 >= .4:
        image = tf.image.random_contrast(image, lower=.8, upper=1.2)
    if p_pixel_3 >= .4:
        image = tf.image.random_brightness(image, max_delta=.1)

    return image


train_datagen = ImageDataGenerator(rescale=1./255,
                                   samplewise_center=True,
                                   samplewise_std_normalization=True,
                                   preprocessing_function=preprocessing_fun)

val_datagen = ImageDataGenerator(rescale=1./255)


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


lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, epsilon=0.000001, patience=5, verbose=1, mode='min')
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=10, restore_best_weights=True)


# lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, verbose=1,
#                                                  min_delta=0.001, min_lr=0.001, mode='min')
#
# es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, mode='min',
#                                                restore_best_weights=True, verbose=1)

counter = Counter(training_set.classes)

max_val = float(max(counter.values()))

class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}


# DesNet_Architecture
pretrained_densenet = tf.keras.applications.resnet50.ResNet50(input_shape=(image_size, image_size, 3), weights='imagenet', include_top=False)

FREEZE_LAYERS = -50

for layer in pretrained_densenet.layers[:FREEZE_LAYERS]:
    layer.trainable = False

for layer in pretrained_densenet.layers[FREEZE_LAYERS:]:
    layer.trainable = True


x1 = pretrained_densenet.output
# print(x1.shape)

#  Build model -------------------------------------------------------------------------------------------------------
def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))


def activation_block(x):
    x = layers.Activation(gelu)(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)


def five_Block(x):
    x = layers.Conv2D(512, kernel_size=1, strides=2, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=1)(x)

    x = layers.Conv2D(1024, kernel_size=1, strides=2, activation='relu')(x)
    return x


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, kernel_size=1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # x = activation_block(x)

    return x


def wilfried(depth=5, filters=512, kernel_size=1, patch_size=6, num_classes=3):   # depth=4, image_size=224,

    inputs = keras.Input((image_size, image_size, 3))
    # inputs = x1

    # Extract patch embeddings.
    x = conv_stem(inputs, filters, patch_size)

    x = x1(x)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # x = five_Block(x)

    # Extract patch embeddings.
    # x = conv_stem(inputs, filters_s, patch_size)

    # x = five_Block(x)

    # # Pointwise convolution.
    # x = layers.Conv2D(filters, kernel_size, activation='relu')(x)
    # x = layers.BatchNormalization()(x)
    # # x = activation_block(x)
    print(x.shape)
    exit()

    # ConvMixer blocks.
    # for _ in range(depth):
    #     x = conv_mixer_block(x, filters, kernel_size)
    #
    # print(x.shape)

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)

    return keras.Model(pretrained_densenet.input, outputs)


my_model = wilfried()
my_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001, clipvalue=0.2), loss='categorical_crossentropy',
               metrics=['accuracy'])


# loss='categorical_crossentropy

if __name__ == "__main__":
    history = my_model.fit(training_set, validation_data=Validation_set, callbacks=[lr_reduce, es_callback], epochs=300)

    # display_training_curves2(history.history['loss'], history.history['val_loss'], 'loss', 211)
    # display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
    # plt.show()
    #
    # #Saving the Model
    # my_model.save("model_D.h5")
