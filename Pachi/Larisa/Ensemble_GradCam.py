import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

train_datagen = ImageDataGenerator(rescale=1./255,
                                   zoom_range=0.2,
                                   rotation_range=15,
                                   horizontal_flip=True)


val_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory(data_path + '/Train',
                                                 target_size=(image_size, image_size),
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='categorical', shuffle=True)


Validation_set = val_datagen.flow_from_directory(data_path + '/Val',
                                                 target_size=(image_size, image_size),
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='categorical', shuffle=False)


def categorical_smooth_loss(y_true, y_pred, label_smoothing=0.1):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
    return loss


#  Let build our Ensemble network -- this is the pretrained model
pretrained_D = tf.keras.applications.DenseNet201(input_shape=(image_size, image_size, 3), weights='imagenet',
                                                 include_top=False)
pretrained_VGG = tf.keras.applications.VGG16(input_shape=(image_size, image_size, 3), weights='imagenet',
                                             include_top=False)
pretrained_googleNet = tf.keras.applications.InceptionV3(input_shape=(image_size, image_size, 3), weights='imagenet',
                                                         include_top=False)
# Combining the models
for layer in pretrained_D.layers:
    layer.trainable = False

for layer in pretrained_VGG.layers:
    layer.trainable = False

for layer in pretrained_googleNet.layers:
    layer.trainable = False

visible = tf.keras.layers.Input(shape=(image_size, image_size, 3))

x1 = pretrained_D(visible)
x3 = pretrained_VGG(visible)
x2 = pretrained_googleNet(visible)

x2 = tf.keras.layers.ZeroPadding2D(padding=((0, 2), (0, 2)))(x2)
merge = tf.keras.layers.concatenate([x1, x2, x3], name="concatallprobs")
x4 = Conv2D(filters=4480, kernel_size=1, padding="same", activation="relu", name="use_layer")(merge)
x5 = tf.keras.layers.Flatten(name="flatten_head")(x4)
x5 = tf.keras.layers.Dense(128, activation="relu", name="dense_head")(x5)
x5 = tf.keras.layers.Dropout(0.5, name="dropout_head")(x5)
x5 = tf.keras.layers.Dense(64, activation="relu", name="dense_head_2")(x5)
x5 = tf.keras.layers.Dropout(0.5, name="dropout_head_2")(x5)
model_out = tf.keras.layers.Dense(3, activation='softmax', name="predictions_head")(x5)

modelA = Model(inputs=visible, outputs=model_out)
modelA.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=categorical_smooth_loss, metrics=['accuracy'])

if __name__ == "__main__":
    history = modelA.fit(training_set, validation_data=Validation_set, epochs=100)  # 30

    # Saving the Model
    modelA.save("ensemble_GrandcamA.h5")
