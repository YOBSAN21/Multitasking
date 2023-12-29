import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda
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

image_size = 256
BATCH_SIZE = 4
STEPS_PER_EPOCH = int(Nos_Train // BATCH_SIZE)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
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


lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, epsilon=0.01, patience=10, verbose=1,
                                                 mode='min', cooldown=5, min_lr=0.01)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)

counter = Counter(training_set.classes)

max_val = float(max(counter.values()))

class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}


def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))


def five_Block(x):
    x = layers.Conv2D(512, kernel_size=1, strides=2, activation=gelu)(x)
    x = layers.MaxPooling2D(pool_size=1)(x)

    x = layers.Conv2D(1024, kernel_size=1, strides=2, activation=gelu)(x)
    return x


# # DesNet_Architecture
pretrained_densenet = tf.keras.applications.MobileNetV2(input_shape=(image_size, image_size, 3), weights='imagenet',
                                                        include_top=False)
pretrained_densenet_1 = tf.keras.applications.DenseNet201(input_shape=(image_size, image_size, 3), weights='imagenet',
                                                          include_top=False)

visible = tf.keras.layers.Input(shape=(image_size, image_size, 3))
FREEZE_LAYERS = -100
FREEZE_LAYERS_1 = -150

for layer in pretrained_densenet.layers[:FREEZE_LAYERS]:
    layer.trainable = False

for layer in pretrained_densenet.layers[FREEZE_LAYERS:]:
    layer.trainable = True

for layer in pretrained_densenet_1.layers[:FREEZE_LAYERS_1]:
    layer.trainable = False

for layer in pretrained_densenet_1.layers[FREEZE_LAYERS_1:]:
    layer.trainable = True

x1 = pretrained_densenet(visible)
x2 = pretrained_densenet_1(visible)

merge = tf.keras.layers.concatenate([x1, x2], name="concatallprobs")
merge = five_Block(merge)

# pt_depth = pretrained_densenet.output
# pt_features = pretrained_densenet(in_lay)
# #  here we do an attention mechanism to turn pixels in the GAP on and off
#
# bn_features = tf.keras.layers.BatchNormalization()(pt_features)
#
# attn_layer = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(bn_features)
# attn_layer = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
# attn_layer = Conv2D(1,
#                     kernel_size=(1, 1),
#                     padding='valid',
#                     activation='sigmoid')(attn_layer)
#
#
# # fan it out to all the channels
# up_c2_w = np.ones((1, 1, 1, pt_depth))
# up_c2 = Conv2D(pt_depth, kernel_size=(1, 1), padding='same',
#                activation='linear', use_bias=False, weights=[up_c2_w])
#
# up_c2.trainable = False
# attn_layer = up_c2(attn_layer)
#
#
# mask_features = multiply([attn_layer, bn_features])
# gap_features = GlobalAveragePooling2D()(mask_features)
# gap_mask = GlobalAveragePooling2D()(attn_layer)
#
#
# # to account for missing values from the attention model
# gap = Lambda(lambda x: x[0]/x[1], name='RescaleGAP')([gap_features, gap_mask])
# gap_dr = Dropout(0.5)(gap)
#
# dr_steps = Dropout(0.25)(Dense(128, activation='elu')(gap_dr))
# out_layer = Dense(3, activation='softmax', name="predictions_head")(dr_steps)
# print(out_layer.shape)
# exit()

# x2 = tf.keras.layers.AveragePooling2D(name="averagepooling2d_head")(merge)

x2 = tf.keras.layers.Flatten(name="flatten_head")(merge)

x2 = tf.keras.layers.Dense(32, activation=gelu, name="dense_head")(x2)

x2 = tf.keras.layers.Dropout(0.2, name="dropout_head")(x2)

model_out = tf.keras.layers.Dense(3, activation='softmax', name="predictions_head")(x2)


wilfried = Model(inputs=visible, outputs=model_out)
wilfried.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-05), loss='categorical_crossentropy', metrics=['accuracy'])

# loss='categorical_crossentropy, categorical_smooth_loss

if __name__ == "__main__":
    history = wilfried.fit(training_set, validation_data=Validation_set, callbacks=[lr_reduce, es_callback], epochs=300)

    # display_training_curves2(history.history['loss'], history.history['val_loss'], 'loss', 211)
    # display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
    # plt.show()
    #
    # #Saving the Model
    # my_model.save("model_D.h5")
