import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU, Dropout, Softmax, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
np.seterr(divide='ignore', invalid='ignore')

data_path = 'D:/Lari/Larisa/Data_2/'
data_path_ = 'D:/Lari/Larisa/Data_2/Train'

# s_1 = os.listdir(data_path_ + "/Bones")
# s_2 = os.listdir(data_path_ + "/Brain")
# s_3 = os.listdir(data_path_ + "/Normal")
s_1 = os.listdir(data_path_ + "/Normal")
s_2 = os.listdir(data_path_ + "/Stroke")

# Nos_Train = len(s_1) + len(s_2) + len(s_3)
Nos_Train = len(s_1) + len(s_2)

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


def categorical_smooth_loss(y_true, y_pred, label_smoothing=0.2):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
    return loss


# training call backs
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.4, patience=5, verbose=1,
                                                 min_delta=1e-2, min_lr=1e-4, mode='max')

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=1e-3, patience=5, mode='max',
                                               restore_best_weights=True, verbose=1)

counter = Counter(training_set.classes)

max_val = float(max(counter.values()))

class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}

# =====================================================================================
model = Sequential()

model.add(Conv2D(8, kernel_size=(5, 5), strides=3, activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=3))

model.add(Conv2D(16, kernel_size=(5, 5), strides=3, activation='relu'))
model.add(MaxPooling2D())
model.add(BatchNormalization(axis=1, center=True))

# model.add(Conv2D(32, kernel_size=(1, 1), strides=1, activation='relu'))
# model.add(MaxPooling2D())

model.add(Conv2D(32, kernel_size=(1, 1), strides=1, activation='relu'))
# model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# model.add(Dense(16, activation='relu'))

model.add(Dense(2, activation="softmax"))

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2, clipvalue=0.2), loss='binary_crossentropy',
              metrics=['accuracy'])

if __name__ == "__main__":
    history = model.fit(training_set, validation_data=Validation_set, callbacks=[lr_reduce, es_callback], epochs=300)

    # display_training_curves2(history.history['loss'], history.history['val_loss'], 'loss', 211)
    # display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
    # plt.show()
    #
    # #Saving the Model
    # model.save("model_D.h5")
