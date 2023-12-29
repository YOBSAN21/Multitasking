import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, ZeroPadding2D, GlobalAveragePooling2D, Dense, Flatten, Dropout, Reshape, Multiply, BatchNormalization, Add
from PIL import ImageFile
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.signal import convolve2d


data_path = 'F:/willie/OPENSARSHIP_2'
# data_path = "D:/Breast_Code/Dataset_2/4_Classes_40"

image_size = 512
BATCH_SIZE = 1
num_classes = 6


def preprocessing_fun(image):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
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


def lee_filter(image):
    image = img_to_array(image)
    image = image[:, :, 0]  # assuming grayscale image
    kernel = np.ones((3, 3)) / 9
    mean = convolve2d(image, kernel, mode='same', boundary='symm')
    var = convolve2d(np.square(image - mean), kernel, mode='same', boundary='symm')
    factor = var / (var + 0.5)
    image = mean + factor * (image - mean)
    image = np.expand_dims(image, axis=-1)
    return image


train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
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


lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=0.005)

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', min_delta=0, patience=20, restore_best_weights=True)

counter = Counter(training_set.classes)

max_val = float(max(counter.values()))

class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}


def attention_block(input_tensor, compression_ratio=16):
    """
    Implementation of an attention block for convolutional neural networks
    """
    channels = int(input_tensor.shape[-1])
    x = GlobalAveragePooling2D()(input_tensor)
    x = Dense(channels//compression_ratio, activation='relu')(x)
    x = Dense(channels, activation='sigmoid')(x)
    x = Reshape((1,1,channels))(x)
    attention = Multiply()([input_tensor, x])
    return attention


def feature_fusion_block(tensor1, tensor2):
    """
    Implementation of a feature fusion block for convolutional neural networks
    """
    out_channels = int(tensor1.shape[-1])
    tensor2 = Conv2D(out_channels, kernel_size=(1,1))(tensor2)
    tensor2 = BatchNormalization()(tensor2)
    tensor1 = Add()([tensor1, tensor2])
    tensor1 = Activation('relu')(tensor1)
    return tensor1


# Load VGG16 pre-trained model
base_model = VGG16(weights='imagenet', include_top=False) #, input_shape=(224, 224, 3))

# Freeze layers
for layer in base_model.layers:
    layer.trainable = False

# Add attention and feature fusion blocks
input_tensor = Input(shape=(512, 512, 3))
x = base_model(input_tensor)

attention1 = attention_block(x)
conv1 = Conv2D(64, (3,3), activation='relu', padding='same')(attention1)
batch1 = BatchNormalization()(conv1)

attention2 = attention_block(batch1)
conv2 = Conv2D(128, (3,3), activation='relu', padding='same')(attention2)
pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

attention3 = attention_block(pool2)
conv3 = Conv2D(256, (3,3), activation='relu', padding='same')(attention3)
pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

attention4 = attention_block(pool3)
conv4 = Conv2D(512, (3,3), activation='relu', padding='same')(attention4)
conv4 = ZeroPadding2D(padding=((0,1),(0,1)))(conv4)
pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

attention5 = attention_block(pool4)
conv5 = Conv2D(512, (3,3), activation='relu', padding='same')(attention5)
conv5 = ZeroPadding2D(padding=((0,1),(0,1)))(conv5)
pool5 = MaxPooling2D(pool_size=(2,2))(conv5)

fusion = feature_fusion_block(pool5, conv5)

gap = GlobalAveragePooling2D()(fusion)
# flatten = Flatten()(fusion)

output_layer = Dense(64, kernel_regularizer=regularizers.l2(0.03))(gap)
output_layer = Dropout(0.2)(output_layer)
output = Dense(6, activation='softmax')(output_layer)

my_model = Model(inputs=input_tensor, outputs=output)

my_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000000001, clipvalue=0.2), loss='categorical_crossentropy',
                 metrics=['accuracy'])


if __name__ == "__main__":
    history = my_model.fit(training_set, validation_data=Validation_set, callbacks=[lr_reduce, es_callback], epochs=300,
                           class_weight=class_weights)

    # display_training_curves2(history.history['loss'], history.history['val_loss'], 'loss', 211)
    # display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
    # plt.show()

    # Saving the Model
    my_model.save("Wilfried.h5")
