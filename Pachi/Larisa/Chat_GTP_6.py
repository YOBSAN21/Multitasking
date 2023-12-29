from collections import Counter
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow as tensorflow
from tensorflow.keras import regularizers
from tensorflow.keras import backend as k
from tensorflow.python.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, Dense, Lambda, multiply
from tensorflow.keras.layers import Reshape, Lambda, GlobalAveragePooling2D, UpSampling2D, Concatenate, Dropout

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

data_path = '/home/Pachi/FUSAR'

BATCH_SIZE = 2
num_classes = 7
input_shape = (256, 256, 3)
img_size = (256, 256)


def preprocessing_fun(image):
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    # p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_1 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_2 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_3 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    # Flips
    # image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)

    if p_spatial > .75:
        image = tf.image.transpose(image)

    # Rotates
    # if p_rotate > .75:
    #     image = tf.image.rot90(image, k=3)  # rotate 270º
    # elif p_rotate > .5:
    #     image = tf.image.rot90(image, k=2)  # rotate 180º
    # elif p_rotate > .25:
    #     image = tf.image.rot90(image, k=1)  # rotate 90º

    # Pixel-level transforms
    if p_pixel_1 >= .4:
        image = tf.image.random_saturation(image, lower=.5, upper=1.5)
    if p_pixel_2 >= .4:
        image = tf.image.random_contrast(image, lower=.5, upper=1.5)
    if p_pixel_3 >= .4:
        image = tf.image.random_brightness(image, max_delta=.1)

    return image


train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   preprocessing_function=preprocessing_fun)


val_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory(data_path + '/Train',
                                                 target_size=img_size,
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='categorical', shuffle=True)


Validation_set = val_datagen.flow_from_directory(data_path + '/Val',
                                                 target_size=img_size,
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


lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1.1121e-5)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', min_delta=0, patience=20, restore_best_weights=True)
counter = Counter(training_set.classes)
max_val = float(max(counter.values()))
class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}


# Model ========================================================================================================
input_layer = Input(shape=input_shape)

# Convert RGB to grayscale
gray_layer = Lambda(lambda x: k.mean(x, axis=3, keepdims=True))(input_layer)

# Label smoothing
alpha = 0.1
smoothed_labels = Lambda(lambda x: (1 - alpha) * x + alpha / num_classes)(gray_layer)

# Convolutional layers
conv1 = Conv2D(16, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(smoothed_labels)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(pool1)
conv3 = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(conv2)

# Spatial attention approach
Spatial_attention1 = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(conv3)
Spatial_attention1_1 = Lambda(lambda x: x[0] * x[1])([conv3, Spatial_attention1])

# channel-wise attention mechanism
Channel_attention1 = GlobalAveragePooling2D()(conv3)
Channel_attention1_1 = Reshape((1, 1, 64))(Channel_attention1)
Channel_attention1_2 = Conv2D(64, (1, 1), activation='sigmoid', padding='same')(Channel_attention1_1)
Channel_attention1_3 = multiply([conv3, Channel_attention1_2])

# concatenate the two attention mechanisms
Attn = Concatenate()([Spatial_attention1_1, Channel_attention1_3])
batch2 = BatchNormalization()(Attn)
    
conv4 = Conv2D(128, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(batch2)
conv5 = Conv2D(256, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(conv4)
Batch = BatchNormalization()(conv5)

# Global self-attention mechanism
gap = GlobalAveragePooling2D()(Batch)
gap_1 = Reshape((1, 1, 256))(gap)

att_conv = Conv2D(256, (1, 1), activation='sigmoid', padding='same')(Batch)
att_conv_1 = att_conv * gap_1

att_conv_2 = Conv2D(8, (1, 1), activation='relu', padding='same')(att_conv_1)
att_conv_3 = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(att_conv_2)

att_conv_4 = Batch * att_conv_3

# Perform  convolutions on an input with grey image ==================================================================
Conv_Hog_2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(batch2)
Batch_2 = BatchNormalization()(Conv_Hog_2)
Conv_Hog_3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(Batch_2)
Batch_3 = BatchNormalization()(Conv_Hog_3)
Conv_Hog_4 = Conv2D(64, kernel_size=(3, 3), activation='relu')(Batch_3)
Conv_Hog_5 = Conv2D(32, kernel_size=(3, 3), activation='relu')(Conv_Hog_4)

# Step 2: Apply histogram of oriented gradient (HOG) to the output of step 1
hog_layer = tf.image.extract_patches(
    images=Conv_Hog_5,
    sizes=[1, 16, 16, 1],
    strides=[1, 8, 8, 1],
    rates=[1, 1, 1, 1],
    padding='VALID'
)

hog_layer_1 = UpSampling2D(size=(9, 9))(hog_layer)
hog_layer_2 = ZeroPadding2D((1, 1))(hog_layer_1)

# Feature fusion
Conc_Conv = Concatenate()([att_conv_4, hog_layer_2])

# Spatial attention approach
Spatial_attention2 = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(Conc_Conv)
Spatial_attention2_1 = Lambda(lambda x: x[0] * x[1])([Conc_Conv, Spatial_attention2])

# channel-wise attention mechanism
Channel_attention2 = GlobalAveragePooling2D()(Conc_Conv)
Channel_attention2_1 = Reshape((1, 1, 8448))(Channel_attention2)
Channel_attention2_2 = Conv2D(8448, (1, 1), activation='sigmoid', padding='same')(Channel_attention2_1)
Channel_attention2_3 = multiply([Conc_Conv, Channel_attention2_2])

# concatenate the two attention mechanisms
Attn1 = Concatenate()([Spatial_attention2_1, Channel_attention2_3])
batch3 = BatchNormalization()(Attn1)
    
conv6 = Conv2D(256, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(batch3)
conv7 = Conv2D(512, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(conv6)
conv8 = Conv2D(1024, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(conv7)
Batch1 = BatchNormalization()(conv8)

gap_1 = GlobalAveragePooling2D()(Batch1)

# Fully connected balance mechanism
fc_bal = Dense(128, activation='relu')(gap_1)
fc_bal_1 = Dense(64, kernel_regularizer=regularizers.l2(0.05))(fc_bal)
fc_bal_2 = Dropout(0.3)(fc_bal_1)


# Output layer
fc = Dense(7, activation='softmax')(fc_bal_2)


my_model = Model(inputs=input_layer, outputs=fc)
my_model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=0.2), loss='categorical_crossentropy',
                 metrics=['accuracy'])

if __name__ == "__main__":
    history = my_model.fit(training_set, validation_data=Validation_set, callbacks=[lr_reduce, es_callback], epochs=300,
                           class_weight=class_weights)

    # display_training_curves2(history.history['loss'], history.history['val_loss'], 'loss', 211)
    # display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
    # plt.show()

    # Saving the Model
    my_model.save("Wilfried.h5")

