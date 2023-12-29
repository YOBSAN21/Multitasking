import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Concatenate, GlobalAveragePooling2D, Dense, Lambda, Dropout, Reshape, multiply, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.color import rgb2gray
from skimage.filters import sobel_h, sobel_v, scharr_h, scharr_v
from skimage.feature import canny
from scipy import ndimage as ndi
from keras.preprocessing.image import ImageDataGenerator


data_path = 'F:/willie/OPENSARSHIP_2'


image_size = 224
input_shape = (224, 224, 3)
BATCH_SIZE = 4
num_classes = 6


# Load the SAR RGB image
sar_rgb = plt.imread('sar_rgb_image.png')

# Convert SAR RGB image to grayscale
sar_gray = rgb2gray(sar_rgb)

# Denoise SAR grayscale image using non-local means filter
sigma_est = np.mean(estimate_sigma(sar_gray, multichannel=False))
sar_denoised = denoise_nl_means(sar_gray, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=3)

# Enhance edges using Sobel and Scharr filters
sobel_h = sobel_h(sar_denoised)
sobel_v = sobel_v(sar_denoised)
scharr_h = scharr_h(sar_denoised)
scharr_v = scharr_v(sar_denoised)

# Compute Canny edges using the enhanced edges
edges = canny(sobel_h + sobel_v + scharr_h + scharr_v)

# Fill holes in the edges to improve feature extraction
filled_edges = ndi.binary_fill_holes(edges)

# Display the SAR RGB image and the processed image
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
ax[0, 0].imshow(sar_rgb)
ax[0, 0].set_title('SAR RGB Image')
ax[0, 1].imshow(sar_denoised, cmap='gray')
ax[0, 1].set_title('Denoised SAR Image')
ax[1, 0].imshow(edges, cmap='gray')
ax[1, 0].set_title('Canny Edges')
ax[1, 1].imshow(filled_edges, cmap='gray')
ax[1, 1].set_title('Filled Edges')
plt.show()



train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   preprocessing_function=denoise)


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


lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=0.1)

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', min_delta=0, patience=20, restore_best_weights=True)

counter = Counter(training_set.classes)

max_val = float(max(counter.values()))

class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}


def build_residual_network():
    input_layer = Input(shape=input_shape)

    # input_layer = findpeaks.lee_filter(input_layer, win_size=winsize, cu=cu_value)

    # Label smoothing
    alpha = 0.3
    smoothed_labels = Lambda(lambda x: (1 - alpha) * x + alpha / num_classes)(input_layer)

    # Multiscale and multilevel features extraction
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(smoothed_labels)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    batch1 = BatchNormalization()(pool1)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(batch1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Spatial attention approach
    Spatial_attention1 = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(pool2)
    Spatial_attention1_1 = Lambda(lambda x: x[0] * x[1])([pool2, Spatial_attention1])

    # channel-wise attention mechanism
    Channel_attention1 = GlobalAveragePooling2D()(pool2)
    Channel_attention1_1 = Reshape((1, 1, 32))(Channel_attention1)
    Channel_attention1_2 = Conv2D(32, (1, 1), activation='sigmoid', padding='same')(Channel_attention1_1)
    Channel_attention1_3 = multiply([pool2, Channel_attention1_2])

    # concatenate the two attention mechanisms
    Attn = Concatenate()([Spatial_attention1_1, Channel_attention1_3])
    batch2 = BatchNormalization()(Attn)

    conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(batch2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Spatial attention approach
    Spatial_attention2 = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(pool4)
    Spatial_attention2_1 = Lambda(lambda x: x[0] * x[1])([pool4, Spatial_attention2])

    # channel-wise attention mechanism
    Channel_attention2 = GlobalAveragePooling2D()(pool4)
    Channel_attention2_1 = Reshape((1, 1, 128))(Channel_attention2)
    Channel_attention2_2 = Conv2D(128, (1, 1), activation='sigmoid', padding='same')(Channel_attention2_1)
    Channel_attention2_3 = multiply([pool4, Channel_attention2_2])

    # concatenate the two attention mechanisms
    Attn2 = Concatenate()([Spatial_attention2_1, Channel_attention2_3])

    # upsample
    # pool2x1 = tf.keras.layers.UpSampling2D(size=(2, 2))(pool2)
    # AttnX1 = tf.keras.layers.UpSampling2D(size=(4, 4))(Attn)
    pool3X1 = tf.keras.layers.UpSampling2D(size=(2, 2))(pool3)
    pool4X1 = tf.keras.layers.UpSampling2D(size=(4, 4))(pool4)
    Attn2X1 = tf.keras.layers.UpSampling2D(size=(4, 4))(Attn2)

    # Merge multiscale features using global dependence fusion approach
    concat = Concatenate()([pool2, Attn, pool3X1, pool4X1, Attn2X1])
    batch3 = BatchNormalization()(concat)

    # Continue with feature extraction
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(batch3)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    conv6 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(pool5)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)

    # Spatial attention approach
    Spatial_attention3 = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(pool6)
    Spatial_attention3_1 = Lambda(lambda x: x[0] * x[1])([pool6, Spatial_attention3])

    # channel-wise attention mechanism
    Channel_attention3 = GlobalAveragePooling2D()(pool6)
    Channel_attention3_1 = Reshape((1, 1, 512))(Channel_attention3)
    Channel_attention3_2 = Conv2D(512, (1, 1), activation='sigmoid', padding='same')(Channel_attention3_1)
    Channel_attention3_3 = multiply([pool6, Channel_attention3_2])

    # concatenate the two attention mechanisms
    Attn3 = Concatenate()([Spatial_attention3_1, Channel_attention3_3])

    conv7 = Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu')(Attn3)
    batch4 = BatchNormalization()(conv7)

    gap = GlobalAveragePooling2D()(batch4)

    # Classification layer
    output_layer = Dense(64, kernel_regularizer=regularizers.l2(0.03))(gap)
    output_layer = Dropout(0.5)(output_layer)
    # output_layer = Activation('softmax')(output_layer)
    output_layer = Dense(num_classes, activation='softmax')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


my_model = build_residual_network()

my_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=0.2), loss='categorical_crossentropy',
                 metrics=['accuracy'])


if __name__ == "__main__":
    history = my_model.fit(training_set, validation_data=Validation_set, callbacks=[lr_reduce, es_callback], epochs=300,
                           class_weight=class_weights)

    # display_training_curves2(history.history['loss'], history.history['val_loss'], 'loss', 211)
    # display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
    # plt.show()

    # Saving the Model
    my_model.save("Breast_8.h5")
