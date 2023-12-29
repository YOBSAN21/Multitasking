import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, GlobalAveragePooling2D, Dense, Reshape, multiply
from PIL import ImageFile


data_path = 'D:/Lari/Multi'

image_size = 512
BATCH_SIZE = 4


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


train_datagen = ImageDataGenerator(rescale=1./255,
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


lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, epsilon=0.001, patience=10, verbose=1,
                                                 mode='min', cooldown=5, min_lr=0.01)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)

counter = Counter(training_set.classes)

max_val = float(max(counter.values()))

class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}


def dual_attention_model(input_shape=(224, 224, 3), num_classes=3):
    # input layer
    input_layer = Input(shape=input_shape)

    # convolutional layers
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)

    # spatial attention mechanism
    sa = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(conv3)
    sa = multiply([conv3, sa])

    # channel-wise attention mechanism
    ca = GlobalAveragePooling2D()(conv3)
    ca = Reshape((1, 1, 128))(ca)
    ca = Conv2D(128, (1, 1), activation='sigmoid', padding='same')(ca)
    ca = multiply([conv3, ca])

    # concatenate the two attention mechanisms
    attn = concatenate([sa, ca])

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(attn)
    pool4 = MaxPooling2D((2, 2))(conv4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    pool5 = MaxPooling2D((2, 2))(conv5)
    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool5)

    # fully connected layers
    flat = GlobalAveragePooling2D()(conv6)
    dense1 = Dense(256, activation='relu')(flat)
    dropout = Dropout(0.5)(dense1)
    dense2 = Dense(num_classes, activation='softmax')(dropout)

    # create model
    model = Model(inputs=input_layer, outputs=dense2)

    return model


my_model = dual_attention_model()

my_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.2), loss='categorical_crossentropy',
                 metrics=['accuracy'])


if __name__ == "__main__":
    history = my_model.fit(training_set, validation_data=Validation_set, callbacks=[lr_reduce, es_callback], epochs=300,
                           class_weight=class_weights)

    # display_training_curves2(history.history['loss'], history.history['val_loss'], 'loss', 211)
    # display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
    # plt.show()
    #
    # #Saving the Model
    # my_model.save("model_D.h5")