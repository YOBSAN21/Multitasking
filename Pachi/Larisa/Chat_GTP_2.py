import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, GlobalAveragePooling2D, Dense, Reshape, multiply, Flatten


data_path = 'D:/Lari/Larisa/Data_1/'
data_path_ = 'D:/Lari/Larisa/Data_1/Train'

image_size = 224
BATCH_SIZE = 4

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


lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, epsilon=0.0001, patience=10, verbose=1,
                                                 mode='min', cooldown=5, min_lr=0.01)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)

counter = Counter(training_set.classes)

max_val = float(max(counter.values()))

class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}


def tri_attention_patch_model(input_shape=(224, 224, 3), num_classes=3):
    # input layer
    input_layer = Input(shape=input_shape)

    # convolutional layers
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    print(conv3.shape)

    # patch-based attention mechanism
    patches = []
    for i in range(4):
        for j in range(4):
            patch = conv3[:, i * 56:(i + 1) * 56, j * 56:(j + 1) * 56, :]
            patch_gap = GlobalAveragePooling2D()(patch)
            patch_dense = Dense(32, activation='relu')(patch_gap)
            patches.append(patch_dense)
    patch_attention = concatenate(patches)
    patch_attention = Dense(128, activation='sigmoid')(patch_attention)
    patch_attention = Reshape((1, 1, 128))(patch_attention)
    print(patch_attention.shape)
    patch_attention = Conv2D(128, (1, 1), activation='sigmoid', padding='same')(patch_attention)
    print(patch_attention.shape)
    patch_attention = multiply([conv3, patch_attention])


    # spatial attention mechanism
    spatial_attention = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(conv3)
    spatial_attention = multiply([conv3, spatial_attention])

    # channel-wise attention mechanism
    channel_attention = GlobalAveragePooling2D()(conv3)
    channel_attention = Reshape((1, 1, 128))(channel_attention)
    channel_attention = Conv2D(128, (1, 1), activation='sigmoid', padding='same')(channel_attention)
    channel_attention = multiply([conv3, channel_attention])

    # concatenate the three attention mechanisms
    attn = concatenate([patch_attention, spatial_attention, channel_attention])

    # fully connected layers
    flat = Flatten()(attn)
    dense1 = Dense(256, activation='relu')(flat)
    dropout = Dropout(0.5)(dense1)
    dense2 = Dense(num_classes, activation='softmax')(dropout)

    # create model
    model = Model(inputs=input_layer, outputs=dense2)

    return model


my_model = tri_attention_patch_model()

my_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.2), loss='categorical_crossentropy',
                 metrics=['accuracy'])


if __name__ == "__main__":
    history = my_model.fit(training_set, validation_data=Validation_set, callbacks=[lr_reduce, es_callback], epochs=300)

    # display_training_curves2(history.history['loss'], history.history['val_loss'], 'loss', 211)
    # display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
    # plt.show()
    #
    # #Saving the Model
    # my_model.save("model_D.h5")