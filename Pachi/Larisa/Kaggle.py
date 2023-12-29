import seaborn as sns
import tensorflow as tf
sns.set_style('darkgrid')
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from collections import Counter

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


lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, epsilon=0.001, patience=10, verbose=1,
                                                 mode='min', cooldown=5, min_lr=0.01)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)

counter = Counter(training_set.classes)

max_val = float(max(counter.values()))

class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}


# create pre-trained model
base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3), pooling='max')

model = Sequential([
    base_model,
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
    Dense(256, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
                bias_regularizer=regularizers.l1(0.006), activation='relu'),
    Dropout(rate=0.5, seed=123),
    Dense(3, activation='softmax')
])

model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 40
patience = 1 		# number of epochs to wait to adjust lr if monitored value does not improve
stop_patience = 3 	# number of epochs to wait before stopping training if monitored value does not improve
threshold = 0.9 	# if train accuracy is < threshhold adjust monitor accuracy, else monitor validation loss
factor = 0.5 		# factor to reduce lr by
freeze = False 		# if true free weights of  the base model


if __name__ == "__main__":
    history = model.fit(training_set, validation_data=Validation_set, callbacks=[lr_reduce, es_callback], epochs=300)

    # display_training_curves2(history.history['loss'], history.history['val_loss'], 'loss', 211)
    # display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
    # plt.show()
    #
    # #Saving the Model
    # my_model.save("model_D.h5")