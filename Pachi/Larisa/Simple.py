import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import schedules
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter


data_path = 'D:/Lari/Larisa/Data_1/'
data_path_ = 'D:/Lari/Larisa/Data_1/Train'

s_1 = os.listdir(data_path_ + "/Bones")
s_2 = os.listdir(data_path_ + "/Brain")
s_3 = os.listdir(data_path_ + "/Normal")

Nos_Train = len(s_1) + len(s_2) + len(s_3)

image_size = 224
BATCH_SIZE = 1
num_classes = 3
RESIZE_TO = 384
CROP_TO = 224
NUM_CLASSES = 3  # number of classes
SCHEDULE_LENGTH = (
    500  # we will train on lower resolution images and will still attain good results
)
SCHEDULE_BOUNDARIES = [
    200,
    300,
    400,
]  # more the dataset size the schedule length increase


train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=15,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(data_path + '/Train',
                                                 target_size=(image_size, image_size),
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='categorical', shuffle=True)

testing_set = test_datagen.flow_from_directory(data_path + '/Val',
                                               target_size=(image_size, image_size),
                                               batch_size=BATCH_SIZE,
                                               class_mode='categorical', shuffle=True)


def display_training_curves(training_accuracy, validation_accuracy, title, subplot):
    if subplot % 10 == 1:  # set up the subplots on the first call
        plt.subplots(figsize=(10, 10))
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.plot(training_accuracy)
    ax.plot(validation_accuracy)
    ax.set_title('Model ' + title)
    ax.set_ylabel(title)
    ax.set_xlabel('Epoch')
    ax.legend(['Train_Accuracy', 'Val_Accuracy'])


def display_training_curves2(training_loss, validation_loss, title, subplot):
    if subplot % 10 == 1:  # set up the subplots on the first call
        plt.subplots(figsize=(10, 10))
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.plot(training_loss)
    ax.plot(validation_loss)
    ax.set_title('Model ' + title)
    ax.set_ylabel(title)
    ax.set_xlabel('Epoch')
    ax.legend(['Train_Loss', 'Val_Loss'])


# label smoothing
def categorical_smooth_loss(y_true, y_pred, label_smoothing=0.1):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
    return loss


# training call backs
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, epsilon=0.01, patience=3, verbose=1)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
counter = Counter(training_set.classes)
max_val = float(max(counter.values()))
class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}


# DesNet_Architecture
bit_model_url = "https://tfhub.dev/google/bit/m-r50x1/1"
bit_module = hub.KerasLayer(bit_model_url)


class MyBiTModel(keras.Model):
    def __init__(self, num_classes, module, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.head = keras.layers.Dense(num_classes, kernel_initializer="zeros")
        self.bit_model = module

    def call(self, images):
        bit_embedding = self.bit_model(images)
        return self.head(bit_embedding)


model = MyBiTModel(num_classes=NUM_CLASSES, module=bit_module)

learning_rate = 0.0001

# Decay learning rate by a factor of 10 at SCHEDULE_BOUNDARIES.
lr_schedule = schedules.PiecewiseConstantDecay(
    boundaries=SCHEDULE_BOUNDARIES,
    values=[
        learning_rate,
        learning_rate * 0.1,
        learning_rate * 0.01,
        learning_rate * 0.001,
    ],
)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

train_callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=2, restore_best_weights=True
    )
]

if __name__ == "__main__":

    # history = model.fit(training_set, validation_data=testing_set, epochs=100)
    history = model.fit(training_set, batch_size=BATCH_SIZE, epochs=100, validation_data=testing_set, callbacks=train_callbacks)

    # def plot_hist(hist):
    #     plt.plot(hist.history["accuracy"])
    #     plt.plot(hist.history["val_accuracy"])
    #     plt.plot(hist.history["loss"])
    #     plt.plot(hist.history["val_loss"])
    #     plt.title("Training Progress")
    #     plt.ylabel("Accuracy/Loss")
    #     plt.xlabel("Epochs")
    #     plt.legend(["train_acc", "val_acc", "train_loss", "val_loss"], loc="upper left")
    #     plt.show()
    #
    # plot_hist(history)

    # display_training_curves2(history.history['loss'], history.history['val_loss'], 'loss', 211)
    # display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
    # plt.show()

    # Saving the Model
    model.save("model_D.tf")
