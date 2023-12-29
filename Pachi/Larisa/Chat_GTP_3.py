import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Dropout, LayerNormalization
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter


data_path = 'D:/Lari/Larisa/Data_1/'
data_path_ = 'D:/Lari/Larisa/Data_1/Train'


image_size = 224
BATCH_SIZE = 4
output_dim = 3


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


class GraphConvolution(layers.Layer):
    def __init__(self, output_dim, activation=None, use_bias=True, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.num_nodes = input_shape[1]
        self.input_dim = input_shape[2]

        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.output_dim,),
                                        initializer='zeros',
                                        trainable=True)
        super(GraphConvolution, self).build(input_shape)

    def call(self, x, adj, mask=None):
        h = tf.matmul(x, self.kernel)

        # apply mask if given
        if mask is not None:
            mask = tf.expand_dims(mask, axis=-1)
            h = h * mask

        # apply adjacency matrix
        output = tf.matmul(adj, h)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 3
        return (input_shape[0], self.num_nodes, self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'activation': self.activation,
            'use_bias': self.use_bias
        }
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DifferentiablePool(layers.Layer):
    def __init__(self, output_dim, activation=None, **kwargs):
        super(DifferentiablePool, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activation

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.num_nodes = input_shape[1]
        self.input_dim = input_shape[2]

        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)

        super(DifferentiablePool, self).build(input_shape)

    def call(self, x):
        h = tf.matmul(x, self.kernel)

        # apply max pooling
        output = tf.reduce_max(h, axis=1)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 3
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'activation': self.activation
        }
        base_config = super(DifferentiablePool, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SpatialAttention(layers.Layer):
    def __init__(self, output_dim, activation=None, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activation

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.num_nodes = input_shape[1]
        self.input_dim = input_shape[2]

        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)

        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim,),
                                    initializer='zeros',
                                    trainable=True)

        super(SpatialAttention, self).build(input_shape)

    def call(self, x):
        h = tf.matmul(x, self.kernel) + self.bias
        alpha = tf.nn.softmax(h, axis=1)

        output = tf.reduce_sum(tf.multiply(x, alpha), axis=1)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 3
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'activation': self.activation
        }
        base_config = super(SpatialAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ChannelAttention(layers.Layer):
    def __init__(self, reduction_ratio=4, activation=None, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.activation = activation

    def build(self, input_shape):
        self.num_channels = input_shape[-1]
        self.pool_size = max(self.num_channels // self.reduction_ratio, 1)

        self.avg_pool = layers.GlobalAveragePooling1D()
        self.max_pool = layers.GlobalMaxPooling1D()

        self.dense1 = layers.Dense(self.pool_size, activation='relu')
        self.dense2 = layers.Dense(self.num_channels, activation='sigmoid')

        super(ChannelAttention, self).build(input_shape)

    def call(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)

        h = tf.concat([avg_pool, max_pool], axis=-1)
        h = self.dense1(h)
        alpha = self.dense2(h)

        output = tf.multiply(x, tf.expand_dims(alpha, axis=1))

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'reduction_ratio': self.reduction_ratio,
            'activation': self.activation
        }
        base_config = super(ChannelAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DualAttentionHierarchicalGraphModel(Model):
    def __init__(self, num_nodes, node_embedding_dim, num_classes, num_layers, hidden_dim):
        super(DualAttentionHierarchicalGraphModel, self).__init__()

        self.node_embedding_dim = node_embedding_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Define input layers
        self.node_inputs = Input(shape=(num_nodes,), name="node_inputs")
        self.edge_inputs = Input(shape=(num_nodes, num_nodes), name="edge_inputs")
        self.mask_inputs = Input(shape=(num_nodes,), name="mask_inputs")

        # Define node embeddings layer
        self.node_embeddings = Embedding(input_dim=num_nodes, output_dim=node_embedding_dim,
                                         name="node_embeddings")

        # Define graph convolutional layers
        self.graph_convolutions = []
        for i in range(num_layers):
            self.graph_convolutions.append(GraphConvolution(hidden_dim, name=f"graph_conv_{i}"))

        # Define pooling layer
        # self.pooling = DifferentiablePool(name="pooling")
        self.pooling = DifferentiablePool(output_dim=3, name="pooling")

        # Define attention layers
        # self.attention_1 = SpatialAttention(name="attention_1")
        self.attention_1 = SpatialAttention(output_dim=3, name="attention_1")
        self.attention_2 = ChannelAttention(name="attention_2")
        # self.attention_2 = ChannelAttention(output_dim=3, name="attention_2")

        # Define classification layers
        self.dropout = Dropout(0.5, name="dropout")
        self.dense = Dense(num_classes, activation="softmax", name="dense")

    def call(self, inputs):
        x = self.node_embeddings(inputs["node_inputs"])
        a = inputs["edge_inputs"]
        mask = inputs["mask_inputs"]

        # Apply graph convolutions
        for i in range(self.num_layers):
            x = self.graph_convolutions[i]([x, a])
            x = LayerNormalization()(x)
            x = tf.nn.relu(x)

        # Apply attention layers
        u1 = self.attention_1([x, mask])
        u2 = self.attention_2([x, mask])

        # Apply pooling layer
        x_pool, a_pool, mask_pool = self.pooling([x, a, mask])

        # Concatenate attention outputs and pooling output
        z = Concatenate()([u1, u2, x_pool])

        # Apply dropout and classification layers
        z = self.dropout(z)
        z = self.dense(z)

        return z


my_model = DualAttentionHierarchicalGraphModel(
    num_nodes = 4,
    node_embedding_dim = 16,
    num_classes = 3,
    num_layers = 3,
    hidden_dim = 8,
)

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