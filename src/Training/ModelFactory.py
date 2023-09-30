import keras.layers
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten, Lambda
import tensorflow as tf
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


def get_model_mnih2013(env):
    model = tf.keras.Sequential([
        Lambda(lambda tensor: tf.transpose(tensor, [0, 2, 3, 1]), output_shape=(84, 84, 4), input_shape=(4, 84, 84)),
        Conv2D(16, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(4, 84, 84)),
        Conv2D(32, kernel_size=(4, 4), strides=2, activation="relu"),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(env.action_space.n, activation='linear')
    ])
    rms = tf.keras.optimizers.RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01)
    model.compile(loss="mse", optimizer=rms)

    return model


def get_model_mnih2015(env):
    model = Sequential([
        Lambda(lambda tensor: tf.transpose(tensor, [0, 2, 3, 1]), output_shape=(84, 84, 4), input_shape=(4, 84, 84)),
        Conv2D(32, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(4, 84, 84)),
        Conv2D(64, kernel_size=(4, 4), strides=2, activation="relu"),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(env.action_space.n, activation="linear"),
    ])
    rms = tf.keras.optimizers.RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01)
    model.compile(loss="mse", optimizer=rms)

    return model


def dqn_with_huber_loss_and_adam(env):
    model = Sequential([
        Lambda(lambda tensor: tf.transpose(tensor, [0, 2, 3, 1]), output_shape=(84, 84, 4), input_shape=(4, 84, 84)),
        Conv2D(32, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(4, 84, 84)),
        Conv2D(64, kernel_size=(4, 4), strides=2, activation="relu"),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(env.action_space.n, activation="linear"),
    ])
    adam = tf.keras.optimizers.Adam(learning_rate=0.00025)   # adam with small lr => 1/4 of default lr
    model.compile(loss=tf.keras.losses.Huber(), optimizer=adam)

    return model


def get_interpretable_cnn(env):
    model = keras.Sequential([
        keras.layers.Lambda(lambda tensor: tf.expand_dims(tensor[:, -1, :, :], axis=-1), output_shape=(84, 84),
                            input_shape=(4, 84, 84)),
        keras.layers.Conv2D(32, kernel_size=(4, 4), strides=2, activation="relu", input_shape=(84, 84)),
        keras.layers.Conv2D(64, kernel_size=(4, 4), strides=2, activation="relu"),
        keras.layers.Conv2D(env.action_space.n, kernel_size=(2, 2), strides=2, activation="sigmoid"),
        keras.layers.MaxPool2D(pool_size=(9, 9)),
        keras.layers.Flatten(),
    ])
    adam = tf.keras.optimizers.Adam(learning_rate=0.00025)  # adam with small lr => 1/4 of default lr
    model.compile(loss=tf.keras.losses.Huber(), optimizer=adam)

    return model


def get_explanation_heatmap_for_best_action(model, input_tensor) -> tf.Tensor:
    # get best action for input
    pred = model.predict(input_tensor, verbose=0).reshape(1, -1)
    best_action = np.argmax(pred)

    # remove last layer that's only needed for Training/playing
    interpreter_model = keras.Sequential(model.layers[:-2])

    # get prediction and standardize it
    pred = interpreter_model.predict(input_tensor, verbose=0)[0, :, :, best_action]
    standardized = tf.divide(
       pred - tf.reduce_min(pred),
       tf.reduce_max(pred) - tf.reduce_min(pred)
    )
    return standardized


def plot_explanation_heatmap(input_tensor, heatmap_tensor, vmin=0, vmax=1,
                             title="Highlighted Areas had the Most Influence on the Prediction") -> None:
    """
    Plot the heatmap onto the input image and save it to a file.
    Only use in combination with `get_explanation_heatmap_for_best_action` and the interpretable CNN model!
    :param input_tensor: The input tensor that was fed into the interpretable CNN model (Frame of an atari game)
    :param heatmap_tensor: The heatmap tensor that was returned by `get_explanation_heatmap_for_best_action`
    :param vmin:
    :param vmax:
    :param title: The title of the plot
    """
    ax = sns.heatmap(heatmap_tensor, alpha=0.3, cmap=matplotlib.cm.YlOrRd, vmin=vmin, vmax=vmax, zorder=2 )
    ax.imshow(input_tensor[0, -1, :, :], alpha=1, cmap=matplotlib.cm.binary, aspect=ax.get_aspect(),
              extent=ax.get_xlim() + ax.get_ylim(), zorder=1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(title)
    plt.show()


def save_explanation_heatmap(input_tensor, heatmap_tensor, file_name, vmin=0, vmax=1
                             , title="Highlighted Areas had the Most Influence on the Prediction") -> None:
    """
    Plot the heatmap onto the input image and save it to a file.
    Only use in combination with `get_explanation_heatmap_for_best_action` and the interpretable CNN model!
    :param input_tensor: The input tensor that was fed into the interpretable CNN model (Frame of an atari game)
    :param heatmap_tensor: The heatmap tensor that was returned by `get_explanation_heatmap_for_best_action`
    :param file_name: The name of the file to save the plot to.
    :param vmin:
    :param vmax:
    :param title: The title of the plot
    """
    ax = sns.heatmap(heatmap_tensor, alpha=0.3, cmap=matplotlib.cm.YlOrRd, vmin=vmin, vmax=vmax, zorder=2 )
    ax.imshow(input_tensor[0, -1, :, :], alpha=1, cmap=matplotlib.cm.binary, aspect=ax.get_aspect(),
              extent=ax.get_xlim() + ax.get_ylim(), zorder=1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(title)
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
