from enum import Enum

import numpy as np
import tensorflow as tf


class EnsembleMethods(str, Enum):
    AVERAGE = "average"
    AVERAGE_WITH_CONFIDENCE = "average_with_confidence"
    LOGISTIC_AVERAGE = "logistic_average"
    LOGISTIC_AVERAGE_WITH_CONFIDENCE = "logistic_average_with_confidence"
    MAJORITY_VOTE = "majority_vote"


class Ensemble(tf.keras.Model):
    """
    Ensemble of multiple keras models for CLASSIFICATION. Implements multiple methods for ensembling:
    - averaging
    - averaging with confidence
    - logistic averaging
    - logistic averaging with confidence

    The ensemble is implemented as a keras model and can be used like any other keras model.
    """

    def __init__(self, models, ensemble_method: EnsembleMethods = EnsembleMethods.AVERAGE):
        """
        Initialise the ensemble with the given models and ensemble method. Calls `.compile(run_eagerly=True)`
        on itself to ensure that the model is run eagerly to allow for the use of numpy functions.
        :param models: List of models to be ensembled.
        :param ensemble_method: The type of ensemble to be used. See EnsembleMethods for available methods.
        """
        super(Ensemble, self).__init__()

        if models is None or len(models) == 0:
            raise ValueError("No models provided!")
        if ensemble_method is None or ensemble_method not in EnsembleMethods:
            raise ValueError(f"Ensemble type {ensemble_method} is not valid!")

        self.models = models
        self.ensemble_method = ensemble_method

        ensemble_methods = {
            EnsembleMethods.AVERAGE: self.__average__,
            EnsembleMethods.LOGISTIC_AVERAGE: self.__logistic_average__,
            EnsembleMethods.AVERAGE_WITH_CONFIDENCE: self.__average_with_confidence__,
            EnsembleMethods.LOGISTIC_AVERAGE_WITH_CONFIDENCE: self.__logistic_average_with_confidence__,
            EnsembleMethods.MAJORITY_VOTE: self.__majority_vote__,
        }
        self.__ensemble_method__ = ensemble_methods[ensemble_method]
        self.compile(run_eagerly=True)

    def call(self, x, *args, **kwargs):
        return self.__ensemble_method__(x)

    @tf.function(autograph=False)
    def __get_all_predictions__(self, x):
        return tf.stack([model(x) for model in self.models])

    def __average_with_confidence__(self, x):
        pred = self.__get_all_predictions__(x)
        pred_shape = pred.shape[1:]
        return tf.reshape(tf.convert_to_tensor(
            np.apply_along_axis(self.__calculate_column_avg_with_confidence__, 0, pred)
        ), pred_shape)

    @tf.function(autograph=False)
    def __average__(self, x):
        pred = self.__get_all_predictions__(x)
        pred_shape = pred.shape[1:]
        return tf.reshape(tf.reduce_mean(pred, axis=0), pred_shape)

    @tf.function(autograph=False)
    def __logistic_average__(self, x):
        pred = self.__get_all_predictions__(x)
        pred_shape = pred.shape[1:]
        return tf.reshape(tf.reduce_mean(tf.math.sigmoid(pred), axis=0), pred_shape)

    def __logistic_average_with_confidence__(self, x):
        pred = self.__get_all_predictions__(x)
        pred_shape = pred.shape[1:]
        return tf.reshape(
            np.apply_along_axis(
                self.__calculate_column_avg_with_confidence__, 0, tf.math.sigmoid(pred)
            ), pred_shape)

    @staticmethod
    def __calculate_column_avg_with_confidence__(pred_column):
        return np.average(
            pred_column, axis=0, weights=(pred_column + 0.00001) * 10
        )

    def __majority_vote__(self, x):
        pred_shape = self.models[0](x).shape
        votes = tf.cast(tf.convert_to_tensor(tf.keras.utils.to_categorical(self.__get_all_votes__(x), num_classes=pred_shape[1])), dtype=tf.float32)
        return tf.reshape(
            tf.reduce_sum(votes, axis=0)  # Aggregated votes
            , pred_shape)

    @tf.function(autograph=False)
    def __get_all_votes__(self, x):
        return tf.stack(
            [self.__get_model_prediction_argmax__(model, x) for model in self.models]
        )

    @tf.function(autograph=False)
    def __get_model_prediction_argmax__(self, model, x):
        return tf.argmax(model(x), axis=-1)