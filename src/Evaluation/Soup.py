import tensorflow as tf


class Soup:
    def __init__(self, list_of_models: list):
        assert len(list_of_models) > 0, "Soup needs at least 1 model!"
        assert all([isinstance(model, tf.keras.Sequential) for model in
                    list_of_models]), "All models must be keras Sequential models!"
        assert len(set([len(model.get_weights()) for model in
                        list_of_models])) == 1, "All models must have the same architecture!"

        self._list_of_models = list_of_models
        self.soup_model = self._create_soup_model()

    def _create_soup_model(self):
        # Create deep copy of first model
        soup_model = tf.keras.models.clone_model(self._list_of_models[0])

        # Set weights of soup model to average of all models
        new_layer_weights = [tf.math.add_n(l) / len(self._list_of_models) for l in
                             zip(*[model.get_weights() for model in self._list_of_models])]
        soup_model.set_weights(new_layer_weights)

        return soup_model

    def get_soup_model(self):
        return self.soup_model

    def add_ingredient(self, model):
        assert isinstance(model, tf.keras.Sequential), "Model must be a keras Sequential model!"
        assert len(model.get_weights()) == len(self.soup_model.get_weights()), \
            "Model must have the same architecture as the soup model!"

        self._list_of_models.append(model)
        self.soup_model = self._create_soup_model()
