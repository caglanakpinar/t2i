import tensorflow as tf
from keras import Input, layers, Model, optimizers

from mlp import BaseModel, Params, log


class UNet(BaseModel):
    def __init__(self, params: Params):
        self.params = params
        self.hidden_layers = self.params.get('hidden_layers')
        self.hidden_units = self.params.get('hidden_units')
        self.num_classes = self.params.get('num_classes')
        self.epochs = self.params.epochs
        self.img_size = self.params.get('img_size')
        self.build()

    def build(self):
        _units = self.cal_hidden_layer_of_units(
            self.hidden_layers,
            self.hidden_units
        )
        inputs = Input(shape=self.img_size + (3,))

        # !!! down sampling
        # Entry block
        _hidden = layers.Conv2D(
            self.hidden_units, 3, strides=2, padding="same")(inputs)
        _hidden = layers.BatchNormalization()(_hidden)
        _hidden = layers.Activation(self.params.activation)(_hidden)

        previous_block_activation = _hidden  # Set aside residual

        for unit in list(reversed(_units))[1:]:
            _hidden = layers.Activation(self.params.activation)(_hidden)
            _hidden = layers.SeparableConv2D(unit, 3, padding="same")(_hidden)
            _hidden = layers.BatchNormalization()(_hidden)
            _hidden = layers.Activation(self.params.activation)(_hidden)
            _hidden = layers.SeparableConv2D(unit, 3, padding="same")(_hidden)
            _hidden = layers.BatchNormalization()(_hidden)
            _hidden = layers.MaxPooling2D(3, strides=2, padding="same")(_hidden)
            # Project residual
            residual = layers.Conv2D(unit, 1, strides=2, padding="same")(
                previous_block_activation
            )
            _hidden = layers.add([_hidden, residual])
            previous_block_activation = _hidden

        # !!! up sampling
        for unit in _units:
            _hidden = layers.Activation(self.params.activation)(_hidden)
            _hidden = layers.Conv2DTranspose(unit, 3, padding="same")(_hidden)
            _hidden = layers.BatchNormalization()(_hidden)
            _hidden = layers.Activation(self.params.activation)(_hidden)
            _hidden = layers.Conv2DTranspose(unit, 3, padding="same")(_hidden)
            _hidden = layers.BatchNormalization()(_hidden)
            _hidden = layers.UpSampling2D(2)(_hidden)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(unit, 1, padding="same")(residual)
            _hidden = layers.add([_hidden, residual])  # Add back residual
            previous_block_activation = _hidden  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(
            self.num_classes, 3,
            activation=self.params.get("output_activation"), padding="same"
        )(_hidden)
        self.model = Model(inputs, outputs)
        self.model.compile(
            optimizer=optimizers.Adam(self.params.get('lr')),
            loss=self.params.get('loss'),
        )
        log(log.info, self.model.summary())

    def train(self, dataset: tuple[tf.data.Dataset, tf.data.Dataset]):
        self.model.fit(
            dataset[0],
            epochs=self.epochs,
            validation_data=dataset[1],
            verbose=2,
        )
