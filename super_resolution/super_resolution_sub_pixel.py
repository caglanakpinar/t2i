import tensorflow as tf
from keras import ops, layers, Input, Model, optimizers, losses

from mlp import BaseModel, Params, log


class DepthToSpace(layers.Layer):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def call(self, input):
        batch, height, width, depth = ops.shape(input)
        depth = depth // (self.block_size**2)

        x = ops.reshape(
            input, [batch, height, width, self.block_size, self.block_size, depth]
        )
        x = ops.transpose(x, [0, 1, 3, 2, 4, 5])
        x = ops.reshape(
            x, [batch, height * self.block_size, width * self.block_size, depth]
        )
        return x


class SuperResolution(BaseModel):
    def __init__(self, params: Params):
        self.params = params
        self.hidden_layers = self.params.get('hidden_layers')
        self.hidden_units = self.params.get('hidden_units')
        self.channels = self.params.get('channels')
        self.kernel_size_ENCODER = self.params.get('kernel_size').get('encoder')
        self.kernel_size_DECODER = self.params.get('kernel_size').get('decoder')
        self.conv_args = {
            "activation": self.params.get("conv_activation"),
            "kernel_initializer": self.params.get('conv_kernel_initializer'),
            "padding": self.params.get('conv_padding'),
        }
        self.upscale_factor = self.params.get('upscale_factor')
        self.loss_fn = losses.MeanSquaredError()
        self.optimizer = optimizers.Adam(learning_rate=self.params.get('lr'))
        self.epochs = self.params.epochs
        self.build()

    def build(self):
        _units = self.cal_hidden_layer_of_units(
            self.hidden_layers,
            self.hidden_units
        )
        inputs = Input(shape=(None, None, self.channels))
        _hidden = inputs
        for unit in reversed(_units):
            _hidden = layers.Conv2D(unit, self.kernel_size_DECODER, **self.conv_args)(_hidden)
        for unit in _units:
            _hidden = layers.Conv2D(unit, self.kernel_size_ENCODER, **self.conv_args)(_hidden)
        _hidden = layers.Conv2D(self.channels * (self.upscale_factor ** 2), 3, **self.conv_args)(_hidden)
        outputs = DepthToSpace(self.upscale_factor)(_hidden)
        self.model = Model(inputs, outputs)
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
        )
        log(log.info, self.model.summary())

    def train(self, dataset: tuple[tf.data.Dataset, tf.data.Dataset]):
        self.model.fit(
            dataset[0], epochs=self.epochs, validation_data=dataset[1], verbose=2
        )
