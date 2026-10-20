"""Multi-channel asymmetric auto-encoder with an optional classification head (MTL)."""
from dataclasses import dataclass
from typing import Optional, Tuple

from keras import layers, regularizers
from keras.initializers import LecunNormal
from keras.models import Model

from .utils import DEFAULT_SEED

MODES = ("MTL", "classifier", "AE")


@dataclass
class HParams:
    """Hyper-parameters of the model and of its training."""

    num_layers: int = 6           # conv+pool blocks per channel encoder (and deconv layers per decoder)
    kernel_size: int = 5
    filters: int = 8
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    multi_channel: bool = True    # one encoder/decoder per sensor channel; False = one shared for all channels
    encoder_af: str = "selu"
    decoder_af: str = "selu"
    patience: int = 20            # early-stopping patience (epochs) on the validation loss
    min_delta: float = 1e-3
    loss_weights: Optional[Tuple[float, float]] = None  # (reconstruction, classification), MTL mode only
    seed: int = DEFAULT_SEED


class MTLModel(Model):
    """Shared encoder feeding a decoder (signal reconstruction) and/or a classifier.

    Modes:
        ``MTL``        encoder + decoder + classifier, returns ``(reconstruction, class_probabilities)``
        ``classifier`` encoder + classifier (supervised baseline), returns ``class_probabilities``
        ``AE``         encoder + decoder (unsupervised baseline), returns ``reconstruction``
    """

    def __init__(self, signal_length, num_channels, num_classes, hp: HParams, mode="MTL", **kwargs):
        super().__init__(**kwargs)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got '{mode}'")
        self.mode = mode
        self.hp = hp
        self.num_channels = num_channels
        # every conv block halves the time axis, so the latent sequence is signal_length / 2**num_layers
        self.latent_len = int(signal_length / 2 ** hp.num_layers)
        groups = [(c, c + 1) for c in range(num_channels)] if hp.multi_channel else [(0, num_channels)]

        self.encoder = self._build_encoder(signal_length, groups)
        if mode != "AE":
            self.classifier = self._build_classifier(num_classes)
        if mode != "classifier":
            self.decoder = self._build_decoder(groups)

    # ------------------------------------------------------------------ builders
    def _build_encoder(self, signal_length, groups):
        hp = self.hp
        inputs = layers.Input(shape=(signal_length, self.num_channels))
        branches = []
        for start, stop in groups:
            x = layers.BatchNormalization(axis=-1)(inputs[:, :, start:stop])
            for _ in range(hp.num_layers):
                x = self._conv()(x)
                x = layers.MaxPool1D(pool_size=5, strides=2, padding="same")(x)
            branches.append(x)
        fused = layers.Concatenate(axis=-1)(branches)
        latent = layers.Dense(self.num_channels, activation=hp.encoder_af,
                              kernel_initializer=LecunNormal(hp.seed),
                              kernel_regularizer=regularizers.l2(0.01))(fused)
        return Model(inputs, latent, name="encoder")

    def _build_classifier(self, num_classes):
        latent = layers.Input(shape=(self.latent_len, self.num_channels))
        x = layers.Flatten()(latent)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(num_classes, kernel_regularizer=regularizers.l2(0.01))(x)
        return Model(latent, layers.Softmax()(x), name="classifier")

    def _build_decoder(self, groups):
        hp = self.hp
        latent = layers.Input(shape=(self.latent_len, self.num_channels))
        branches = []
        for start, stop in groups:
            x = layers.BatchNormalization(axis=-1)(latent[:, :, start:stop])
            for _ in range(hp.num_layers):
                x = self._deconv()(x)
            branches.append(x)
        fused = layers.Concatenate(axis=-1)(branches)
        out = layers.Conv1D(self.num_channels, hp.kernel_size, activation=hp.decoder_af,
                            kernel_initializer=LecunNormal(hp.seed),
                            kernel_regularizer=regularizers.l2(0.01), strides=1, padding="same")(fused)
        return Model(latent, out, name="decoder")

    def _conv(self):
        hp = self.hp
        if hp.encoder_af == "selu":
            return layers.Conv1D(hp.filters, hp.kernel_size, activation=hp.encoder_af, strides=1,
                                 kernel_initializer=LecunNormal(hp.seed), padding="same")
        return layers.Conv1D(hp.filters, hp.kernel_size, activation=hp.encoder_af, strides=1, padding="same")

    def _deconv(self):
        hp = self.hp
        kwargs = {"kernel_initializer": LecunNormal(hp.seed)} if hp.decoder_af == "selu" else {}
        return layers.Conv1DTranspose(hp.filters, hp.kernel_size, activation=hp.decoder_af, strides=2,
                                      kernel_regularizer=regularizers.l2(0.01), padding="same", **kwargs)

    # ------------------------------------------------------------------ forward
    def call(self, x):
        encoded = self.encoder(x)
        if self.mode == "MTL":
            print(self.decoder(encoded).shape, self.classifier(encoded).shape)
            return self.decoder(encoded), self.classifier(encoded)
        if self.mode == "classifier":
            return self.classifier(encoded)
        return self.decoder(encoded)
