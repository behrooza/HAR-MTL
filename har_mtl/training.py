"""Build, compile and fit the model for one of the three modes."""
import keras
from keras import backend as K
from keras import ops
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from .models import HParams, MTLModel
from .utils import set_seed


def rmse(y_true, y_pred):
    """Root-mean-square error over time, kept per sample and channel."""
    return ops.sqrt(ops.mean(ops.square(y_pred - y_true), axis=1))


def train(dataset, hp: HParams = None, mode="MTL"):
    """Train a model on ``dataset`` (see :func:`har_mtl.data.load_uci_har`) and return ``(model, history)``."""
    hp = hp or HParams()
    set_seed(hp.seed)

    x_tr, y_tr = dataset["train"]["data"], dataset["train"]["label"]
    x_val, y_val = dataset["validation"]["data"], dataset["validation"]["label"]
    _, signal_length, num_channels = x_tr.shape
    print("Shape the data", x_tr.shape, y_tr.shape, x_val.shape, y_val.shape)
    model = MTLModel(signal_length, num_channels, len(dataset["labels"]), hp, mode=mode)
    ce = keras.losses.CategoricalCrossentropy(from_logits=False)
    loss_weights = None

    if mode == "MTL":
        # Positional lists (reconstruction, classification): subclassed models have no reliable output names.
        loss = ["huber", ce]
        metrics = [[rmse], ["accuracy"]]
        if hp.loss_weights is not None:
            loss_weights = list(hp.loss_weights)
        fit_y, val_y = [x_tr, y_tr], [x_val, y_val]
    elif mode == "classifier":
        loss, metrics = ce, ["accuracy"]
        fit_y, val_y = y_tr, y_val
    else:  # AE
        loss, metrics = "huber", [rmse]
        fit_y, val_y = x_tr, x_val

    model.compile(optimizer=Adam(learning_rate=hp.learning_rate), loss=loss, metrics=metrics,
                  loss_weights=loss_weights)
    model.build((None, signal_length, num_channels))
    model.summary()

    early_stopping = EarlyStopping(monitor="val_loss", patience=hp.patience, min_delta=hp.min_delta,
                                   restore_best_weights=True)
    history = model.fit(x_tr, fit_y, epochs=hp.epochs, batch_size=hp.batch_size,
                        validation_data=(x_val, val_y), shuffle=True, callbacks=[early_stopping])
    return model, history
