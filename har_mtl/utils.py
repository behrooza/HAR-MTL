import random

import numpy as np

DEFAULT_SEED = 41


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy and Keras/TensorFlow for reproducible runs."""
    import keras

    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)  # also seeds TensorFlow
