"""Evaluation helpers: predictions, classification report and confusion matrix."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def predict_labels(model, x, mode="MTL", batch_size=128):
    """Return integer class predictions for ``x`` (``MTL`` and ``classifier`` modes only)."""
    out = model.predict(x, batch_size=batch_size, verbose=0)
    probs = out[1] if mode == "MTL" else out
    return np.argmax(probs, axis=1)


def report(y_true, y_pred, labels):
    """Print and return sklearn's classification report."""
    text = classification_report(y_true, y_pred, target_names=labels, digits=3)
    print(text)
    return text


def plot_confusion_matrix(y_true, y_pred, labels, path=None, show=False):
    """Plot a row-normalised confusion matrix; save it to ``path`` if given."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, normalize="true", cmap="Greens",
        values_format=".2f", xticks_rotation="vertical", ax=ax, im_kw={"vmin": 0, "vmax": 1})
    fig.tight_layout()
    if path:
        fig.savefig(path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
