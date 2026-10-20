"""Train and evaluate the MTL model on UCI-HAR.

Example:
    python train.py --mode MTL --epochs 100
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from har_mtl.data import load_uci_har
from har_mtl.evaluation import plot_confusion_matrix, predict_labels, report
from har_mtl.models import MODES, HParams
from har_mtl.training import train
from har_mtl.utils import DEFAULT_SEED


def parse_args():
    d = HParams()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=MODES, default="MTL",
                   help="MTL: reconstruction + classification, classifier: supervised baseline, "
                        "AE: unsupervised baseline")
    p.add_argument("--data-dir", default="data", help="where UCI-HAR is (or will be downloaded to)")
    p.add_argument("--output-dir", default="outputs", help="where weights, history and plots are written")
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--learning-rate", type=float, default=d.learning_rate)
    p.add_argument("--num-layers", type=int, default=d.num_layers)
    p.add_argument("--kernel-size", type=int, default=d.kernel_size)
    p.add_argument("--filters", type=int, default=d.filters)
    p.add_argument("--encoder-af", default=d.encoder_af)
    p.add_argument("--decoder-af", default=d.decoder_af)
    p.add_argument("--single-channel", action="store_true",
                   help="use one shared encoder/decoder instead of one per sensor channel")
    p.add_argument("--loss-weights", type=float, nargs=2, metavar=("RECON", "CLS"), default=None,
                   help="loss weights for reconstruction and classification (MTL mode)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main():
    args = parse_args()
    hp = HParams(num_layers=args.num_layers, kernel_size=args.kernel_size, filters=args.filters,
                 epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                 multi_channel=not args.single_channel, encoder_af=args.encoder_af,
                 decoder_af=args.decoder_af, seed=args.seed,
                 loss_weights=tuple(args.loss_weights) if args.loss_weights else None)

    out = Path(args.output_dir) / args.mode
    out.mkdir(parents=True, exist_ok=True)

    dataset = load_uci_har(args.data_dir, seed=args.seed)
    model, history = train(dataset, hp, mode=args.mode)

    model.save_weights(out / "model.weights.h5")
    (out / "history.json").write_text(json.dumps(history.history, indent=2))
    (out / "hparams.json").write_text(json.dumps(asdict(hp), indent=2))

    if args.mode == "AE":
        print("Autoencoder-only mode: no classification metrics. Training history saved to", out)
        return

    y_true = np.argmax(dataset["test"]["label"], axis=1)
    y_pred = predict_labels(model, dataset["test"]["data"], args.mode, args.batch_size)
    (out / "classification_report.txt").write_text(report(y_true, y_pred, dataset["labels"]))
    plot_confusion_matrix(y_true, y_pred, dataset["labels"], path=out / "confusion_matrix.png")
    print("Results written to", out)


if __name__ == "__main__":
    main()
