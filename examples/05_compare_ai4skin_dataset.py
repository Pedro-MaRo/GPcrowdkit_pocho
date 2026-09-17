"""Compare every annotator strategy, majority vote, and published aggregators, on AI4SkINv2-2.

`data/ai4skin.py` loads AI4SkINv2-2, a *real* crowdsourcing dataset of skin
whole-slide images (WSIs) graded 0-5 by up to 10 pathologists -- unlike
`data/moons.py`'s synthetic annotators, or `gpcrowdkit.synthetic`'s
per-worker-constant confusion matrices, there is no ground-truth annotator
model here to recover, only real human disagreement. Like the moons dataset,
it ships a clean held-out test split with no annotations at all, so this
script checks the same two things example 04 does: does the crowd model beat
majority vote on the annotated items, and does the latent GP it learns
generalise to WSIs no pathologist ever graded?

It also checks a third thing the synthetic datasets cannot: AI4SkINv2-2 ships
its own precomputed Dawid-Skene, GLAD and MACE aggregations of the same
annotations, so a GPCrowdModel can be benchmarked against established
label-aggregation methods, not just against a recomputed majority vote.

Run with::

    ./run.sh examples/05_compare_ai4skin_dataset.py
    ./run.sh examples/05_compare_ai4skin_dataset.py --iterations 500 --config data/ai4skin.yaml

Requires ``pip install -e ".[data]"`` for `pyyaml`, and a checkout of
AI4SkINv2-2 (see ``data_root`` in ``data/ai4skin.yaml``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gpflow
import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.ai4skin import load_ai4skin_dataset, load_config  # noqa: E402

from gpcrowdkit import (  # noqa: E402
    ALL_ANNOTATOR_STRATEGIES,
    FeatDepDirichletAnnotator,
    FreeCategoricalZ,
    GPCrowdModel,
    SVGPLatent,
    init_alpha_tilde,
    train,
)
from gpcrowdkit.annotators.base import AnnotatorModel  # noqa: E402
from gpcrowdkit.metrics import balanced_accuracy, cross_entropy, label_accuracy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "data" / "ai4skin.yaml",
        help="YAML config for the AI4SkINv2-2 dataset (default: data/ai4skin.yaml).",
    )
    parser.add_argument(
        "--iterations", type=int, default=2000,
        help="Upper bound on optimiser steps per model (with --early-stopping, "
        "usually not all of them run -- see --no-early-stopping).",
    )
    parser.add_argument(
        "--warmup-iterations", type=int, default=0,
        help="Steps run first with the GP kernel/inducing points frozen (paper Algorithm 1).",
    )
    parser.add_argument(
        "--no-early-stopping", action="store_true",
        help="Always train for the full --iterations instead of stopping once the ELBO plateaus.",
    )
    parser.add_argument(
        "--patience", type=int, default=50,
        help="Non-improving checks allowed before stopping early (see --no-early-stopping).",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size (items).")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Adam learning rate.")
    parser.add_argument("--num-inducing", type=int, default=50, help="Inducing points for the GP.")
    return parser.parse_args()


def build_annotator(cls: type[AnnotatorModel], labels, class_probs, X: np.ndarray) -> AnnotatorModel:
    """The only place that knows which strategy is which -- see example 02."""
    if cls is ALL_ANNOTATOR_STRATEGIES[0]:
        return cls(
            labels.num_workers,
            labels.num_classes,
            alpha_tilde_init=init_alpha_tilde(labels, class_probs),
        )
    if cls is FeatDepDirichletAnnotator:
        # Confusion matrices depend on X here, but the X-independent baseline they're a
        # correction on can still be vote-seeded, same as VariationalDirichletAnnotator.
        # X itself is also required: kl_divergence() needs the full training set to stay
        # complete every call, the same invariant the other strategies get for free.
        #
        # Deliberately no hidden_units here: with ~500 training WSIs and 512-1024-d
        # foundation-model features, a trainable D -> hidden layer has more parameters
        # than there are annotations to fit it (see FeatDepDirichletAnnotator's class
        # docstring), which measurably made this strategy worse than majority vote on
        # its own training data. The default projection_dim=16 fixed PCA projection
        # carries no such risk -- it needs no annotation-dependent gradient step at all.
        return cls(
            labels.num_workers, labels.num_classes, X,
            alpha_tilde_init=init_alpha_tilde(labels, class_probs),
        )
    return cls(labels.num_workers, labels.num_classes)


def report_baseline_aggregators(dataset) -> None:
    """Scores AI4SkINv2-2's own precomputed aggregators against the hidden ground truth.

    Unlike majority vote (recomputed below from the sparse annotations, as a
    cross-check that the loader parsed them correctly), Dawid-Skene, GLAD and
    MACE are not implemented in gpcrowdkit -- these are the published numbers
    the dataset ships, included here so a GPCrowdModel has something
    established to beat, not just majority vote.
    """
    print("\n=== AI4SkINv2-2's own aggregators (train pool, where computed) ===")
    for name, pred in dataset.baseline_preds.items():
        available = pred != -1
        if not available.any():
            print(f"{name:<8s} not available for this config (e.g. val items when DS/GLAD/MACE were train-only)")
            continue
        acc = label_accuracy(pred[available], dataset.z_train[available])
        bal = balanced_accuracy(pred[available], dataset.z_train[available])
        print(f"{name:<8s} acc {acc:.3f}   balanced {bal:.3f}   (n={available.sum()}/{len(pred)})")


def main() -> None:
    args = parse_args()
    gpflow.config.set_default_float(np.float64)

    cfg = load_config(args.config)
    dataset = load_ai4skin_dataset(cfg)
    labels = dataset.labels
    class_probs = labels.empirical_class_probs()

    print(f"Dataset: {labels}")
    print(f"  backbone: {cfg.get('backbone', 'CONCH')}   feature dim: {dataset.X_train.shape[1]}")
    print(f"  train items: {dataset.X_train.shape[0]}   held-out test items: {dataset.X_test.shape[0]}\n")

    report_baseline_aggregators(dataset)

    # Majority vote only exists where there are annotations, so it can only be
    # scored on the training split -- it has no way to say anything about an
    # item nobody graded.
    mv_pred = labels.majority_vote()
    mv_train_acc = label_accuracy(mv_pred, dataset.z_train)
    mv_train_bal = balanced_accuracy(mv_pred, dataset.z_train)

    rows = [
        ("majority vote (baseline)", "-", mv_train_acc, mv_train_bal, None, None, None, None, None)
    ]

    X_train_tf = tf.constant(dataset.X_train, dtype=tf.float64)
    X_test_tf = tf.constant(dataset.X_test, dtype=tf.float64)
    # Real embeddings are much higher-dimensional than the moons dataset's 2-D
    # toy features, so a unit lengthscale would saturate the kernel; scale it
    # with sqrt(D) instead, the usual rule of thumb for standardized features.
    lengthscale = float(np.sqrt(dataset.X_train.shape[1]))

    for cls in ALL_ANNOTATOR_STRATEGIES:
        print(f"\nTraining {cls.__name__}...")
        model = GPCrowdModel(
            latent=SVGPLatent(
                kernel=gpflow.kernels.SquaredExponential(lengthscales=lengthscale),
                num_classes=labels.num_classes,
                inducing_points=dataset.X_train[: args.num_inducing].copy(),
            ),
            annotator=build_annotator(cls, labels, class_probs, dataset.X_train),
            num_data=labels.num_items,
            q_z=FreeCategoricalZ(labels.num_items, labels.num_classes, init_probs=class_probs),
        )
        history = train(
            model, dataset.X_train, labels,
            iterations=args.iterations, warmup_iterations=args.warmup_iterations,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            early_stopping=not args.no_early_stopping, patience=args.patience,
        )

        train_pred = model.infer_true_labels(X_train_tf, labels)
        train_acc = label_accuracy(train_pred, dataset.z_train)
        train_bal = balanced_accuracy(train_pred, dataset.z_train)

        test_probs = model.predict_class_probs(X_test_tf).numpy()
        test_pred = np.argmax(test_probs, axis=1)
        test_acc = label_accuracy(test_pred, dataset.z_test)
        test_bal = balanced_accuracy(test_pred, dataset.z_test)
        test_ce = cross_entropy(test_probs, dataset.z_test)

        params_per_worker = sum(
            np.prod(v.shape) for v in model.annotator.trainable_variables
        ) / (labels.num_workers or 1)

        rows.append((
            cls.__name__, params_per_worker, train_acc, train_bal, test_acc, test_bal, test_ce,
            history.elbo[-1], len(history.elbo),
        ))
        print(
            f"  done -- train acc {train_acc:.3f}, test acc {test_acc:.3f}, "
            f"final elbo {history.elbo[-1]:.2f}, {len(history.elbo)} iterations used"
        )

    def fmt(value: float | str | None, spec: str) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, str):
            return value
        return format(value, spec)

    header = (
        f"{'strategy':<28s} {'params/worker':>14s} {'train acc':>10s} {'train bal':>10s} "
        f"{'test acc':>10s} {'test bal':>10s} {'test CE':>10s} {'final elbo':>12s} {'iterations':>11s}"
    )
    separator = "-" * len(header)

    print("\n\n=== Comparison summary ===")
    print(header)
    print(separator)
    for name, params, train_acc, train_bal, test_acc, test_bal, test_ce, elbo, n_iterations in rows:
        print(
            f"{name:<28s} {fmt(params, '>14.0f'):>14s} {fmt(train_acc, '>10.3f'):>10s} "
            f"{fmt(train_bal, '>10.3f'):>10s} {fmt(test_acc, '>10.3f'):>10s} "
            f"{fmt(test_bal, '>10.3f'):>10s} {fmt(test_ce, '>10.3f'):>10s} {fmt(elbo, '>12.2f'):>12s} "
            f"{fmt(n_iterations, '>11d'):>11s}"
        )

    print(
        "\ntrain acc/bal: inferred labels vs. hidden ground truth on the graded WSIs "
        "(the number majority vote, and AI4SkINv2-2's own aggregators above, should be beaten on)."
    )
    print(
        "test acc/bal/CE: the latent GP's predictions on WSIs with *no* annotations at all -- "
        "the generalisation none of the aggregators above can attempt."
    )
    print(
        "iterations: optimiser steps actually run before the ELBO plateaued "
        f"(--no-early-stopping to always run the full --iterations={args.iterations})."
    )


if __name__ == "__main__":
    main()
