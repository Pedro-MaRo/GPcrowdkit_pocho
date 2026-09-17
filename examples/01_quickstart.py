"""Quickstart: fit a crowdsourcing GP and check it against majority vote.

This is the sanity check every model in this library must clear before it is
trusted on real data: on a dataset where votes alone are not enough (only half
the workers reliable, three annotations per item), does combining the worker
model with the latent classifier actually beat a plain vote count?

Run with::

    ./run.sh examples/01_quickstart.py
"""

from __future__ import annotations

import gpflow
import numpy as np
import tensorflow as tf

from gpcrowdkit import (
    FeatDepVariationalDirichletAnnotator,
    FreeCategoricalZ,
    GPCrowdModel,
    SVGPLatent,
    VariationalDirichletAnnotator,
    init_alpha_tilde,
    make_synthetic,
    train,
)


def accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.asarray(pred) == np.asarray(true)))


def main() -> None:
    gpflow.config.set_default_float(np.float64)

    # Hard CR dataset: only 3 annotations per data, 0.5 of the annotators are reliable.
    data = make_synthetic(
        num_items=300,
        num_features=2,
        num_classes=3,
        num_workers=10,
        labels_per_item=3,
        good_worker_fraction=0.5,
        seed=1,
    )
    labels = data.labels
    print(labels)

    # Initialise q(Z) and the annotator concentrations from the vote histogram,
    # using Laplace smoothing to avoid null probabilities.
    class_probs = labels.empirical_class_probs()

    # Baseline variant: static confusion matrices, shared across all items.
    baseline_model = GPCrowdModel(
        latent=SVGPLatent(
            kernel=gpflow.kernels.SquaredExponential(lengthscales=2.0),
            num_classes=labels.num_classes,
            inducing_points=data.X[:25].copy(),
        ),
        annotator=VariationalDirichletAnnotator(
            labels.num_workers,
            labels.num_classes,
            alpha_tilde_init=init_alpha_tilde(labels, class_probs),
        ),
        num_data=labels.num_items,
        q_z=FreeCategoricalZ(labels.num_items, labels.num_classes, init_probs=class_probs),
    )

    # Feature-dependent variant: confusion matrices vary with the item features.
    feature_model = GPCrowdModel(
        latent=SVGPLatent(
            kernel=gpflow.kernels.SquaredExponential(lengthscales=2.0),
            num_classes=labels.num_classes,
            inducing_points=data.X[:25].copy(),
        ),
        annotator=FeatDepVariationalDirichletAnnotator(
            labels.num_workers,
            labels.num_classes,
            data.X,
            alpha_tilde_init=init_alpha_tilde(labels, class_probs),
        ),
        num_data=labels.num_items,
        q_z=FreeCategoricalZ(labels.num_items, labels.num_classes, init_probs=class_probs),
    )

    def report(iteration: int, elbo: float) -> None:
        if iteration % 50 == 0:
            print(f"  iter {iteration:4d}   elbo {elbo:12.2f}")

    # iterations=1500 is just an upper bound: early_stopping lets each model stop on its
    # own once its ELBO stops improving, rather than both training for the same fixed
    # count regardless of how quickly each actually converges.
    print("\nTraining baseline model...")
    baseline_history = train(
        baseline_model, data.X, labels, iterations=1500, learning_rate=0.05,
        callback=report, early_stopping=True,
    )

    print("\nTraining feature-dependent model...")
    feature_history = train(
        feature_model, data.X, labels, iterations=1500, learning_rate=0.05,
        callback=report, early_stopping=True,
    )

    mv_acc = accuracy(labels.majority_vote(), data.z)
    baseline_acc = accuracy(baseline_model.infer_true_labels(tf.constant(data.X), labels), data.z)
    feature_acc = accuracy(feature_model.infer_true_labels(tf.constant(data.X), labels), data.z)

    print("\nELBO decomposition, first vs last 10 iterations (baseline):")
    for name, series in [
        ("latent", baseline_history.latent),
        ("crowd", baseline_history.crowd),
        ("entropy", baseline_history.entropy),
        ("kl_latent", baseline_history.kl_latent),
        ("kl_annotator", baseline_history.kl_annotator),
        ("total_elbo", baseline_history.elbo),
    ]:
        print(f"  {name:12s} {np.mean(series[:10]):12.2f} -> {np.mean(series[-10:]):12.2f}")

    print("\nAccuracy against the (normally hidden) true labels:")
    print(f"  majority vote : {mv_acc:.3f}")
    print(f"  baseline model : {baseline_acc:.3f}   ({len(baseline_history.elbo)} iterations used)")
    print(f"  feature-dependent model : {feature_acc:.3f}   ({len(feature_history.elbo)} iterations used)")

    est_confusion = baseline_model.annotator.confusion_matrices().numpy()
    confusion_mae = np.abs(est_confusion - data.confusion).mean()
    print(f"\nMean absolute error of recovered worker confusion matrices: {confusion_mae:.3f}")

    assert baseline_acc > mv_acc, "sanity check failed: the baseline model did not beat majority vote"
    print("\nSanity check passed: the baseline model beat majority vote.")
    print("Feature-dependent variant was also trained and evaluated in the same script.")


if __name__ == "__main__":
    main()
