"""Concrete annotator strategies.

Four interchangeable worker-noise models, all satisfying the
[AnnotatorModel][gpcrowdkit.annotators.base.AnnotatorModel] object. The core engine
never learns which one it holds, so adding a fifth requires no change
anywhere else in the library.

They span the useful range of complexity:

============================            =========================  ==============================
Strategy                                Parameters                 Worker uncertainty
============================            =========================  ==============================
VariationalDirichletAnnotator           ``A * C * C``              full posterior over ``R``
SoftmaxPointAnnotator                   ``A * C * C``              none (point estimate)
OneCoinAnnotator                        ``A``                      none (point estimate)
FeatDepDirichletAnnotator               ``A * emb * C * C``        none (point estimate)
============================            =========================  ==============================

[OneCoinAnnotator][gpcrowdkit.annotators.strategies.OneCoinAnnotator] is the one that justifies the shape of the base
class. It carries a single scalar per worker, so an interface promising an
``[A, C, C]`` parameter tensor would have been the wrong abstraction -- it
happens to *build* such a tensor, but it does not store one.

Index convention throughout: tensors are ``[A, C_obs, C_true]``, normalised
down axis 1, so each column is one distribution over observed labels given a
fixed true class.
"""

from __future__ import annotations

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import positive

from ..data import CrowdBatch, CrowdLabels
from .base import AnnotatorModel, ConfusionAnnotator

__all__ = [
    "VariationalDirichletAnnotator",
    "SoftmaxPointAnnotator",
    "OneCoinAnnotator",
    "FeatDepDirichletAnnotator",
    "FeatDepVariationalDirichletAnnotator",
    "init_alpha_tilde",
    "ALL_ANNOTATOR_STRATEGIES",
]

FLOAT = tf.float64


class VariationalDirichletAnnotator(ConfusionAnnotator):
    """Full variational Dirichlet posterior over confusion matrices (SVGPCR).

    Places an independent Dirichlet on each *column* of each worker's confusion
    matrix -- one distribution over observed labels per true class::

        q(R^a_{.,j}) = Dir(alpha_tilde^a_{.,j})
        p(R^a_{.,j}) = Dir(alpha^a_{.,j})

    This is the strategy from SVGPCR:

    Morales-Alvarez, P., Ruiz, P., Coughlin, S., Molina, R., & Katsaggelos, A. K. (2022).
    Scalable Variational Gaussian Processes for Crowdsourcing: Glitch Detection in LIGO.
    IEEE transactions on pattern analysis and machine intelligence, 44(3), 1534-1551.

    The only one here
    that represents *uncertainty* about a worker rather than a best guess. That
    matters when workers are sparse: a worker with three annotations and a
    worker with three thousand can have identical point estimates but wildly
    different posteriors, and only this strategy can tell them apart.

    Attributes:
        alpha (gpflow.Parameter): Prior concentrations, non-trainable.
            Shape ``[A, C_obs, C_true]``.
        alpha_tilde (gpflow.Parameter): Variational concentrations, trainable,
            constrained positive. Shape ``[A, C_obs, C_true]``.
    """

    def __init__(
        self,
        num_workers: int,
        num_classes: int,
        alpha_prior: np.ndarray | float = 1.0,
        alpha_tilde_init: np.ndarray | None = None,
    ) -> None:
        """Initialises the prior and variational Dirichlet concentrations.

        Args:
            num_workers: Number of annotators ``A``.
            num_classes: Number of classes ``C``.
            alpha_prior: Prior concentrations: a scalar broadcast over
                ``[A, C, C]``, or an explicit array of that shape. A flat 1.0
                (the reference implementation's choice) is the uniform
                Dirichlet -- no prior opinion about any worker.
            alpha_tilde_init: Optional ``[A, C, C]`` starting point. Defaults to
                a mild diagonal bias, encoding the assumption that workers are
                better than chance. See [init_alpha_tilde][gpcrowdkit.annotators.strategies.init_alpha_tilde] for the
                data-driven alternative, which converges considerably faster.

        Note:
            ``alpha`` is stored as a non-trainable ``Parameter`` rather than a
            ``tf.constant`` so that it appears in ``gpflow.utilities.print_summary``
            alongside everything else, and so that a subclass can make the
            prior learnable (empirical Bayes) by flipping one flag.
        """
        super().__init__(num_workers, num_classes)
        shape = (self.A, self.C, self.C)
        # Adapt the alpha_prior to the shape and type required by the flow.
        prior = np.broadcast_to(np.asarray(alpha_prior, dtype=np.float64), shape).copy()
        # Set the prior as a gpflow parameter and lock it to non-trainable.
        self.alpha = gpflow.Parameter(prior, transform=positive(), trainable=False)

        # By default, initialise the annotators to be better than random (1+1/C on the diagonal, 1/C off-diagonal).
        if alpha_tilde_init is None:
            alpha_tilde_init = np.full(shape, 1.0 / self.C) + np.stack(
                [np.eye(self.C) for _ in range(self.A)]
            )
        self.alpha_tilde = gpflow.Parameter(
            np.asarray(alpha_tilde_init, dtype=np.float64), transform=positive()
        )

    def expected_log_confusion(self) -> tf.Tensor:
        """``E_q[log R] = psi(alpha_tilde) - psi(sum_i alpha_tilde_{i,j})``.

        The standard Dirichlet identity. Note this is an expectation of a
        logarithm, computed exactly -- not the logarithm of the mean, which
        would be a different and biased quantity.

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, float64.
        """
        a = self.alpha_tilde
        return tf.math.digamma(a) - tf.math.digamma(tf.reduce_sum(a, axis=1, keepdims=True))

    def kl_divergence(self) -> tf.Tensor:
        """Analytic Dirichlet-Dirichlet KL, summed over workers and columns.

        For a single column, ``KL(Dir(q) || Dir(p))`` is::

            log B(p) - log B(q) + sum_i (q_i - p_i) (psi(q_i) - psi(sum q))

        The implementation evaluates all ``A * C`` columns at once by
        distributing that expression across the tensor.

        Returns:
            tf.Tensor: Scalar, non-negative, and exactly zero when
            ``alpha_tilde == alpha``.

        Note:
            ``tf.math.lbeta`` reduces the *last* axis, but the Dirichlet lives
            along axis 1 (observed classes). ``matrix_transpose`` swaps the last
            two axes so that the reduction lands on the right one. Omitting it
            computes a log-beta over true classes instead: still finite, still
            differentiable, silently wrong.
        """
        q, p = self.alpha_tilde, self.alpha
        diff = q - p
        term1 = tf.reduce_sum(diff * tf.math.digamma(q))
        term2 = -tf.reduce_sum(
            tf.math.digamma(tf.reduce_sum(q, axis=1)) * tf.reduce_sum(diff, axis=1)
        )
        term3 = tf.reduce_sum(
            tf.math.lbeta(tf.linalg.matrix_transpose(p))
            - tf.math.lbeta(tf.linalg.matrix_transpose(q))
        )
        return term1 + term2 + term3

    def confusion_matrices(self) -> tf.Tensor:
        """Posterior mean ``E_q[R] = alpha_tilde / sum_i alpha_tilde_{i,j}``.

        The Dirichlet is the normalised vector of independent Gammas, so its
        mean is each concentration over their sum.

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, columns summing to 1.

        Note:
            Deliberately *not* ``softmax(expected_log_confusion())``. That is
            the normalised geometric mean, which is more sharply peaked than
            the posterior mean and so overstates worker accuracy -- by around
            0.06 on the diagonal at ``alpha = [10, 1, 1]``, and more as the
            concentrations shrink. The bias is therefore largest exactly where
            data is scarce and the estimate matters most.
        """
        return self.alpha_tilde / tf.reduce_sum(self.alpha_tilde, axis=1, keepdims=True)


class SoftmaxPointAnnotator(ConfusionAnnotator):
    """Deterministic point-estimate confusion matrices, ``R^a = softmax(W^a)``.

    Same parameter count as the Dirichlet strategy but no distribution over
    them, so no KL term and no notion of how confident the estimate is. Cheaper
    and often adequate when every worker has plenty of annotations.

    Attributes:
        logits (gpflow.Parameter): Unconstrained logits. Shape ``[A, C, C]``.
    """

    def __init__(self, num_workers: int, num_classes: int, diagonal_init: float = 2.0) -> None:
        """Initialises the logits with a diagonal bias.

        Args:
            num_workers: Number of annotators ``A``.
            num_classes: Number of classes ``C``.
            diagonal_init: Logit mass on the diagonal at initialisation.
                Softmax of ``2.0`` on the diagonal gives a competent-but-not-
                certain worker, a reasonable neutral start.

        Note:
            No constraint transform is needed: the softmax in
            [expected_log_confusion][gpcrowdkit.annotators.base.ConfusionAnnotator.expected_log_confusion] handles normalisation, so the raw
            logits are free parameters over all of R.
        """
        super().__init__(num_workers, num_classes)
        init = np.tile(np.eye(num_classes) * diagonal_init, (num_workers, 1, 1))
        self.logits = gpflow.Parameter(init.astype(np.float64))

    def expected_log_confusion(self) -> tf.Tensor:
        """``log R = log_softmax(logits)`` down the observed-class axis.

        Exact rather than an expectation, since ``R`` is deterministic here.

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, float64.

        Note:
            ``log_softmax`` rather than ``log(softmax(...))``: it subtracts the
            max internally, so it stays finite for logits where the naive
            composition would produce ``log(0) = -inf``.
        """
        return tf.nn.log_softmax(self.logits, axis=1)

    def kl_divergence(self) -> tf.Tensor:
        """Zero: a point estimate carries no distribution to penalise."""
        return tf.constant(0.0, dtype=FLOAT)

    def confusion_matrices(self) -> tf.Tensor:
        """Exact confusion matrices, ``softmax(logits)``.

        No expectation is involved, so unlike the Dirichlet case this really is
        the exponential of [expected_log_confusion][gpcrowdkit.annotators.base.ConfusionAnnotator.expected_log_confusion].

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, columns summing to 1.
        """
        return tf.nn.softmax(self.logits, axis=1)


class OneCoinAnnotator(ConfusionAnnotator):
    """Single scalar accuracy per worker -- the classic one-coin model.

    Worker ``a`` is correct with probability ``beta_a`` and otherwise spreads
    the remaining mass uniformly::

        R^a_{ij} = beta_a               if i == j
                   (1 - beta_a)/(C-1)   otherwise

    Strong assumption -- it cannot express a worker who systematically confuses
    two particular classes -- but with ``A`` parameters instead of ``A*C*C`` it
    is far better behaved when annotations per worker are scarce, which is the
    usual situation.

    It is also the design test for the base contract: it stores ``A`` scalars,
    not a confusion tensor. An interface that had promised ``[A, C, C]``
    parameters would have been the wrong abstraction.

    Attributes:
        beta_logit (gpflow.Parameter): Unconstrained per-worker accuracy
            logits. Shape ``[A]``.
    """

    def __init__(self, num_workers: int, num_classes: int, init_accuracy: float = 0.7) -> None:
        """Initialises per-worker accuracies.

        Args:
            num_workers: Number of annotators ``A``.
            num_classes: Number of classes ``C``, at least 2.
            init_accuracy: Initial accuracy shared by all workers, in ``(0, 1)``.

        Raises:
            ValueError: If ``num_classes < 2`` (the off-diagonal mass would be
                divided by zero), or if ``init_accuracy`` is not in ``(0, 1)``.

        Note:
            Stored as a logit with the sigmoid applied in [beta][gpcrowdkit.annotators.strategies.OneCoinAnnotator.beta], rather
            than as a constrained ``Parameter``. Both work; the logit keeps the
            dependency surface small and makes the unconstrained optimisation
            explicit at the point of use.
        """
        if num_classes < 2:
            raise ValueError("OneCoinAnnotator requires at least two classes.")
        if not 0.0 < init_accuracy < 1.0:
            raise ValueError("init_accuracy must lie strictly in (0, 1).")
        super().__init__(num_workers, num_classes)
        logit = float(np.log(init_accuracy / (1.0 - init_accuracy)))
        self.beta_logit = gpflow.Parameter(np.full(num_workers, logit, dtype=np.float64))

    @property
    def beta(self) -> tf.Tensor:
        """Per-worker accuracy in ``(0, 1)``. Shape ``[A]``."""
        return tf.sigmoid(self.beta_logit)

    def _confusion(self) -> tf.Tensor:
        """Builds the ``[A, C, C]`` confusion tensor from the ``A`` scalars.

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, columns summing to 1.
        """
        beta = self.beta[:, None, None]  # [A, 1, 1]
        eye = tf.eye(self.C, dtype=FLOAT)[None, :, :]  # [1, C, C]
        off = (1.0 - beta) / tf.constant(self.C - 1, dtype=FLOAT)
        return eye * beta + (1.0 - eye) * off

    def expected_log_confusion(self) -> tf.Tensor:
        """Log of the constructed confusion tensor.

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, float64.

        Note:
            The ``1e-12`` floor guards ``log(0)`` if the sigmoid saturates at
            0 or 1 during optimisation. Saturation is itself a symptom -- a
            worker driven to perfect accuracy usually means too few
            annotations and no prior to regularise them.
        """
        return tf.math.log(self._confusion() + 1e-12)

    def kl_divergence(self) -> tf.Tensor:
        """Zero: a point estimate carries no distribution to penalise."""
        return tf.constant(0.0, dtype=FLOAT)

    def confusion_matrices(self) -> tf.Tensor:
        """Exact confusion matrices built from the per-worker accuracies.

        Returns:
            tf.Tensor: Shape ``[A, C_obs, C_true]``, columns summing to 1.
        """
        return self._confusion()


def init_alpha_tilde(
    labels: CrowdLabels, class_probs: np.ndarray, prior_strength: float = 1.0
) -> np.ndarray:
    """Data-driven initialisation of the variational Dirichlet concentrations.

    Reproduces ``_init_behaviors`` from the reference implementation: each
    annotation ``(n, a, y)`` adds the soft class assignment ``class_probs[n]``
    to ``alpha_tilde[a, y, :]``, so a worker who labels an item "cat" when the
    votes point to "cat" accumulates diagonal mass.

    Starting here rather than from a flat prior matters. The ELBO is not convex,
    and from a uniform start the model can settle into a labelling that is a
    permutation of the truth -- self-consistent, high-likelihood, and useless.
    Vote-based initialisation breaks that symmetry before optimisation begins.

    Args:
        labels: The annotations.
        class_probs: Soft per-item class assignments ``[N, C]``, typically
            [empirical_class_probs][gpcrowdkit.data.CrowdLabels.empirical_class_probs].
        prior_strength: Constant added to every entry, keeping concentrations
            comfortably positive and the digamma well conditioned.

    Returns:
        np.ndarray: Shape ``[A, C_obs, C_true]``, float64.

    Note:
        The two nested Python loops of the original become two ``np.add.at``
        scatter-adds, turning an ``O(L)`` interpreter loop into two vectorised
        passes. On a dataset with millions of annotations this is the
        difference between minutes and seconds of setup.
    """
    A, C = labels.num_workers, labels.num_classes
    acc = np.full((A, C, C), 1.0 / C)
    counts = np.ones((A, C))

    np.add.at(acc, (labels.worker_idx, labels.label), class_probs[labels.item_idx])
    np.add.at(counts, (labels.worker_idx, labels.label), 1.0)

    acc /= counts[:, :, None]
    acc *= (counts / counts.sum(axis=1, keepdims=True))[:, :, None]
    acc /= acc.sum(axis=1, keepdims=True)
    return acc + prior_strength


def _inverse_softplus(x: np.ndarray) -> np.ndarray:
    """Inverts ``softplus``: returns ``y`` such that ``log1p(exp(y)) == x``.

    Used to seed `FeatDepDirichletAnnotator`'s per-annotator head biases so
    that ``softplus(bias) + 1e-3`` reproduces a target concentration array
    (typically [init_alpha_tilde][gpcrowdkit.annotators.strategies.init_alpha_tilde]'s output) exactly at
    initialisation. ``log(expm1(x))`` rather than ``log(exp(x) - 1)``: `expm1`
    avoids the catastrophic cancellation of computing ``exp(x) - 1`` directly
    for the modest, near-1 concentrations `init_alpha_tilde` produces.

    Args:
        x: Positive array, shape arbitrary.

    Returns:
        np.ndarray: Same shape as ``x``.
    """
    return np.log(np.expm1(np.maximum(x, 1e-6)))


def _alpha_tilde_to_head_bias(alpha_tilde_init: np.ndarray, num_classes: int) -> np.ndarray:
    """Converts an ``[A, C_obs, C_true]`` concentration array to per-annotator head biases.

    A `FeatDepDirichletAnnotator` head's raw ``C * C``-vector output is
    reshaped as ``(C_true, C_obs)`` row-major, then transposed to
    ``[C_obs, C_true]`` (see `FeatDepDirichletAnnotator._alpha_tilde_per_annotation`).
    This applies the inverse mapping, in inverse-softplus space, so that a
    head with this bias and a zero kernel reproduces ``alpha_tilde_init``
    exactly regardless of its input.

    Args:
        alpha_tilde_init: Shape ``[A, C_obs, C_true]``.
        num_classes: ``C``.

    Returns:
        np.ndarray: Shape ``[A, C * C]``.
    """
    A = alpha_tilde_init.shape[0]
    logits = _inverse_softplus(np.asarray(alpha_tilde_init, dtype=np.float64) - 1e-3)  # [A, C_obs, C_true]
    logits = np.transpose(logits, (0, 2, 1))  # [A, C_true, C_obs]
    return logits.reshape(A, num_classes * num_classes)


def _init_trunk_params(
    feature_dim: int, hidden_units: list[int]
) -> tuple[list[gpflow.Parameter], list[gpflow.Parameter]]:
    """He-normal-initialised kernels and zero biases for a stack of ReLU layers.

    Plain `gpflow.Parameter`s rather than `tf.keras.layers.Dense`, matching
    every other strategy in this module -- see `FeatDepDirichletAnnotator`'s
    class docstring for why that also matters for correctly freezing the
    trunk later, not just for consistency.

    Args:
        feature_dim: Input width ``D``.
        hidden_units: Output width of each layer; ``hidden_units[-1]`` is the
            final embedding dimension.

    Returns:
        tuple[list[gpflow.Parameter], list[gpflow.Parameter]]: Kernels
        (shapes ``[fan_in, fan_out]``) and biases (shapes ``[fan_out]``), one
        pair per entry of ``hidden_units``.
    """
    dims = [feature_dim] + list(hidden_units)
    kernels, biases = [], []
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        scale = np.sqrt(2.0 / fan_in)
        kernels.append(gpflow.Parameter(np.random.normal(scale=scale, size=(fan_in, fan_out))))
        biases.append(gpflow.Parameter(np.zeros(fan_out)))
    return kernels, biases


def _fit_pca(X: np.ndarray, projection_dim: int | float) -> tuple[np.ndarray, np.ndarray]:
    """Fits a whitened, ``D -> r`` PCA projection directly from data -- no labels, no gradient descent.

    `FeatDepDirichletAnnotator` uses this, not a randomly-initialised trainable
    layer, for its first ``D -> r`` step. Where ``D`` is small relative to the
    dataset (e.g. 2-D synthetic features over thousands of items) a random
    projection has plenty of data to be shaped away from noise by the ELBO
    gradient. Where ``D`` is large relative to the dataset -- e.g. AI4SkINv2-2's
    512-1024-d foundation-model embeddings over some 500 whole-slide images --
    a single random ``Dense(D, r)`` layer already has ``D * r`` parameters
    (16,384 at ``D=512, r=32``, several times the ~3,800 real annotations
    available to fit them) with nothing but that noisy annotation likelihood
    pulling on it and no prior at all pulling it back. Empirically, on
    AI4SkINv2-2 this is not merely "undertrained": the strategy ends up
    *worse than majority vote on the very training items it was fit on*
    (0.66 vs. 0.77 accuracy) -- the signature of a projection that has been
    pulled toward some direction the sparse, noisy annotation signal favours
    by chance, not a genuinely useful summary of ``X``. PCA cannot do that: it
    is a closed-form, unsupervised decomposition of ``X``'s own covariance,
    using every item regardless of whether it has any annotations at all, so
    it is exactly as well-conditioned at ``D=1024`` as at ``D=2`` and carries
    no risk of fitting annotation noise -- there is no annotation-dependent
    gradient step involved.

    Args:
        X: Training features, shape ``[N, D]``.
        projection_dim: Either a fixed number of components ``r`` (int), or
            the minimum fraction of variance to retain (float in ``(0, 1]``)
            -- the same two conventions `sklearn.decomposition.PCA`'s own
            ``n_components`` accepts. Silently capped at
            ``min(N, D)`` components, the most a PCA of ``X`` can ever
            produce.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(mean, components)``. ``mean`` has
        shape ``[D]``; ``components`` has shape ``[D, r]`` and is scaled
        (whitened) so that ``(X - mean) @ components`` has unit variance in
        every one of its ``r`` columns, on ``X`` itself -- without whitening,
        the earliest components (by far the largest-variance directions of a
        typical foundation-model embedding) would dominate the per-annotator
        head's linear map over the later ones purely by scale, not by how
        informative they actually are about annotator behaviour.
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    mean = X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)

    if isinstance(projection_dim, float):
        if not 0.0 < projection_dim <= 1.0:
            raise ValueError(f"projection_dim as a float must be in (0, 1], got {projection_dim}")
        explained = np.cumsum(S**2) / np.sum(S**2)
        r = int(np.searchsorted(explained, projection_dim) + 1)
    else:
        r = int(projection_dim)
    r = min(r, len(S))

    scale = S[:r] / np.sqrt(max(N - 1, 1))
    scale = np.where(scale > 1e-12, scale, 1.0)  # guards a direction with ~zero variance in X
    components = Vt[:r].T / scale  # [D, r]
    return mean, components


class FeatDepDirichletAnnotator(AnnotatorModel):
    """Feature-dependent Dirichlet posterior: a shared trunk, one head per annotator.

    note: It inherits from AnnotatorModel instead of ConfusionAnnotator because
    the confusion matrices are now dependent on the input features X, and the implementation
    doesn't match the structure of ConfusionAnnotator.

    Three stages, replacing the single combined network an earlier version of
    this strategy used (one shared trunk fed ``(x, annotator embedding,
    one-hot true class)`` and evaluated once per ``(item, annotator, class)``
    combination):

    * a **fixed projection**, ``X -> PCA(X)`` (see `_fit_pca`): a whitened
      PCA basis computed once from ``X`` at construction time, no gradient
      descent involved. This, not a randomly-initialised trainable layer, is
      what makes this strategy usable on real, high-dimensional features
      (e.g. AI4SkINv2-2's 512-1024-d foundation-model embeddings over a few
      hundred whole-slide images): a random ``Dense(D, r)`` there has more
      parameters than there are annotations to fit them, and empirically
      that setup does not merely undertrain -- it fits noise and ends up
      *worse than majority vote on its own training items*. PCA cannot do
      that: it has no annotation-dependent gradient step to overfit with, and
      is exactly as well-conditioned at ``D=1024`` as at ``D=2``;
    * an optional **shared trunk**, ``PCA(X) -> embedding`` (widths given by
      `hidden_units`, empty by default -- i.e. the embedding *is* the PCA
      projection unless you opt into more capacity), trained on every
      annotation regardless of who made it;
    * one **head per annotator**, a linear map ``embedding -> C * C``
      (reshaped to that worker's ``[C_obs, C_true]`` Dirichlet-column
      concentrations) that only that worker's own annotations ever train.

    The ``A`` heads are stored as a single pair of tensors with a leading
    ``A`` axis (`head_kernel`, `head_bias`) rather than as ``A`` separate
    layers -- the same idiom `VariationalDirichletAnnotator.alpha_tilde`,
    `SoftmaxPointAnnotator.logits` and `OneCoinAnnotator.beta_logit` already
    use for their own per-worker parameters. This is what keeps
    `label_log_terms` cheap: for a batch of ``L`` annotations it gathers just
    the ``L`` relevant (item embedding, worker head) pairs and runs one
    batched ``einsum`` -- no ``[B, A, C, C]`` tensor, and no per-``(item,
    annotator, class)`` forward pass, is ever built. Annotator identity is no
    longer a network *input* at all: it is which head gets used. The shared
    trunk is likewise plain matrices rather than `tf.keras` layers, for the
    same reason every other strategy here is: a `gpflow.Parameter` is a
    `tf.Variable` whose own ``trainable`` flag is exactly what
    `gpflow.Module.trainable_variables` (what `train` optimises) checks,
    whereas a Keras layer's ``.trainable`` flag is bookkeeping local to Keras
    and is invisible to that check -- freezing a Keras layer here would
    silently do nothing (see `add_workers`, where genuinely freezing the
    trunk matters).

    ``alpha_tilde(x, a)`` starts as a residual on top of a vote-based
    baseline: each head's kernel is zero-initialised and its bias seeded (in
    inverse-softplus space) from `init_alpha_tilde`, so at step 0 every
    annotator's confusion is exactly `VariationalDirichletAnnotator`'s
    vote-seeded, ``X``-independent estimate, and only grows feature-dependent
    as training moves a head's kernel away from zero -- avoiding the flat,
    symmetric starting point `init_alpha_tilde`'s docstring warns can settle
    into "a labelling that is a permutation of the truth".

    Unlike `VariationalDirichletAnnotator`, this strategy keeps no Dirichlet
    posterior over ``R`` at all -- `head_kernel`/`head_bias` are point
    estimates, exactly in the spirit of `SoftmaxPointAnnotator`, and
    `kl_divergence` is (by default) a plain, optional L2 penalty on
    `head_kernel` alone rather than a KL term. This was not the original
    design: an earlier version computed a genuine per-annotation
    ``KL(Dirichlet(alpha_tilde(x_n, a)) || Dirichlet(alpha))``, on the
    reasoning that since ``alpha_tilde`` now varies by item, the textbook
    treatment of an amortised variational posterior is to sum its KL once per
    realised annotation (the same way a VAE's KL term is a sum over data
    points) and rescale by ``N/B`` for minibatching, mirroring the
    ``latent``/``crowd``/``entropy`` ELBO terms. That reasoning is internally
    consistent, but it has a damaging practical consequence this class does
    not actually want: an annotator's *single* confusion estimate gets
    counted once *per annotation they made*, so a prolific annotator's
    estimate is pulled toward the prior far harder than a sparse annotator's
    -- backwards from the usual, and desired, behaviour where more evidence
    should let an estimate move *further* from the prior with less relative
    penalty. On AI4SkINv2-2 (10 pathologists, a few hundred annotations each)
    this was not a subtle effect: the per-annotation KL for a single 50-item
    batch measured ~2400, dwarfing the ~6 that `VariationalDirichletAnnotator`
    pays in total, *exactly*, over the whole dataset -- and the trained model
    ended up worse than majority vote on its own training items (0.65 vs.
    0.77 accuracy), worse even than freezing `head_kernel` at exactly zero
    (which isolates the bug: with no feature-dependence contributing anything
    at all, this is mathematically the same estimator `VariationalDirichletAnnotator`
    computes, just parameterised differently -- and it should not have scored
    any differently). Dropping the per-annotation KL and letting `head_kernel`
    be a bare point estimate (`kernel_l2=0.0`, the default) recovered the
    ~0.78 accuracy every other strategy gets on that dataset; a small
    `kernel_l2` was no better or worse in that experiment, but is exposed for
    datasets where some shrinkage on top of the zero-init warm start turns out
    to help. `head_bias` is never penalised: it is the ``X``-independent,
    vote-seeded baseline every head starts at, and there is nothing
    suspicious about it settling wherever the (unregularised) vote-based
    estimate itself would.

    See `add_workers` for folding in a new annotator's data later without
    retraining the whole model from scratch.

    Attributes:
        embedding_dim (int): Width of the shared item embedding -- either
            `projection_dim`'s actual component count, or ``hidden_units[-1]``
            when a trunk is used on top of the projection.
        head_kernel (gpflow.Parameter): Per-annotator weight matrices.
            Shape ``[A, embedding_dim, C * C]``.
        head_bias (gpflow.Parameter): Per-annotator biases. Shape ``[A, C * C]``.
    """

    def __init__(
        self,
        num_workers: int,
        num_classes: int,
        X: np.ndarray,
        projection_dim: int | float = 16,
        hidden_units: list[int] = [],
        kernel_l2: float = 0.0,
        alpha_tilde_init: np.ndarray | None = None,
        name: str | None = None,
    ) -> None:
        """Initialises the fixed projection, the optional trunk, and the per-annotator heads.

        Args:
            num_workers: Number of annotators ``A``.
            num_classes: Number of classes ``C``.
            X: Full training feature matrix, shape ``[N, D]``. Used only to
                fit the PCA projection (see `_fit_pca`).
            projection_dim: Width ``r`` of the fixed PCA projection ``D -> r``
                -- either a fixed component count (int, the default is 16),
                or the minimum fraction of variance to retain (float in
                ``(0, 1]``). See `_fit_pca` for why this replaces a trainable
                first layer. Capped at ``min(N, D)``, so on a low-dimensional
                dataset (e.g. 2-D synthetic features) this is a harmless
                whitening rotation of the original space, not an actual
                reduction.
            hidden_units: Widths of an *optional* trainable ReLU trunk applied
                on top of the fixed projection. Empty by default -- i.e. the
                embedding fed to every annotator's head is the PCA projection
                itself. Pass e.g. ``[16]`` to add one trainable 16-unit
                refinement layer on top, only worth doing when there is
                enough annotation data to support the extra capacity (see the
                class docstring's AI4SkINv2-2 numbers for what "not enough"
                looks like).
            kernel_l2: Weight of an optional ``lambda * sum(head_kernel ** 2)``
                penalty, returned by `kl_divergence`. Zero (a bare point
                estimate) by default -- see the class docstring for why a
                Dirichlet KL is deliberately not used here, and why zero was
                already enough to fix the regression that motivated removing
                it. A small positive value is a cheap way to try further
                shrinkage of the feature-dependent correction back toward the
                ``X``-independent baseline, without touching `head_bias`.
            alpha_tilde_init: Optional ``[A, C, C]`` starting point for every
                head's bias, in the same convention as
                `VariationalDirichletAnnotator`'s own parameter of the same
                name -- typically
                [init_alpha_tilde][gpcrowdkit.annotators.strategies.init_alpha_tilde],
                so every strategy can be seeded from the same call. Defaults
                to the same mild diagonal bias `VariationalDirichletAnnotator`
                falls back to when unset.
            name: Optional module name, forwarded to ``gpflow.Module``.
        """
        super().__init__(num_workers, num_classes, name=name)

        feature_mean, pca_components = _fit_pca(X, projection_dim)
        self._feature_mean = gpflow.Parameter(feature_mean, trainable=False)
        self._pca_projection = gpflow.Parameter(pca_components, trainable=False)  # [D, r]
        projected_dim = pca_components.shape[1]

        self.embedding_dim = int(hidden_units[-1]) if hidden_units else projected_dim
        self._trunk_kernels, self._trunk_biases = _init_trunk_params(projected_dim, hidden_units)
        self.kernel_l2 = float(kernel_l2)

        if alpha_tilde_init is None:
            alpha_tilde_init = np.full((self.A, self.C, self.C), 1.0 / self.C) + np.stack(
                [np.eye(self.C) for _ in range(self.A)]
            )
        # Zero kernel + vote-seeded bias: see class docstring for why this
        # starts every head numerically identical to VariationalDirichletAnnotator.
        self.head_kernel = gpflow.Parameter(
            np.zeros((self.A, self.embedding_dim, self.C * self.C), dtype=np.float64)
        )
        self.head_bias = gpflow.Parameter(_alpha_tilde_to_head_bias(alpha_tilde_init, self.C))

        # Cache of the most recent label_log_terms() batch, for
        # confusion_matrices()'s X=None fallback (for reporting).
        self._last_X: tf.Tensor | None = None

    def _embed(self, X: tf.Tensor) -> tf.Tensor:
        """Shared per-item embedding, common to every annotator. Shape ``[B, embedding_dim]``.

        Always starts from the fixed, whitened PCA projection (see
        `_fit_pca`), then applies whatever trainable trunk layers
        `hidden_units` asked for -- none, by default.
        """
        h = tf.matmul(tf.cast(X, FLOAT) - self._feature_mean, self._pca_projection)
        for kernel, bias in zip(self._trunk_kernels, self._trunk_biases):
            h = tf.nn.relu(tf.matmul(h, kernel) + bias)
        return h

    def _alpha_tilde_per_annotation(
        self, embedding: tf.Tensor, item_local: tf.Tensor, worker_idx: tf.Tensor
    ) -> tf.Tensor:
        """``alpha_tilde`` for exactly the ``L`` annotations in the batch.

        Gathers only the item embedding and worker head each annotation
        actually needs, rather than the dense product over every ``(item,
        worker)`` pair a ``[B, A, C, C]`` tensor would require.

        Returns:
            tf.Tensor: Shape ``[L, C_obs, C_true]``.
        """
        emb_l = tf.gather(embedding, item_local)  # [L, E]
        kernel_l = tf.gather(self.head_kernel, worker_idx)  # [L, E, C*C]
        bias_l = tf.gather(self.head_bias, worker_idx)  # [L, C*C]
        raw = tf.einsum("le,lec->lc", emb_l, kernel_l) + bias_l  # [L, C*C]
        raw = tf.reshape(raw, (-1, self.C, self.C))  # [L, C_true, C_obs]
        raw = tf.linalg.matrix_transpose(raw)  # [L, C_obs, C_true]
        return tf.nn.softplus(raw) + 1e-3

    def label_log_terms(self, batch: CrowdBatch) -> tf.Tensor:
        """See [AnnotatorModel.label_log_terms][gpcrowdkit.annotators.base.AnnotatorModel.label_log_terms]."""
        self._last_X = batch.X
        embedding = self._embed(batch.X)
        alpha_tilde = self._alpha_tilde_per_annotation(embedding, batch.item_local, batch.worker_idx)

        E_log_R = tf.math.digamma(alpha_tilde) - tf.math.digamma(
            tf.reduce_sum(alpha_tilde, axis=1, keepdims=True)
        )  # [L, C_obs, C_true]
        idx = tf.stack([tf.range(tf.shape(batch.label)[0]), batch.label], axis=-1)  # [L, 2]
        return tf.gather_nd(E_log_R, idx)  # [L, C_true]

    def kl_divergence(self) -> tf.Tensor:
        """Optional ``kernel_l2 * sum(head_kernel ** 2)`` penalty; zero when ``kernel_l2 == 0``.

        See the class docstring for why this is a plain L2 term on
        `head_kernel` rather than a Dirichlet KL: a per-annotation KL is the
        textbook-correct treatment for a posterior that genuinely varies with
        ``x``, but it penalises an annotator's estimate in proportion to how
        many annotations they made -- backwards from the usual "more evidence
        should mean less relative penalty" -- and measurably made this
        strategy worse than majority vote on a real dataset. This term has no
        such dependence on annotation counts: it is a fixed function of
        `head_kernel` alone, independent of the batch.

        Returns:
            tf.Tensor: Scalar.
        """
        if self.kernel_l2 == 0.0:
            return tf.constant(0.0, dtype=FLOAT)
        return tf.constant(self.kernel_l2, dtype=FLOAT) * tf.reduce_sum(tf.square(self.head_kernel))

    def confusion_matrices(self, X: tf.Tensor | None = None) -> tf.Tensor:
        """Posterior mean confusion matrix per item and annotator.

        Unlike `label_log_terms`, which only ever evaluates the one worker
        who produced each annotation, this evaluates every annotator's head
        against every item in ``X`` -- meant for reporting/diagnostics (e.g.
        plotting recovered vs. true confusion matrices for a handful of
        workers), not the training path.

        Args:
            X: Items to evaluate. Defaults to the most recent batch passed to
                `label_log_terms`.

        Returns:
            tf.Tensor: Shape ``[B, A, C_obs, C_true]``, columns summing to 1.

        Raises:
            ValueError: If ``X`` is omitted and `label_log_terms` has not run yet.
        """
        target_X = X if X is not None else self._last_X
        if target_X is None:
            raise ValueError(
                "confusion_matrices() requires X, or a prior label_log_terms() call to cache one."
            )
        embedding = self._embed(target_X)  # [B, E]
        raw = tf.einsum("be,aec->bac", embedding, self.head_kernel) + self.head_bias[None]  # [B, A, C*C]
        raw = tf.reshape(raw, (tf.shape(raw)[0], self.A, self.C, self.C))  # [B, A, C_true, C_obs]
        raw = tf.linalg.matrix_transpose(raw)  # [B, A, C_obs, C_true]
        alpha_tilde = tf.nn.softplus(raw) + 1e-3
        return alpha_tilde / tf.reduce_sum(alpha_tilde, axis=2, keepdims=True)

    def add_workers(self, num_new: int, freeze_trunk: bool = False) -> None:
        """Extends the model to cover ``num_new`` additional annotators.

        Appends ``num_new`` fresh rows to `head_kernel`/`head_bias` --
        zero-initialised kernels and a neutral, ``X``-independent bias, the
        same "start indistinguishable from majority vote" recipe the
        constructor uses -- without touching any existing annotator's row or
        the shared trunk's learned weights. New workers are appended at the
        end of the existing index range, ``[A_old, A_old + num_new)``,
        matching how `CrowdLabels` assigns worker indices via ``np.unique``.

        Because `label_log_terms` only ever gathers the row of the worker who
        actually produced each annotation, continuing training on a dataset
        that mixes old and new annotators leaves an old worker's row exactly
        where it was for every step in which that worker has no annotation in
        the sampled batch -- old annotators need no explicit protection. The
        shared trunk is different: it receives gradient from *every*
        annotation regardless of worker, so training after this call keeps
        adapting it to the combined data unless `freeze_trunk` says otherwise.

        Args:
            num_new: Number of annotators to add.
            freeze_trunk: If True, rebuilds the trunk's parameters as
                non-trainable (copying their current, already-learned
                values), so further training touches only annotator heads --
                old and new -- never the shared feature representation the
                old annotators' estimates also depend on. Leave False to let
                the trunk keep adapting as new data arrives -- usually what
                you want, since a trunk that has learned from more annotators
                generalises better for all of them. Freezing is done by
                recreating the trunk's `gpflow.Parameter`s with
                ``trainable=False``, not by toggling a `tf.keras.layers.Layer`
                -- see the class docstring for why the latter would silently
                have no effect under this library's training loop.
        """
        if num_new <= 0:
            raise ValueError(f"num_new must be positive, got {num_new}")

        new_kernel = np.zeros((num_new, self.embedding_dim, self.C * self.C), dtype=np.float64)
        neutral_init = np.tile(
            np.full((self.C, self.C), 1.0 / self.C) + np.eye(self.C), (num_new, 1, 1)
        )
        new_bias = _alpha_tilde_to_head_bias(neutral_init, self.C)

        self.head_kernel = gpflow.Parameter(
            np.concatenate([self.head_kernel.numpy(), new_kernel], axis=0)
        )
        self.head_bias = gpflow.Parameter(
            np.concatenate([self.head_bias.numpy(), new_bias], axis=0)
        )
        self.A += num_new

        if freeze_trunk:
            self._trunk_kernels = [
                gpflow.Parameter(k.numpy(), trainable=False) for k in self._trunk_kernels
            ]
            self._trunk_biases = [
                gpflow.Parameter(b.numpy(), trainable=False) for b in self._trunk_biases
            ]


ALL_ANNOTATOR_STRATEGIES = (
    VariationalDirichletAnnotator,
    SoftmaxPointAnnotator,
    OneCoinAnnotator,
    FeatDepDirichletAnnotator,
)

# Backward-compatible alias: some examples and docs use the longer, explicit name.
FeatDepVariationalDirichletAnnotator = FeatDepDirichletAnnotator
