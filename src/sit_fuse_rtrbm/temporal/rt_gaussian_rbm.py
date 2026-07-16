"""Gaussian-Bernoulli Recurrent Temporal Restricted Boltzmann Machine.

Extends RTRBM (Bernoulli visible) with Gaussian visible units, making it
suitable for continuous-valued data like biomechanical time series.

The recurrent mechanism (W_prime, h0, hidden_sampling, fit_subseries,
fit, forward, reconstruct, sample) is inherited unchanged from RTRBM --
recurrence only touches the hidden layer, not the visible one.

Only three things change vs. Bernoulli RTRBM:
  1. energy()          -- quadratic visible term (v - a)^2 instead of -v*a
  2. visible_sampling() -- linear activations, not Bernoulli samples
  3. normalize/input_normalize -- optional per-batch normalization,
     matching GaussianRBM's convention in gaussian_rbm.py

References:
    I. Sutskever, G. Hinton, G. Taylor. The recurrent temporal restricted
    Boltzmann machine. NeurIPS (2008).

    K. Cho, A. Ilin, T. Raiko. Improved learning of Gaussian-Bernoulli
    restricted Boltzmann machines. ICANN (2011).
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from learnergy.utils import logging
from sit_fuse_rtrbm.temporal.rtrbm import RTRBM

logger = logging.get_logger(__name__)


class RTGaussianRBM(RTRBM):
    """Gaussian-Bernoulli Recurrent Temporal RBM.

    Extends RTRBM by replacing Bernoulli visible units with Gaussian
    visible units (variance fixed to 1, same as GaussianRBM in
    learnergy's gaussian_rbm.py). All recurrent machinery inherited
    from RTRBM unchanged.

    Use this instead of RTRBM for continuous-valued input data
    (e.g. biomechanical joint angles, velocities, muscle activations).
    """

    def __init__(
        self,
        n_visible: int = 128,
        n_hidden: int = 128,
        steps: int = 1,
        learning_rate: float = 0.1,
        momentum: float = 0.0,
        decay: float = 0.0,
        temperature: float = 1.0,
        use_gpu: bool = False,
        normalize: bool = True,
        input_normalize: bool = True,
    ) -> None:
        """Initialization method.

        Args mirror GaussianRBM.__init__ (gaussian_rbm.py) with the
        addition of all RTRBM recurrent parameters (W_prime, h0) added
        by the parent RTRBM.__init__.

        Args:
            n_visible: Amount of visible units.
            n_hidden: Amount of hidden units.
            steps: Number of Gibbs' sampling steps.
            learning_rate: Learning rate.
            momentum: Momentum parameter.
            decay: Weight decay used for penalization.
            temperature: Temperature factor.
            use_gpu: Whether GPU should be used or not.
            normalize: Whether or not to use batch normalization during
                fit(). Mirrors GaussianRBM's normalize parameter.
            input_normalize: Whether or not to normalize inputs during
                forward(). Mirrors GaussianRBM's input_normalize parameter.
        """
        # Set normalize flags BEFORE calling super().__init__() --
        # matches the order GaussianRBM.__init__ uses in gaussian_rbm.py.
        self._normalize = normalize
        self._input_normalize = input_normalize

        logger.info("Overriding class: RTRBM -> RTGaussianRBM.")

        super(RTGaussianRBM, self).__init__(
            n_visible,
            n_hidden,
            steps,
            learning_rate,
            momentum,
            decay,
            temperature,
            use_gpu,
        )

        logger.info("Class overrided.")

    @property
    def normalize(self) -> bool:
        """Whether or not to use batch normalization during fit()."""
        return self._normalize

    @normalize.setter
    def normalize(self, normalize: bool) -> None:
        self._normalize = normalize

    @property
    def input_normalize(self) -> bool:
        """Whether or not to normalize inputs during forward()."""
        return self._input_normalize

    @input_normalize.setter
    def input_normalize(self, input_normalize: bool) -> None:
        self._input_normalize = input_normalize

    def energy(self, samples: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Calculates the system's energy for Gaussian visible units.

        Overrides RTRBM.energy() to use the Gaussian visible energy term:
            E = 0.5 * sum((v - a)^2) - sum(softplus(W^T v + W'h + b))

        The quadratic visible term 0.5*(v-a)^2 replaces the linear -v*a
        term used in the Bernoulli case, matching GaussianRBM.energy()
        in gaussian_rbm.py -- the only change for Gaussian visibles.
        The recurrent bias term (W_prime @ h_prev + b) is inherited from
        RTRBM.energy().

        Args:
            samples: Visible samples, shape (batch, n_visible).
            h_prev: Previous hidden probabilities, shape (batch, n_hidden).

        Returns:
            System energy per sample, shape (batch,).
        """
        recurrent_bias = F.linear(h_prev, self.W_prime, self.b)
        activations = F.linear(samples, self.W.t()) + recurrent_bias

        s = nn.Softplus()
        h = torch.sum(s(activations), dim=1)

        # Gaussian visible term -- replaces Bernoulli's -v*a
        # Mirrors GaussianRBM.energy() in gaussian_rbm.py exactly
        v = 0.5 * torch.sum((samples - self.a) ** 2, dim=1)

        energy = v - h

        return energy

    def visible_sampling(
        self, h: torch.Tensor, scale: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs visible layer sampling for Gaussian units, P(v|h).

        Overrides RBM.visible_sampling (inherited via RTRBM). For
        Gaussian visible units, the mean of P(v|h) is simply the linear
        activation W*h + a -- no sigmoid, no Bernoulli sampling. This
        gives continuous-valued outputs instead of binary ones, which is
        exactly what we need for biomechanical data.

        Mirrors GaussianRBM.visible_sampling() in gaussian_rbm.py exactly.

        Args:
            h: Hidden layer tensor, shape (batch, n_hidden).
            scale: Whether to divide by temperature T.

        Returns:
            (probs, states): both are the linear activation for Gaussian
            units (probs = sigmoid of states, states = linear activation).
        """
        activations = F.linear(h, self.W, self.a)

        if scale:
            states = torch.div(activations, self.T)
        else:
            states = activations

        probs = torch.sigmoid(states)

        return probs, states
