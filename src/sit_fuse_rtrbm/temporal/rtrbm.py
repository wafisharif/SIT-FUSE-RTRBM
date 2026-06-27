"""Recurrent Temporal Restricted Boltzmann Machine (Bernoulli-Bernoulli).

DEV NOTE: this file lives in the sit_fuse_rtrbm dev package for now so it
can be built/tested against the real installed `learnergy` package without
naming collisions. Once ready for the actual upstream contribution, this
file gets copied into a fork of github.com/gugarosa/learnergy at
learnergy/models/temporal/rtrbm.py -- see Technical Design Note.

Reference:
    I. Sutskever, G. Hinton, G. Taylor. The recurrent temporal restricted
    Boltzmann machine. NeurIPS (2008).
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import learnergy.utils.constants as c
import learnergy.utils.exception as e
from learnergy.models.bernoulli import RBM
from learnergy.utils import logging

logger = logging.get_logger(__name__)


class RTRBM(RBM):
    """Single-layer Bernoulli-Bernoulli RTRBM. Extends RBM (learnergy's
    bernoulli/rbm.py) with a hidden-to-hidden recurrent weight matrix W'
    and a learnable initial hidden state, so the hidden bias at each
    timestep is conditioned on the mean-field hidden probabilities of the
    PREVIOUS timestep.
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
    ) -> None:
        """Initialization method.

        Args mirror RBM's __init__ exactly (see learnergy's rbm.py) -- no
        new hyperparameters at the base layer beyond what's needed for the
        recurrent connection, added below.
        """

        logger.info("Overriding class: RBM -> RTRBM.")

        super(RTRBM, self).__init__(
            n_visible, n_hidden, steps, learning_rate, momentum,
            decay, temperature, use_gpu,
        )

        # Recurrent hidden-to-hidden weights: W' in the paper.
        self.W_prime = nn.Parameter(torch.randn(n_hidden, n_hidden) * 0.01)

        # Learnable initial hidden state, used at t=0 (no h_{-1} exists).
        self.h0 = nn.Parameter(torch.zeros(n_hidden))

        # Mirrors VarianceGaussianRBM's pattern (gaussian_rbm.py) of
        # registering new nn.Parameters with the optimizer AFTER
        # super().__init__() has already built it.
        self.optimizer.add_param_group({"params": [self.W_prime, self.h0]})

        if self.device == "cuda":
            self.cuda()

        logger.info("Class overrided.")

    @property
    def W_prime(self) -> torch.nn.Parameter:
        """Recurrent hidden-to-hidden weights matrix."""
        return self._W_prime

    @W_prime.setter
    def W_prime(self, W_prime: torch.nn.Parameter) -> None:
        self._W_prime = W_prime

    @property
    def h0(self) -> torch.nn.Parameter:
        """Learnable initial hidden state (used at the first timestep)."""
        return self._h0

    @h0.setter
    def h0(self, h0: torch.nn.Parameter) -> None:
        self._h0 = h0

    def hidden_sampling(
        self, v: torch.Tensor, h_prev: torch.Tensor, scale: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the hidden layer sampling, i.e., P(h_t | v_t, h_{t-1}).

        Overrides RBM.hidden_sampling (learnergy's rbm.py) to add the
        recurrent term. NOTE the changed signature vs. the parent class --
        this now REQUIRES h_prev (previous timestep's mean-field hidden
        probs). Deliberate deviation from RBM's API -- see RTRBM Technical
        Design Note re: forward-compatibility with DBN.forward().

        Args:
            v: Visible layer tensor for the CURRENT timestep, shape
                (batch, n_visible).
            h_prev: Mean-field hidden probabilities from the PREVIOUS
                timestep, shape (batch, n_hidden). Use self.h0 (broadcast
                to batch size) for the first timestep in a sequence.
            scale: same role as in RBM.hidden_sampling -- whether to divide
                by temperature T.

        Returns:
            Probabilities and states of the hidden layer sampling.
        """
        recurrent_bias = F.linear(h_prev, self.W_prime, self.b)
        activations = F.linear(v, self.W.t()) + recurrent_bias

        if scale:
            probs = torch.sigmoid(torch.div(activations, self.T))
        else:
            probs = torch.sigmoid(activations)

        states = torch.bernoulli(probs)

        return probs, states

    def fit(self, dataset, batch_size: int = 128, epochs: int = 10):
        """
        TODO -- NOT YET IMPLEMENTED.

        OPEN QUESTION FOR NICK: training an RTRBM requires propagating the
        mean-field hidden probabilities forward through a whole sequence
        (backprop-through-time, BPTT) for the W_prime/h0 gradients, while
        still using CD-k (per RBM.fit in learnergy's rbm.py) for the
        W/a/b gradients at each individual timestep.

        We have NOT decided yet:
          1. Full BPTT through the whole sequence vs. truncated BPTT vs.
             treating each timestep's CD-k update as independent.
          2. Whether to reuse RBM.fit's batching loop structure, or write a
             sequence-level loop since each "batch" is now a batch of
             SEQUENCES, not independent rows.

        Leaving unimplemented until discussed with Nick.
        """
        raise NotImplementedError(
            "RTRBM.fit: training strategy (BPTT depth, CD-k integration) "
            "is an open design question -- confirm with Nick before "
            "implementing."
        )
