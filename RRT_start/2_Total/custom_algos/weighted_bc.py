from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from d3rlpy.algos import QLearningAlgoBase
from d3rlpy.algos.qlearning.bc import DiscreteBCConfig as OriginalConfig
from d3rlpy.algos.qlearning.torch.bc_impl import (
    DiscreteBCImpl,
    DiscreteBCModules,
)
from d3rlpy.models.torch import DiscreteImitationLoss
from d3rlpy.constants import ActionSpace
from d3rlpy.models.builders import create_categorical_policy


# ===============================
# NIEUWE IMPLEMENTATIE
# ===============================


class WeightedDiscreteBCImpl(DiscreteBCImpl):
    def __init__(
        self,
        *args,
        class_weights: Optional[Sequence[float]] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            self._class_weights = w.to(self._device)
        else:
            self._class_weights = None

    def compute_loss(self, obs_t, act_t):
        dist = self._modules.imitator(obs_t)
        penalty = (dist.logits ** 2).mean()
        log_probs = F.log_softmax(dist.logits, dim=1)

        imitation_loss = F.nll_loss(
            log_probs,
            act_t.long().view(-1),
            weight=self._class_weights,
        )

        regularization_loss = self._beta * penalty

        return DiscreteImitationLoss(
            loss=imitation_loss + regularization_loss,
            imitation_loss=imitation_loss,
            regularization_loss=regularization_loss,
        )


# ===============================
# CONFIG
# ===============================


class DiscreteBCConfig(OriginalConfig):
    """
    Drop-in replacement voor d3rlpy DiscreteBCConfig,
    maar met extra parameter:

        class_weights = [w0, w1, ...]

    """

    def __init__(
        self,
        *args,
        class_weights: Optional[Sequence[float]] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def create(self, device=False, enable_ddp=False):
        return DiscreteBC(self, device, enable_ddp)


# ===============================
# ALGO WRAPPER
# ===============================


class DiscreteBC(QLearningAlgoBase):
    def inner_create_impl(self, observation_shape, action_size):
        imitator = create_categorical_policy(
            observation_shape,
            action_size,
            self._config.encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )

        optim = self._config.optim_factory.create(
            imitator.named_modules(),
            lr=self._config.learning_rate,
            compiled=self.compiled,
        )

        modules = DiscreteBCModules(optim=optim, imitator=imitator)

        self._impl = WeightedDiscreteBCImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            beta=self._config.beta,
            class_weights=self._config.class_weights,
            compiled=self.compiled,
            device=self._device,
        )

    def get_action_type(self):
        return ActionSpace.DISCRETE
