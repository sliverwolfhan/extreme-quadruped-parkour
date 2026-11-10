# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ActorCriticScan(nn.Module):
    """Actor-Critic with optional scan and priv_obs encoders.

    - Actor input: prop_obs + (optional) scan_latent
    - Critic input: prop_obs + priv_obs_or_latent + (optional) scan_latent
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims=None,
        critic_hidden_dims=None,
        scan_encoder_dims=None,
        num_prop_obs: int = 52,
        num_scan_obs: int = 187,
        num_actor_scan_obs: int | None = None,
        num_critic_scan_obs: int | None = None,
        actor_scan_encoder_dims=None,
        critic_scan_encoder_dims=None,
        encode_scan_for_critic: bool = True,
        priv_obs_encoder_dims=None,
        priv_encoder_dims=None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticScan.__init__ ignored unexpected arguments: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        # raw observation dims passed from runner/env
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_prop = num_prop_obs
        base_scan = 0 if num_scan_obs is None else num_scan_obs
        self.num_actor_scan = max(0, base_scan if num_actor_scan_obs is None else num_actor_scan_obs)
        self.num_critic_scan = max(0, base_scan if num_critic_scan_obs is None else num_critic_scan_obs)
        self.num_priv = num_critic_obs - self.num_prop - self.num_critic_scan
        if self.num_priv < 0:
            raise ValueError(
                f"Invalid critic obs split: num_critic_obs={num_critic_obs}, "
                f"num_prop={self.num_prop}, num_scan={self.num_critic_scan}"
            )

        # scan encoders (actor/critic can be configured independently)
        self.encode_scan_for_critic = encode_scan_for_critic
        actor_scan_encoder_dims = scan_encoder_dims if actor_scan_encoder_dims is None else actor_scan_encoder_dims
        critic_scan_encoder_dims = scan_encoder_dims if critic_scan_encoder_dims is None else critic_scan_encoder_dims

        self.actor_scan_latent_dim = self.num_actor_scan
        if self.num_actor_scan > 0 and actor_scan_encoder_dims is not None and len(actor_scan_encoder_dims) > 0:
            self.actor_scan_encoder = self._make_encoder(self.num_actor_scan, actor_scan_encoder_dims, activation)
            self.actor_scan_latent_dim = actor_scan_encoder_dims[-1]
        else:
            self.actor_scan_encoder = None

        self.critic_scan_latent_dim = self.num_critic_scan
        if self.num_critic_scan > 0 and critic_scan_encoder_dims is not None and len(critic_scan_encoder_dims) > 0:
            self.critic_scan_encoder = self._make_encoder(self.num_critic_scan, critic_scan_encoder_dims, activation)
            self.critic_scan_latent_dim = critic_scan_encoder_dims[-1]
        else:
            self.critic_scan_encoder = None
        self.critic_scan_input_dim = (
            self.num_critic_scan if not self.encode_scan_for_critic else self.critic_scan_latent_dim
        )

        # priv_obs encoder (critic only)
        self.priv_latent_dim = self.num_priv
        if self.num_priv > 0 and priv_encoder_dims is not None and len(priv_encoder_dims) > 0:
            self.priv_encoder = self._make_legacy_priv_encoder(self.num_priv, priv_encoder_dims, activation)
            self.priv_latent_dim = priv_encoder_dims[-1]
        elif self.num_priv > 0 and priv_obs_encoder_dims is not None and len(priv_obs_encoder_dims) > 0:
            self.priv_encoder = self._make_encoder(self.num_priv, priv_obs_encoder_dims, activation)
            self.priv_latent_dim = priv_obs_encoder_dims[-1]
        else:
            self.priv_encoder = None

        # default hidden dims
        actor_hidden_dims = actor_hidden_dims or [256, 256, 256]
        critic_hidden_dims = critic_hidden_dims or [256, 256, 256]

        # actor/critic input dims after optional encoding
        if self.num_actor_scan > 0:
            actor_input_dim = self.num_prop + self.actor_scan_latent_dim
        else:
            actor_input_dim = num_actor_obs
        if self.num_critic_scan > 0:
            critic_input_dim = self.num_prop + self.priv_latent_dim + self.critic_scan_input_dim
        else:
            critic_input_dim = self.num_prop + self.priv_latent_dim

        # Actor MLP
        actor_layers = [nn.Linear(actor_input_dim, actor_hidden_dims[0]), activation]
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Critic MLP
        critic_layers = [nn.Linear(critic_input_dim, critic_hidden_dims[0]), activation]
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self):
        return self.distribution.mean if self.distribution is not None else None

    @property
    def action_std(self):
        return self.distribution.stddev if self.distribution is not None else None

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1) if self.distribution is not None else None

    def _make_encoder(self, in_dim, dims, activation):
        layers = []
        for i, out_dim in enumerate(dims):
            layers.append(nn.Linear(in_dim, out_dim))
            if i == len(dims) - 1:
                layers.append(nn.Tanh())
            else:
                layers.append(activation)
            in_dim = out_dim
        return nn.Sequential(*layers)

    def _make_legacy_priv_encoder(self, in_dim, dims, activation):
        layers = []
        for out_dim in dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation)
            in_dim = out_dim
        return nn.Sequential(*layers)

    def _encode_actor_scan(self, scan: torch.Tensor) -> torch.Tensor:
        if self.actor_scan_encoder is None:
            return scan
        return self.actor_scan_encoder(scan)

    def _encode_critic_scan(self, scan: torch.Tensor) -> torch.Tensor:
        if self.critic_scan_encoder is None:
            return scan
        return self.critic_scan_encoder(scan)

    def _encode_priv_obs(self, priv: torch.Tensor) -> torch.Tensor:
        if self.priv_encoder is None:
            return priv
        return self.priv_encoder(priv)

    def load_state_dict(self, state_dict, strict: bool = True):  # noqa: D102
        if any(key.startswith("critic_priv_encoder.") for key in state_dict.keys()):
            remapped = dict(state_dict)
            for key in list(remapped.keys()):
                if key.startswith("critic_priv_encoder."):
                    new_key = "priv_encoder." + key[len("critic_priv_encoder.") :]
                    remapped[new_key] = remapped.pop(key)
            state_dict = remapped
        return super().load_state_dict(state_dict, strict=strict)

    def _build_actor_input(self, observations: torch.Tensor) -> torch.Tensor:
        if self.num_actor_scan <= 0:
            return observations
        obs_prop = observations[:, : self.num_prop]
        obs_scan = observations[:, self.num_prop : self.num_prop + self.num_actor_scan]  # prop || scan
        z_scan = self._encode_actor_scan(obs_scan)
        return torch.cat([obs_prop, z_scan], dim=-1)

    def _build_critic_input(self, critic_observations: torch.Tensor) -> torch.Tensor:
        if self.num_priv <= 0 and self.num_critic_scan <= 0:
            return critic_observations
        obs_prop = critic_observations[:, : self.num_prop]
        offset = self.num_prop
        obs_priv = None
        if self.num_priv > 0:
            obs_priv = critic_observations[:, offset : offset + self.num_priv]
            obs_priv = self._encode_priv_obs(obs_priv)
            offset += self.num_priv
        obs_scan = None
        if self.num_critic_scan > 0:
            obs_scan = critic_observations[:, offset : offset + self.num_critic_scan]
            if self.encode_scan_for_critic:
                obs_scan = self._encode_critic_scan(obs_scan)
        parts = [obs_prop]
        if obs_priv is not None:
            parts.append(obs_priv)
        if obs_scan is not None:
            parts.append(obs_scan)
        return torch.cat(parts, dim=-1)

    def update_distribution(self, observations):
        obs_enc = self._build_actor_input(observations)
        mean = self.actor(obs_enc)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        obs_enc = self._build_actor_input(observations)
        actions_mean = self.actor(obs_enc)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        critic_input = self._build_critic_input(critic_observations)
        value = self.critic(critic_input)
        return value

    def reset(self, dones=None):
        # stateless actor-critic; nothing to reset
        pass
