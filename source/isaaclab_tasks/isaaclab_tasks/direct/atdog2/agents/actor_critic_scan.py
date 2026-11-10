# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Any

from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation


class ActorCriticScan(nn.Module):
    """Actor-Critic with optional scan and priv_obs encoders.

    Supports both:
      - RSL-RL 2.x interface: __init__(obs, obs_groups, num_actions, ...)
      - RSL-RL 1.x interface: __init__(num_actor_obs, num_critic_obs, num_actions, ...)

    - Actor input: prop_obs + (optional) scan_latent
    - Critic input: prop_obs + priv_obs_or_latent + (optional) scan_latent
    """

    is_recurrent: bool = False

    def __init__(
        self,
        *args,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        scan_encoder_dims: list[int] | None = None,
        num_prop_obs: int = 52,
        num_scan_obs: int = 187,
        num_actor_scan_obs: int | None = None,
        num_critic_scan_obs: int | None = None,
        actor_scan_encoder_dims: list[int] | None = None,
        critic_scan_encoder_dims: list[int] | None = None,
        encode_scan_for_critic: bool = True,
        priv_obs_encoder_dims: list[int] | None = None,
        priv_encoder_dims: list[int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticScan.__init__ ignored unexpected arguments: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # Resolve positional arguments
        self.obs_groups = None
        if len(args) >= 3:
            first_arg = args[0]
            if isinstance(first_arg, (int, float)):
                # Legacy RSL-RL signature: (num_actor_obs, num_critic_obs, num_actions)
                self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
                num_actor_obs = int(args[0])
                num_critic_obs = int(args[1])
                num_actions = int(args[2])
            else:
                # Modern RSL-RL 2.x signature: (obs, obs_groups, num_actions)
                obs, obs_groups, num_actions = args[0], args[1], args[2]
                self.obs_groups = obs_groups
                num_actor_obs = 0
                for group in obs_groups["policy"]:
                    num_actor_obs += obs[group].shape[-1]
                num_critic_obs = 0
                for group in obs_groups["critic"]:
                    num_critic_obs += obs[group].shape[-1]
        elif len(args) == 2:
            obs, obs_groups = args[0], args[1]
            self.obs_groups = obs_groups
            num_actions = kwargs.get("num_actions", 12)
            num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
            num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        elif len(args) == 1:
            obs = args[0]
            num_actor_obs = obs["policy"].shape[-1] if hasattr(obs, "__getitem__") else int(obs)
            num_critic_obs = obs["critic"].shape[-1] if hasattr(obs, "__getitem__") and "critic" in obs else num_actor_obs
            num_actions = kwargs.get("num_actions", 12)
        else:
            num_actor_obs = kwargs.get("num_actor_obs", 239)
            num_critic_obs = kwargs.get("num_critic_obs", 268)
            num_actions = kwargs.get("num_actions", 12)

        self.num_actor_obs = int(num_actor_obs)
        self.num_critic_obs = int(num_critic_obs)
        self.num_actions = int(num_actions)
        self.num_prop = int(num_prop_obs)
        base_scan = 0 if num_scan_obs is None else int(num_scan_obs)
        self.num_actor_scan = max(0, base_scan if num_actor_scan_obs is None else int(num_actor_scan_obs))
        self.num_critic_scan = max(0, base_scan if num_critic_scan_obs is None else int(num_critic_scan_obs))
        self.num_priv = self.num_critic_obs - self.num_prop - self.num_critic_scan
        if self.num_priv < 0:
            raise ValueError(
                f"Invalid critic obs split: num_critic_obs={self.num_critic_obs}, "
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
            actor_input_dim = self.num_actor_obs
        if self.num_critic_scan > 0:
            critic_input_dim = self.num_prop + self.priv_latent_dim + self.critic_scan_input_dim
        else:
            critic_input_dim = self.num_prop + self.priv_latent_dim

        # Observation normalizers
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(self.num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(self.num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.actor = MLP(actor_input_dim, [2, self.num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(actor_input_dim, self.num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        self.critic = MLP(critic_input_dim, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            nn.init.zeros_(self.actor[-2].weight[self.num_actions:])
            if self.noise_std_type == "scalar":
                nn.init.constant_(self.actor[-2].bias[self.num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                nn.init.constant_(
                    self.actor[-2].bias[self.num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(self.num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self) -> torch.Tensor | None:
        return self.distribution.mean if self.distribution is not None else None

    @property
    def action_std(self) -> torch.Tensor | None:
        return self.distribution.stddev if self.distribution is not None else None

    @property
    def entropy(self) -> torch.Tensor | None:
        return self.distribution.entropy().sum(dim=-1) if self.distribution is not None else None

    def _make_encoder(self, in_dim: int, dims: list[int], activation: str):
        layers = []
        for i, out_dim in enumerate(dims):
            layers.append(nn.Linear(in_dim, out_dim))
            if i == len(dims) - 1:
                layers.append(nn.Tanh())
            else:
                layers.append(resolve_nn_activation(activation))
            in_dim = out_dim
        return nn.Sequential(*layers)

    def _make_legacy_priv_encoder(self, in_dim: int, dims: list[int], activation: str):
        layers = []
        for out_dim in dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(resolve_nn_activation(activation))
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

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        if any(key.startswith("critic_priv_encoder.") for key in state_dict.keys()):
            remapped = dict(state_dict)
            for key in list(remapped.keys()):
                if key.startswith("critic_priv_encoder."):
                    new_key = "priv_encoder." + key[len("critic_priv_encoder.") :]
                    remapped[new_key] = remapped.pop(key)
            state_dict = remapped
        return super().load_state_dict(state_dict, strict=strict)

    def get_actor_obs(self, obs: Any) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs
        if hasattr(self, "obs_groups") and self.obs_groups is not None and "policy" in self.obs_groups:
            obs_list = [obs[group] for group in self.obs_groups["policy"]]
            return torch.cat(obs_list, dim=-1)
        if hasattr(obs, "__getitem__") and "policy" in obs:
            return obs["policy"]
        if hasattr(obs, "values"):
            return list(obs.values())[0]
        return obs

    def get_critic_obs(self, obs: Any) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs
        if hasattr(self, "obs_groups") and self.obs_groups is not None and "critic" in self.obs_groups:
            obs_list = [obs[group] for group in self.obs_groups["critic"]]
            return torch.cat(obs_list, dim=-1)
        if hasattr(obs, "__getitem__") and "critic" in obs:
            return obs["critic"]
        if hasattr(self, "obs_groups") and self.obs_groups is not None and "policy" in self.obs_groups:
            obs_list = [obs[group] for group in self.obs_groups["policy"]]
            return torch.cat(obs_list, dim=-1)
        if hasattr(obs, "__getitem__") and "policy" in obs:
            return obs["policy"]
        if hasattr(obs, "values"):
            return list(obs.values())[0]
        return obs

    def update_normalization(self, obs: Any) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

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

    def _update_distribution(self, observations: torch.Tensor) -> None:
        obs_enc = self._build_actor_input(observations)
        if self.state_dependent_std:
            mean_and_std = self.actor(obs_enc)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            mean = self.actor(obs_enc)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def update_distribution(self, observations: Any) -> None:
        obs = self.get_actor_obs(observations)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs)

    def act(self, observations: Any, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_actor_obs(observations)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: Any) -> torch.Tensor:
        obs = self.get_actor_obs(observations)
        obs = self.actor_obs_normalizer(obs)
        obs_enc = self._build_actor_input(obs)
        if self.state_dependent_std:
            return self.actor(obs_enc)[..., 0, :]
        return self.actor(obs_enc)

    def evaluate(self, critic_observations: Any, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(critic_observations)
        obs = self.critic_obs_normalizer(obs)
        critic_input = self._build_critic_input(obs)
        return self.critic(critic_input)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass
