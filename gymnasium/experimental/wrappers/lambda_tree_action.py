"""Experimental LambdaAction wrappers which supports arbitrarily nested action spaces.

* ``LambdaActionTree`` - Transforms the actions based on a function
* ``ClipActionTree`` - Clips the action within a bounds
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import jumpy as jp
import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, WrapperActType
from gymnasium.spaces import Box, Dict, Discrete, Space, Tuple


class LambdaActionTreeV0(gym.ActionWrapper):
    """A wrapper that provides a function to modify the action passed to :meth:`step`."""

    def __init__(
        self,
        env: gym.Env,
        func: Callable[[WrapperActType], ActType],
        action_space: Space | None,
    ):
        """Initialize LambdaAction.

        Args:
            env: The gymnasium environment
            func: Function to apply to ``step`` ``action``
            action_space: The updated action space of the wrapper given the function.
        """
        super().__init__(env)
        if action_space is not None:
            self.action_space = action_space

        self.func = func

    def apply_function_tree(
        self, action: WrapperActType, space: Space
    ) -> WrapperActType:
        """Recursively traverse the action space to apply the function."""
        if isinstance(space, (Box, Discrete)):
            return self.func(action)
        if isinstance(space, Dict):
            return OrderedDict(
                [
                    (k, self.apply_function_tree(action[k], subspace))
                    for k, subspace in space.spaces.items()
                ]
            )
        if isinstance(space, Tuple):
            return tuple(
                self.apply_function_tree(act, subspace)
                for subspace, act in zip(space.spaces, action)
            )

    def action(self, action: WrapperActType) -> ActType:
        """Apply function to action."""
        return self.apply_function_tree(action, self.action_space)


class ClipActionTreeV0(LambdaActionTreeV0):
    """Clip the continuous action within the valid :class:`Box` observation space bound."""

    def __init__(self, env: gym.Env):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(
            env,
            lambda action, low, high: jp.clip(action, low, high),
            self.apply_new_space(env.action_space),
        )

    def apply_function_tree(
        self, action: WrapperActType, space: Space
    ) -> WrapperActType:
        """Recursively traverse the action space to apply the function."""
        if isinstance(space, Box):
            return self.func(action, space.low, space.high)
        if isinstance(space, Dict):
            return OrderedDict(
                [
                    (k, self.apply_function_tree(action[k], subspace))
                    for k, subspace in space.spaces.items()
                ]
            )
        if isinstance(space, Tuple):
            return tuple(
                self.apply_function_tree(act, subspace)
                for subspace, act in zip(space.spaces, action)
            )
        else:
            return action

    def apply_new_space(self, space: Space) -> Space:
        """Recursively build the action space."""
        if isinstance(space, Box):
            return Box(-np.inf, np.inf, shape=space.shape, dtype=space.dtype)
        if isinstance(space, Dict):
            return Dict(
                [
                    (k, self.apply_new_space(subspace))
                    for k, subspace in space.spaces.items()
                ]
            )
        if isinstance(space, Tuple):
            return Tuple(self.apply_new_space(subspace) for subspace in space.spaces)
        if isinstance(space, Discrete):
            return space
        else:
            raise
