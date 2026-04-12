from __future__ import annotations

from typing import Any, Callable, Protocol, cast

from torch import nn


class _ParamWithWeightLoader(Protocol):
    # The codebase attaches a dynamic attribute used by weight-loading code.
    weight_loader: Callable[..., Any]


def set_weight_loader(param: nn.Parameter, loader: Callable[..., Any]) -> None:
    cast(_ParamWithWeightLoader, param).weight_loader = loader
