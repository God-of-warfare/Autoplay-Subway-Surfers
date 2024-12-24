import pytorch_lightning as pl
import torch.nn as nn
from typing import Optional, Union, List
import math


class WeightInitCallback(pl.Callback):
    """
    PyTorch Lightning callback for initializing weights in attention and MLP blocks.
    Supports various initialization schemes and layer-specific configurations.
    """

    def __init__(
            self,
            attention_init: str = "xavier_uniform",
            mlp_init: str = "kaiming_uniform",
            attention_gain: float = 1.0,
            mlp_gain: float = math.sqrt(2.0),
            bias_init: str = "zeros",
            layer_norm_init: Optional[str] = "ones",
            excluded_params: Optional[List[str]] = None
    ):
        """
        Initialize the callback with specific initialization schemes.

        Args:
            attention_init: Initialization scheme for attention layers
            mlp_init: Initialization scheme for MLP layers
            attention_gain: Gain factor for attention layer initialization
            mlp_gain: Gain factor for MLP layer initialization
            bias_init: Initialization scheme for bias terms
            layer_norm_init: Initialization scheme for LayerNorm layers
            excluded_params: List of parameter names to exclude from initialization
        """
        super().__init__()
        self.attention_init = attention_init
        self.mlp_init = mlp_init
        self.attention_gain = attention_gain
        self.mlp_gain = mlp_gain
        self.bias_init = bias_init
        self.layer_norm_init = layer_norm_init
        self.excluded_params = excluded_params or []

        self.init_functions = {
            "xavier_uniform": nn.init.xavier_uniform_,
            "xavier_normal": nn.init.xavier_normal_,
            "kaiming_uniform": nn.init.kaiming_uniform_,
            "kaiming_normal": nn.init.kaiming_normal_,
            "orthogonal": nn.init.orthogonal_,
            "zeros": nn.init.zeros_,
            "ones": nn.init.ones_
        }

    def _is_attention_layer(self, name: str) -> bool:
        """Check if the layer name indicates an attention component."""
        attention_keywords = ["attention", "query", "key", "value", "qkv", "attn"]
        return any(keyword in name.lower() for keyword in attention_keywords)

    def _is_mlp_layer(self, name: str) -> bool:
        """Check if the layer name indicates an MLP component."""
        mlp_keywords = ["mlp", "ffn", "feed_forward", "fc", "linear"]
        return any(keyword in name.lower() for keyword in mlp_keywords)

    def _init_weights(self, module: nn.Module, name: str) -> None:
        """Initialize weights for a single module."""
        if any(excluded in name for excluded in self.excluded_params):
            return

        if isinstance(module, nn.Linear):
            if self._is_attention_layer(name):
                self.init_functions[self.attention_init](
                    module.weight, gain=self.attention_gain
                )
            else:  # MLP layers
                self.init_functions[self.mlp_init](
                    module.weight, gain=self.mlp_gain
                )

            if module.bias is not None:
                self.init_functions[self.bias_init](module.bias)

        elif isinstance(module, nn.LayerNorm) and self.layer_norm_init:
            if module.elementwise_affine:
                self.init_functions[self.layer_norm_init](module.weight)
                self.init_functions[self.bias_init](module.bias)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize weights when training starts."""
        for name, module in pl_module.named_modules():
            self._init_weights(module, name)

    def _log_initialization(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log initialization statistics for monitoring."""
        stats = {
            "weight_norm_mean": 0.0,
            "weight_norm_std": 0.0,
            "count": 0
        }

        for name, param in pl_module.named_parameters():
            if "weight" in name and not any(excluded in name for excluded in self.excluded_params):
                norm = param.norm().item()
                stats["weight_norm_mean"] += norm
                stats["weight_norm_std"] += norm ** 2
                stats["count"] += 1

        if stats["count"] > 0:
            stats["weight_norm_mean"] /= stats["count"]
            stats["weight_norm_std"] = math.sqrt(
                stats["weight_norm_std"] / stats["count"] - stats["weight_norm_mean"] ** 2
            )

            trainer.logger.log_metrics({
                "weight_norm_mean": stats["weight_norm_mean"],
                "weight_norm_std": stats["weight_norm_std"]
            })