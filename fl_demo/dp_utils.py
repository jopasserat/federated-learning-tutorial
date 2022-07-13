from typing import Dict, Tuple

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import numpy as np
from torch import nn, optim
from torch.utils import data


def configure_dp_training(
    dp_config: Dict,
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: data.DataLoader,
) -> Tuple[nn.Module, optim.Optimizer, data.DataLoader, PrivacyEngine]:
    privacy_engine = PrivacyEngine(
        secure_mode=dp_config["secure_rng"],
    )

    if dp_config["clip_per_layer"]:
        clipping = "per_layer"
        # Each layer has the same clipping threshold. The total grad norm is still bounded by `args.max_per_sample_grad_norm`.
        n_layers = len([(n, p) for n, p in model.named_parameters() if p.requires_grad])
        max_grad_norm = [
            dp_config["max_per_sample_grad_norm"] / np.sqrt(n_layers)
        ] * n_layers
    else:
        clipping = "flat"
        max_grad_norm = dp_config["max_per_sample_grad_norm"]

    model.train()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=dp_config["noise_multiplier"],
        max_grad_norm=max_grad_norm,
        clipping=clipping,
    )

    return model, optimizer, train_loader, privacy_engine


def fix_model_layers(model):
    """
    Replace BatchNorm with GroupNorm if present
    THIS MUST BE CALLED BEFORE PASSING THE MODEL'S PARAMETERS TO ModuleValidator.fix()
    """
    return ModuleValidator.fix(model)
