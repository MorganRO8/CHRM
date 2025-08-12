from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


def energy_mse_loss(pred, target):
    """Mean squared error loss for energies."""
    return F.mse_loss(pred, target, reduction="sum")


def force_mse_loss(energy: torch.Tensor, positions: torch.Tensor, target_forces: torch.Tensor):
    """Force loss computed from energy gradient w.r.t. positions."""
    forces = -torch.autograd.grad(energy.sum(), positions, create_graph=True)[0]
    loss = F.mse_loss(forces, target_forces, reduction="sum")
    return loss, forces


class ACTLossHead(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        loss_type: str = "energy_mse_loss",
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.energy_weight = energy_weight
        self.force_weight = force_weight

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:

        batch = model_kwargs["batch"]
        requires_forces = "forces" in batch
        if requires_forces:
            batch["positions"] = batch["positions"].requires_grad_(True)

        new_carry, outputs = self.model(**model_kwargs)

        # Energy loss
        energy_pred = outputs["energy"].sum(-1)
        energy_target = new_carry.current_data["energy"]
        energy_loss = self.loss_fn(energy_pred, energy_target)
        outputs["energy"] = energy_pred

        metrics: Dict[str, torch.Tensor] = {
            "count": torch.tensor(batch["atom_types"].shape[0], device=energy_loss.device),
            "energy_loss": energy_loss.detach(),
        }

        total_loss = self.energy_weight * energy_loss

        # Force loss if available
        if requires_forces:
            force_loss, forces = force_mse_loss(
                energy_pred, batch["positions"], new_carry.current_data["forces"]
            )
            total_loss = total_loss + self.force_weight * force_loss
            metrics["force_loss"] = force_loss.detach()
            outputs["forces"] = forces

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
