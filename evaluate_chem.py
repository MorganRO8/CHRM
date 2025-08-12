from __future__ import annotations

import os
from typing import List, Optional, Dict

import torch
import torch.distributed as dist
import yaml
import pydantic
from omegaconf import OmegaConf

from pretrain import PretrainConfig, init_train_state, evaluate, create_dataloader
from utils.chem_io import save_xyz, save_ase_trajectory


class EvalChemConfig(pydantic.BaseModel):
    checkpoint: str
    save_outputs: List[str] = ["atom_types", "positions", "energy", "forces"]
    xyz_path: Optional[str] = None
    traj_path: Optional[str] = None


def _gather_predictions(
    config: PretrainConfig, step: int, world_size: int
) -> Dict[str, torch.Tensor]:
    preds: Dict[str, List[torch.Tensor]] = {}
    for rank in range(world_size):
        preds_file = os.path.join(
            config.checkpoint_path, f"step_{step}_all_preds.{rank}"
        )
        if not os.path.exists(preds_file):
            continue
        loaded = torch.load(preds_file, map_location="cpu")
        for k, v in loaded.items():
            preds.setdefault(k, []).append(v)
    return {k: torch.cat(v, dim=0) for k, v in preds.items()}


def launch() -> None:
    eval_cfg = EvalChemConfig(  # type: ignore
        **OmegaConf.to_container(OmegaConf.from_cli())
    )

    rank = 0
    world_size = 1

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Load training configuration
    with open(
        os.path.join(os.path.dirname(eval_cfg.checkpoint), "all_config.yaml"), "r"
    ) as f:
        config = PretrainConfig(**yaml.safe_load(f))
        config.eval_save_outputs = eval_cfg.save_outputs
        config.checkpoint_path = os.path.dirname(eval_cfg.checkpoint)

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=rank,
        world_size=world_size,
    )
    eval_loader, eval_metadata = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=rank,
        world_size=world_size,
    )

    train_state = init_train_state(config, train_metadata, world_size=world_size)
    try:
        train_state.model.load_state_dict(
            torch.load(eval_cfg.checkpoint, map_location="cuda"), assign=True
        )
    except Exception:
        train_state.model.load_state_dict(
            {
                k.removeprefix("_orig_mod."): v
                for k, v in torch.load(eval_cfg.checkpoint, map_location="cuda").items()
            },
            assign=True,
        )

    train_state.step = 0
    ckpt_filename = os.path.basename(eval_cfg.checkpoint)
    if ckpt_filename.startswith("step_"):
        train_state.step = int(ckpt_filename.removeprefix("step_"))

    train_state.model.eval()
    metrics = evaluate(
        config,
        train_state,
        eval_loader,
        eval_metadata,
        rank=rank,
        world_size=world_size,
    )

    if metrics is not None and rank == 0:
        print(metrics)

    if rank == 0 and (eval_cfg.xyz_path or eval_cfg.traj_path):
        preds = _gather_predictions(config, train_state.step, world_size)
        if "atom_types" in preds and "positions" in preds:
            if eval_cfg.xyz_path:
                save_xyz(preds, eval_metadata, eval_cfg.xyz_path)
            if eval_cfg.traj_path:
                save_ase_trajectory(preds, eval_metadata, eval_cfg.traj_path)


if __name__ == "__main__":
    launch()
