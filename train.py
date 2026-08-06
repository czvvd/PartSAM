"""Train PartSAM with local GLB meshes."""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset


def build_dataset(cfg: DictConfig) -> Dataset:
    """Build the native dataset and, optionally, append accepted pseudo labels."""

    from PartSAM.data.dataloader import MeshDataset

    common = {
        "num_points": int(cfg.data.num_points),
        "min_part_ratio": float(cfg.data.min_part_ratio),
        "max_part_ratio": float(cfg.data.max_part_ratio),
    }
    native = MeshDataset(
        cfg.data.root,
        seed=int(cfg.seed),
        **common,
    )
    if cfg.data.pseudo_root is None:
        return native

    from PartSAM.data.dataloader import PseudoLabelDataset

    pseudo = PseudoLabelDataset(
        cfg.data.root,
        cfg.data.pseudo_root,
        seed=int(cfg.seed),
        **common,
    )
    return ConcatDataset([native, pseudo])


def build_dataloader(cfg: DictConfig, dataset: Dataset) -> DataLoader:
    from PartSAM.data.collate import PartMaskCollator

    workers = int(cfg.train.num_workers)
    return DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(int(cfg.seed)),
        collate_fn=PartMaskCollator(
            training=True,
            max_targets=int(cfg.data.max_targets),
        ),
    )


def initialize_from_partfield(cfg: DictConfig, model: torch.nn.Module) -> None:
    """Initialize both point-encoder branches for a new run."""

    checkpoint = Path(str(cfg.initialization.partfield_checkpoint))
    encoder = getattr(model, "pc_encoder", None)
    loader = getattr(encoder, "load_partfield_checkpoint", None)
    if not callable(loader):
        raise TypeError("model.pc_encoder does not support PartField initialization")
    report = loader(checkpoint)
    print(
        "Initialized both PartField branches from "
        f"{checkpoint} ({len(report.matched)} tensors)."
    )


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="train_objaverse_demo",
)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    set_seed(int(cfg.seed))

    dataset = build_dataset(cfg)
    model = hydra.utils.instantiate(cfg.model)
    criterion = hydra.utils.instantiate(cfg.loss)
    dataloader = build_dataloader(cfg, dataset)

    if cfg.train.resume_from is None:
        initialize_from_partfield(cfg, model)

    from PartSAM.training import train_steps

    result = train_steps(cfg, model, dataloader, criterion)
    print(f"Training finished at step {result.global_step}.")


if __name__ == "__main__":
    main()
