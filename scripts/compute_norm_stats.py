"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""
import pathlib

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None and data_config.local_repo_path is None:
        raise ValueError("Data config must have a repo_id or local_repo_path")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # If the dataset already contains pre-computed action horizons (action_sequence_keys empty),
    # avoid constructing additional action chunks by forcing a horizon of 1 for norm stats.
    effective_action_horizon = 1 if len(data_config.action_sequence_keys) == 0 else config.model.action_horizon

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, effective_action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, effective_action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    paths = ["state", "actions", "force_torques/left_ft", "force_torques/right_ft"]
    stats: dict[str, normalize.RunningStats] = {}

    def _maybe_get(batch: dict, path: str):
        cur = batch
        for part in path.split("/"):
            if part not in cur:
                return None
            cur = cur[part]
        return cur

    def _prepare_for_stats(value: object, path: str) -> np.ndarray:
        arr = np.asarray(value)
        if not path.startswith("force_torques"):
            return arr

        # Ensure the 6-axis dimension sits in the last position so RunningStats computes per-axis stats.
        if arr.shape[-1] == 6:
            return arr

        # Try to find a dimension of size 6 and move it to the last axis.
        axis_candidates = [i for i, dim in enumerate(arr.shape) if dim == 6]
        if axis_candidates:
            arr = np.moveaxis(arr, axis_candidates[0], -1)
            return arr

        raise ValueError(
            f"Expected a 6-axis force/torque tensor for {path}, got shape {arr.shape}. "
            "Please ensure FT data is shaped (6, horizon) or has a dimension of size 6."
        )

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for path in paths:
            if (val := _maybe_get(batch, path)) is None:
                continue
            if path not in stats:
                stats[path] = normalize.RunningStats()
            stats[path].update(_prepare_for_stats(val, path))

    norm_stats = {path: rs.get_statistics() for path, rs in stats.items()}

    asset_name = (
        data_config.asset_id
        or data_config.repo_id
        or (pathlib.Path(data_config.local_repo_path).name if data_config.local_repo_path else None)
    )
    if asset_name is None:
        raise ValueError("Could not determine asset directory name (repo_id/asset_id/local_repo_path missing).")

    output_path = config.assets_dirs / asset_name
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
