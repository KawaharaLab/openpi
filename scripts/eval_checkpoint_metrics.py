#!/usr/bin/env python3
"""Evaluate comparable metrics from existing checkpoints on a single GPU/process.

This script intentionally avoids distributed checkpoint restore. It loads params
from `<checkpoint_dir>/<step>/params` and evaluates by running `sample_actions`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
from typing import Any

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.models.model_ft as _model_ft
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def init_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _find_latest_step(run_dir: pathlib.Path) -> int:
    steps = sorted(int(p.name) for p in run_dir.iterdir() if p.is_dir() and p.name.isdigit())
    if not steps:
        raise FileNotFoundError(f"No step directories found under: {run_dir}")
    return steps[-1]


def _find_action_norm_stats(norm_stats: dict[str, Any] | None) -> Any | None:
    if norm_stats is None:
        return None
    flat = traverse_util.flatten_dict(norm_stats, sep="/")
    if "actions" in flat:
        return flat["actions"]
    for key, value in flat.items():
        if key.endswith("/actions"):
            return value
    return None


def _align_stats_to_dim(stats_vec: np.ndarray, target_dim: int, *, pad_value: float) -> np.ndarray:
    arr = np.asarray(stats_vec, dtype=np.float32).reshape(-1)
    if arr.shape[0] < target_dim:
        arr = np.pad(arr, (0, target_dim - arr.shape[0]), constant_values=pad_value)
    elif arr.shape[0] > target_dim:
        arr = arr[:target_dim]
    return arr


def _unnormalize_actions(
    actions: jax.Array,
    *,
    use_quantile: bool,
    mean: jax.Array | None,
    std: jax.Array | None,
    q01: jax.Array | None,
    q99: jax.Array | None,
) -> jax.Array:
    if use_quantile:
        assert q01 is not None and q99 is not None
        return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    assert mean is not None and std is not None
    return actions * (std + 1e-6) + mean


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description="Evaluate comparable metrics from a checkpoint.")
    parser.add_argument("config_name", type=str, help="Config name, e.g. pi05_ur3_robotiq")
    parser.add_argument("--exp-name", type=str, required=True, help="Run name under checkpoints/<config>/")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    parser.add_argument("--split", type=str, choices=("train", "val", "all"), default="val")
    parser.add_argument("--num-batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--compare-action-dim", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    # Keep eval single process / single visible device.
    logging.info("JAX devices: %s", jax.devices())

    config = _config.get_config(args.config_name)
    config = dataclasses.replace(config, exp_name=args.exp_name)
    if args.batch_size is not None:
        config = dataclasses.replace(config, batch_size=args.batch_size)
    if args.num_workers is not None:
        config = dataclasses.replace(config, num_workers=args.num_workers)

    run_dir = config.checkpoint_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Checkpoint run dir does not exist: {run_dir}")
    step = args.step if args.step is not None else _find_latest_step(run_dir)
    params_path = run_dir / str(step) / "params"
    if not params_path.exists():
        raise FileNotFoundError(f"Params checkpoint not found: {params_path}")

    # Load dataset in local mode (no mesh/sharding).
    data_loader = _data_loader.create_data_loader(
        config,
        shuffle=False,
        num_batches=args.num_batches,
        split=args.split,
    )
    data_config = data_loader.data_config()

    # Build model skeleton, then load checkpoint params into it.
    init_rng = jax.random.key(args.seed)
    model = config.model.create(init_rng)
    restored_params = _model.restore_params(params_path, restore_type=jax.Array, sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]))
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(restored_params)
    model = nnx.merge(graphdef, state)
    model.eval()
    sample_actions_fn = nnx_utils.module_jit(model.sample_actions)

    action_stats = _find_action_norm_stats(data_config.norm_stats)
    compare_action_dim = min(args.compare_action_dim, config.model.action_dim)
    if action_stats is not None and hasattr(action_stats, "mean"):
        compare_action_dim = min(compare_action_dim, int(np.asarray(action_stats.mean).shape[-1]))

    use_quantile_for_eval = bool(data_config.use_quantile_norm and action_stats is not None)
    mean_eval = std_eval = q01_eval = q99_eval = None
    if action_stats is not None:
        mean_eval = jnp.asarray(
            _align_stats_to_dim(action_stats.mean, compare_action_dim, pad_value=0.0), dtype=jnp.float32
        )
        std_eval = jnp.asarray(
            _align_stats_to_dim(action_stats.std, compare_action_dim, pad_value=1.0), dtype=jnp.float32
        )
        if action_stats.q01 is not None and action_stats.q99 is not None:
            q01_eval = jnp.asarray(
                _align_stats_to_dim(action_stats.q01, compare_action_dim, pad_value=0.0), dtype=jnp.float32
            )
            q99_eval = jnp.asarray(
                _align_stats_to_dim(action_stats.q99, compare_action_dim, pad_value=0.0), dtype=jnp.float32
            )
        else:
            use_quantile_for_eval = False

    per_batch: list[dict[str, float]] = []
    base_rng = jax.random.key(args.seed + 10_000)
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= args.num_batches:
            break
        observation, actions = batch
        step_rng = jax.random.fold_in(base_rng, batch_idx)
        sampled_actions = sample_actions_fn(step_rng, observation, num_steps=args.sample_steps)

        pred = sampled_actions[..., :compare_action_dim]
        target = actions[..., :compare_action_dim]
        metrics = {
            "eval_action_mse_norm": float(np.asarray(jnp.mean(jnp.square(pred - target)))),
        }

        if mean_eval is not None and std_eval is not None:
            pred_raw = _unnormalize_actions(
                pred,
                use_quantile=use_quantile_for_eval,
                mean=mean_eval,
                std=std_eval,
                q01=q01_eval,
                q99=q99_eval,
            )
            target_raw = _unnormalize_actions(
                target,
                use_quantile=use_quantile_for_eval,
                mean=mean_eval,
                std=std_eval,
                q01=q01_eval,
                q99=q99_eval,
            )
            metrics["eval_action_mse_raw"] = float(np.asarray(jnp.mean(jnp.square(pred_raw - target_raw))))

        per_batch.append(metrics)
        if (batch_idx + 1) % 10 == 0:
            logging.info("Evaluated %s/%s batches", batch_idx + 1, args.num_batches)

    if not per_batch:
        raise RuntimeError("No batches were evaluated.")

    keys = sorted(per_batch[0].keys())
    summary: dict[str, float] = {}
    for key in keys:
        vals = np.array([row[key] for row in per_batch], dtype=np.float64)
        summary[f"{key}_mean"] = float(vals.mean())
        summary[f"{key}_std"] = float(vals.std())

    output = {
        "config_name": args.config_name,
        "exp_name": args.exp_name,
        "checkpoint_step": step,
        "split": args.split,
        "num_batches": len(per_batch),
        "compare_action_dim": compare_action_dim,
        "use_quantile_norm": use_quantile_for_eval,
        "metrics": summary,
    }
    print(json.dumps(output, indent=2, ensure_ascii=True))

    if args.output_json:
        out_path = pathlib.Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        logging.info("Wrote summary to %s", out_path)


if __name__ == "__main__":
    main()
