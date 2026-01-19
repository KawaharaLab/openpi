#!/usr/bin/env python3
"""
Lightweight script that loads a single sample batch and prints summaries.
Designed to use minimal memory (no model load, CPU-only, single batch).

Usage:
  python scripts/test_sample_batch.py [--split train|val] [--num_samples N]

This attempts to use the project's `create_data_loader` and a config from
`openpi.training.config.cli`. It forces CPU-only behavior and small batch
size to keep memory minimal.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys
import logging
import gc
import argparse
import dataclasses

import torch
import numpy as np

try:
    from openpi.training import data_loader as _data
    from openpi.training import config as _config
except Exception:
    # Best-effort imports; if these fail the error will be printed below.
    raise


def init_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def summarize_tensor(t):
    if isinstance(t, torch.Tensor):
        try:
            arr = t.detach().cpu()
            return (
                f"torch.Tensor shape={tuple(arr.shape)} dtype={arr.dtype} "
                f"mean={float(arr.float().mean()):.4g} min={float(arr.min()):.4g} max={float(arr.max()):.4g}"
            )
        except Exception:
            return f"torch.Tensor shape={tuple(t.shape)} dtype={t.dtype}"
    if isinstance(t, np.ndarray):
        try:
            return (
                f"np.ndarray shape={t.shape} dtype={t.dtype} mean={float(t.mean()):.4g} min={float(t.min()):.4g} max={float(t.max()):.4g}"
            )
        except Exception:
            return f"np.ndarray shape={t.shape} dtype={t.dtype}"
    return f"{type(t)}"


def dump_sample(sample, prefix=""):
    if hasattr(sample, "to_dict"):
        sample = sample.to_dict()
    if isinstance(sample, dict):
        for k, v in sample.items():
            try:
                s = summarize_tensor(v)
            except Exception:
                s = repr(type(v))
            print(f"{prefix}{k}: {s}")
    else:
        try:
            items = dataclasses.asdict(sample)
        except Exception:
            print(f"{prefix}{type(sample)}: {repr(sample)[:500]}")
            return
        for k, v in items.items():
            try:
                s = summarize_tensor(v)
            except Exception:
                s = repr(type(v))
            print(f"{prefix}{k}: {s}")


def main():
    init_logging()
    parser = argparse.ArgumentParser(description="Print one sample batch with minimal memory use")
    parser.add_argument("config", help="config name (choose from available configs)")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()
    # Obtain a config by name from the project's config registry
    try:
        cfg = _config.get_config(args.config)
    except Exception as e:
        logging.error("Failed to load config '%s': %s", args.config, e)
        raise

    # Force minimal settings to reduce memory and worker usage
    try:
        cfg.batch_size = max(1, args.num_samples)
    except Exception:
        pass
    try:
        cfg.num_workers = 0
    except Exception:
        pass

    # Keep PyTorch single-threaded on CPU
    torch.set_num_threads(1)

    logging.info("Creating data loader (only one batch will be consumed)...")
    loader = _data.create_data_loader(cfg, framework="pytorch", shuffle=False, split=args.split)
    it = iter(loader)
    try:
        observation, actions = next(it)
    except StopIteration:
        logging.error("No data returned by loader.")
        return
    print(observation)
    print("=== Observation keys/summary ===")
    dump_sample(observation, prefix="obs.")
    print("=== Actions summary ===")
    if isinstance(actions, (torch.Tensor, np.ndarray)):
        print(summarize_tensor(actions))
    elif isinstance(actions, dict):
        for k, v in actions.items():
            try:
                print(f"act.{k}: {summarize_tensor(v)}")
            except Exception:
                print(f"act.{k}: {type(v)}")
    else:
        print(repr(actions)[:1000])

    # Print a few example values for the first item in batch when available
    try:
        if hasattr(observation, "to_dict"):
            obs_dict = observation.to_dict()
        elif isinstance(observation, dict):
            obs_dict = observation
        else:
            obs_dict = dataclasses.asdict(observation)
        for k, v in obs_dict.items():
            if isinstance(v, torch.Tensor):
                sample0 = v[0].detach().cpu().numpy()
                print(f"example {k} sample0 shape={sample0.shape} dtype={sample0.dtype}")
                flat = np.ravel(sample0)
                n = min(10, flat.size)
                if n > 0:
                    print("  values:", ", ".join(f"{float(x):.4g}" for x in flat[:n]))
    except Exception:
        pass

    # cleanup quickly
    del it
    del loader
    gc.collect()
    logging.info("Done.")


if __name__ == "__main__":
    main()