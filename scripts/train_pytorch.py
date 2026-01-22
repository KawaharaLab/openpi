"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import logging
import os
import platform
import socket
import pickle
import shutil
import time
import pathlib

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
from openpi.models_pytorch import lora_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    run_dir: pathlib.Path | None,
    run_name: str | None,
    base_dir: pathlib.Path,
    overwrite: bool,
    enabled: bool = True,
):
    """Initialize wandb logging and return the resolved run directory and name."""
    if not enabled:
        wandb.init(mode="disabled")
        resolved_name = run_name or wandb.run.name or "disabled"
        resolved_dir = run_dir or (base_dir / resolved_name)
        return resolved_dir, resolved_name

    if resuming:
        assert run_dir is not None, "run_dir must be set when resuming"
        if not run_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory {run_dir} does not exist.")
        run_id = (run_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.name)
        resolved_name = run_name or wandb.run.name
        return run_dir, resolved_name
    else:
        wandb.init(
            config=dataclasses.asdict(config),
            project=config.name,
        )
        resolved_name = run_name or wandb.run.name or "offline"
        resolved_dir = run_dir or (base_dir / resolved_name)
        if resolved_dir.exists():
            if overwrite:
                shutil.rmtree(resolved_dir)
            else:
                raise FileExistsError(f"Checkpoint directory {resolved_dir} already exists; use --overwrite to replace.")
        resolved_dir.mkdir(parents=True, exist_ok=True)
        (resolved_dir / "wandb_run_name.txt").write_text(resolved_name)
        (resolved_dir / "wandb_id.txt").write_text(wandb.run.id)
        return resolved_dir, resolved_name


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True, split="train")
    val_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False, split="val")
    return data_loader, val_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, checkpoint_dir: pathlib.Path, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        def _make_pickleable(obj):
            """Convert objects (e.g., lambdas in config) into pickle-friendly values."""
            if isinstance(obj, dict):
                return {k: _make_pickleable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(_make_pickleable(v) for v in obj)
            if isinstance(obj, set):
                return {_make_pickleable(v) for v in obj}
            try:
                pickle.dumps(obj)
                return obj
            except Exception:
                return repr(obj)

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": _make_pickleable(dataclasses.asdict(config)),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def log_gpu_environment(use_ddp: bool, local_rank: int, device: torch.device):
    """Log visibility and device info to confirm GPUs are being used as expected."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    world_info = ""
    if dist.is_initialized():
        world_info = f" | rank={dist.get_rank()} world_size={dist.get_world_size()}"

    if torch.cuda.is_available() and device.type == "cuda":
        try:
            name = torch.cuda.get_device_name(device)
        except Exception:
            name = "<unknown>"
        try:
            props = torch.cuda.get_device_properties(device)
            total_mem_gb = getattr(props, "total_memory", 0) / 1e9
        except Exception:
            total_mem_gb = 0.0

        all_devices = []
        for idx in range(torch.cuda.device_count()):
            try:
                dname = torch.cuda.get_device_name(idx)
            except Exception:
                dname = "<unknown>"
            all_devices.append(f"{idx}:{dname}")

        logging.info(
            "GPU env: use_ddp=%s local_rank=%s device=%s cuda_available=%s cuda_visible_devices=%s cuda_device_count=%s current_device=%s device_name=%s total_mem_gb=%.2f%s all_devices=[%s]",
            use_ddp,
            local_rank,
            device,
            torch.cuda.is_available(),
            visible,
            torch.cuda.device_count(),
            torch.cuda.current_device(),
            name,
            total_mem_gb,
            world_info,
            ", ".join(all_devices),
        )
    else:
        logging.info(
            "GPU env: use_ddp=%s local_rank=%s device=%s cuda_available=%s cuda_visible_devices=%s cuda_device_count=%s%s",
            use_ddp,
            local_rank,
            device,
            torch.cuda.is_available(),
            visible,
            torch.cuda.device_count(),
            world_info,
        )


def log_cpu_environment():
    """Log CPU availability and affinity for the current process."""
    total_cpus = os.cpu_count()
    affinity = None
    try:
        affinity = len(os.sched_getaffinity(0))
    except Exception:
        affinity = None
    omp_threads = os.environ.get("OMP_NUM_THREADS", "<unset>")
    logging.info(
        "CPU env: pid=%s total_cpus=%s affinity_cpus=%s OMP_NUM_THREADS=%s",
        os.getpid(),
        total_cpus,
        affinity,
        omp_threads,
    )


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Log GPU visibility and device mapping so multi-GPU runs can be verified in logs.
    log_gpu_environment(use_ddp, local_rank, device)
    log_cpu_environment()

    # Resolve checkpoint root and run directory (use wandb.run.name to avoid accidental overwrite).
    base_checkpoint_root = pathlib.Path(config.checkpoint_base_dir) / config.name
    run_checkpoint_dir: pathlib.Path | None = None
    run_name: str | None = None
    resuming = False

    if config.resume:
        if not config.exp_name:
            raise ValueError("--exp_name must be set when resuming to identify the run directory.")
        run_name = config.exp_name
        run_checkpoint_dir = base_checkpoint_root / run_name
        if run_checkpoint_dir.exists():
            latest_step = get_latest_checkpoint_step(run_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from checkpoint directory: {run_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {run_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Checkpoint directory {run_checkpoint_dir} does not exist for resume")

    # Initialize wandb on the main rank to obtain run_name; broadcast to others.
    if is_main:
        base_checkpoint_root.mkdir(parents=True, exist_ok=True)
        run_checkpoint_dir, run_name = init_wandb(
            config,
            resuming=resuming,
            run_dir=run_checkpoint_dir,
            run_name=run_name,
            base_dir=base_checkpoint_root,
            overwrite=config.overwrite,
            enabled=config.wandb_enabled,
        )

    if use_ddp:
        # Share run_name from rank 0
        name_list = [run_name]
        dist.broadcast_object_list(name_list, src=0)
        run_name = name_list[0]
        run_checkpoint_dir = base_checkpoint_root / run_name

    # For single-process or after broadcast, ensure checkpoint dir exists (main already created when logging enabled).
    if not resuming and run_checkpoint_dir is not None and not run_checkpoint_dir.exists():
        run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Using checkpoint directory: {run_checkpoint_dir}")

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, val_loader, data_config = build_datasets(config)
    val_iter = iter(val_loader)

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]

        def _to_hwc_uint8(img: torch.Tensor) -> np.ndarray:
            """Convert image tensor to HWC uint8 for logging."""
            if img.ndim != 3:
                raise ValueError(f"Expected 3D image tensor, got shape {img.shape}")

            # If channel-first, move channels to the end; otherwise assume already HWC.
            if img.shape[0] in (1, 3) and img.shape[0] <= img.shape[-1]:
                img = img.permute(1, 2, 0)
            np_img = img.detach().cpu().numpy()
            if np_img.dtype != np.uint8:
                print(f"{np_img.dtype}, {np_img.min()}, {np_img.max()}")
                # Handle floats in [0,1] or [-1,1]; clip to [0,255]
                np_img = np_img.astype(np.float32)
                if np_img.min() >= -1.0 and np_img.max() <= 1.0:
                    np_img = (np_img + 1.0) / 2.0
                np_img = np.clip(np_img, 0.0, 1.0) * 255.0
                np_img = np_img.astype(np.uint8)
            return np_img

        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            img_concatenated = np.concatenate(
                [_to_hwc_uint8(img[i]) for img in sample_batch["image"].values()],
                axis=1,
            )
            print(f"Sample image {i} shape: {img_concatenated.shape}")
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    # Determine if LoRA variants are requested (paligemma and/or action expert).
    pal_lora_enabled = "lora" in getattr(model_cfg, "paligemma_variant", "")
    expert_lora_enabled = "lora" in getattr(model_cfg, "action_expert_variant", "")
    lora_enabled = pal_lora_enabled or expert_lora_enabled

    def _log_lora_modules(stage: str):
        base_model = model
        if isinstance(base_model, torch.nn.parallel.DistributedDataParallel):
            base_model = base_model.module
        lora_modules = [(n, m) for n, m in base_model.named_modules() if isinstance(m, lora_pytorch.LoRALinear)]
        logging.info("[%s] LoRA modules detected: %s", stage, len(lora_modules))
        if lora_modules:
            sample = ", ".join(n for n, _ in lora_modules[:5])
            logging.info("[%s] First LoRA module names: %s%s", stage, sample, " ..." if len(lora_modules) > 5 else "")
        elif lora_enabled:
            logging.warning("[%s] LoRA expected from config but no LoRALinear modules were found.", stage)

    if hasattr(model, "gradient_checkpointing_enable") and config.pytorch_gradient_checkpointing:
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info(
            "Gradient checkpointing is %s",
            "disabled by config" if hasattr(model, "gradient_checkpointing_enable") else "not supported for this model",
        )

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Log LoRA presence right after model construction
    if is_main:
        _log_lora_modules("after_model_build")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")

        if getattr(config, "reinit_action_expert", False):
            # Load checkpoint selectively: drop action/state projection weights when their shapes
            # don't match the current model (e.g., switching action_dim).
            state_dict = safetensors.torch.load_file(model_path)
            drop_keys = {
                "action_in_proj.weight",
                "action_in_proj.bias",
                "action_out_proj.weight",
                "action_out_proj.bias",
                "state_proj.weight",
                "state_proj.bias",
            }
            # Drop any force/torque encoder params so they start from scratch too.
            ft_prefix = "force_torque_axis_cnns"
            for key in list(state_dict.keys()):
                base_key = key.removeprefix("module.")
                if base_key in drop_keys or base_key.startswith(ft_prefix):
                    state_dict.pop(key)

            missing, unexpected = (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model).load_state_dict(state_dict, strict=False)
            logging.info(
                "Loaded PyTorch weights with action expert reinit; missing=%s unexpected=%s", missing, unexpected
            )
        else:
            # Allow missing LoRA parameters when base checkpoints do not contain them.
            state_dict = safetensors.torch.load_file(model_path)
            missing, unexpected = (
                model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            ).load_state_dict(state_dict, strict=False)
            logging.info(
                "Loaded PyTorch weights from %s (missing=%s unexpected=%s, lora_enabled=%s)",
                config.pytorch_weight_path,
                missing,
                unexpected,
                lora_enabled,
            )

    # Compute optional FT staged schedule (overrides legacy freeze when enabled).
    # If user provides values, use them. Otherwise, force defaults (3000, 10000)
    # and ignore legacy freeze settings.
    ft_phase1_steps = getattr(config, "ft_cnn_warmup_steps", 0) or 0
    ft_phase2_steps = getattr(config, "ft_cnn_head_steps", 0) or 0
    if ft_phase1_steps == 0 and ft_phase2_steps == 0:
        ft_phase1_steps = 3000
        ft_phase2_steps = 10000
    ft_schedule_enabled = (ft_phase1_steps > 0) or (ft_phase2_steps > 0)
    ft_phase1_end = ft_phase1_steps
    ft_phase2_end = ft_phase1_steps + ft_phase2_steps

    # Compute optional freeze window; apply after we know the resume step.
    freeze_steps_total = 0
    explicit_steps = 0
    # explicit_steps = getattr(config, "freeze_pretrained_steps", 0) or 0
    # When FT schedule is enabled (always, due to defaults above), we intentionally ignore legacy freeze.
    if not ft_schedule_enabled:
        if explicit_steps > 0:
            freeze_steps_total = min(explicit_steps, max(1, config.num_train_steps))
        elif getattr(config, "reinit_action_expert", False):
            # Preserve the legacy fraction behavior only when explicitly reinitializing the expert.
            fraction = getattr(config, "freeze_pretrained_fraction", 0.0) or 0.0
            if fraction > 0:
                freeze_steps_total = max(1, int(config.num_train_steps * fraction))
                freeze_steps_total = min(freeze_steps_total, max(1, config.num_train_steps // 2))

    def _log_trainable_mask(stage: str):
        base_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        trainable = []
        frozen = []
        for name, param in base_model.named_parameters():
            (trainable if param.requires_grad else frozen).append(name)
        logging.info("[%s] trainable (%s): %s", stage, len(trainable), trainable)
        logging.info("[%s] frozen (%s): %s", stage, len(frozen), frozen)

    def _apply_trainable_mask(
        freeze_active: bool,
        *,
        stage: str | None = None,
        ft_phase: str | None = None,
    ):
        base_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        action_roots = {
            "action_in_proj",
            "action_out_proj",
            "state_proj",
            "force_torque_axis_cnns",
            "force_torque_patch_encoders",
            "force_torque_cnns",
        }

        # Identify LoRA param names even if the raw name does not contain "lora_".
        lora_param_names: set[str] = set()
        for mod_name, mod in base_model.named_modules():
            if isinstance(mod, lora_pytorch.LoRALinear):
                for pname, _ in mod.named_parameters(recurse=False):
                    full = f"{mod_name}.{pname}" if mod_name else pname
                    lora_param_names.add(full)

        trainable_params = frozen_params = 0
        for name, param in base_model.named_parameters():
            root = name.split(".")[0]
            is_ft = root in {"force_torque_cnns", "force_torque_axis_cnns", "force_torque_patch_encoders"}
            is_action_head = root in {"action_in_proj", "action_out_proj", "state_proj"}

            if ft_phase == "ft_cnn_only":
                train = is_ft
            elif ft_phase == "ft_cnn_plus_head":
                train = is_ft or is_action_head
            elif ft_phase == "ft_cnn_frozen_full":
                train = not is_ft
            else:
                # Legacy / default behavior
                if root in action_roots:
                    train = True
                elif ".gemma_expert." in name:
                    if getattr(config, "reinit_action_expert", False):
                        # Reinitialized expert trains immediately.
                        train = True
                    elif expert_lora_enabled:
                        # Expert with LoRA: freeze base, train LoRA after freeze window.
                        train = (name in lora_param_names) and (not freeze_active)
                    else:
                        # Expert without LoRA: train after freeze window.
                        train = not freeze_active
                else:
                    # Paligemma (or other non-expert params)
                    if pal_lora_enabled:
                        # Paligemma with LoRA: freeze base, train LoRA after freeze window.
                        train = (name in lora_param_names) and (not freeze_active)
                    else:
                        # Paligemma without LoRA: train after freeze window.
                        train = not freeze_active
            param.requires_grad = train
            if train:
                trainable_params += 1
            else:
                frozen_params += 1
        logging.info(
            "Applied trainable mask (freeze_active=%s): trainable=%s frozen=%s",
            freeze_active,
            trainable_params,
            frozen_params,
        )
        _log_trainable_mask(stage or (ft_phase if ft_phase is not None else ("freeze_active_true" if freeze_active else "freeze_active_false")))

    # FT phased schedule helper
    def _ft_phase_for_step(step: int) -> str | None:
        if not ft_schedule_enabled:
            return None
        if step < ft_phase1_end:
            return "ft_cnn_only"
        if step < ft_phase2_end:
            return "ft_cnn_plus_head"
        return "ft_cnn_frozen_full"

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, run_checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    # Apply optional freeze after knowing starting step (handles resume correctly).
    freeze_steps_remaining = 0
    current_ft_phase = _ft_phase_for_step(global_step)
    if ft_schedule_enabled:
        _apply_trainable_mask(freeze_active=False, ft_phase=current_ft_phase, stage="initial_ft_phase")
        logging.info(
            "Using FT staged schedule: phase1(cnn-only)=%s, phase2(cnn+head)=%s, phase3(full-no-cnn)=rest (starting at step %s in phase %s)",
            ft_phase1_steps,
            ft_phase2_steps,
            global_step,
            current_ft_phase,
        )
    else:
        if freeze_steps_total > 0:
            if global_step >= freeze_steps_total:
                _apply_trainable_mask(freeze_active=False, stage="initial_after_resume_past_freeze")
                logging.info(
                    f"Skip freeze: resuming at step {global_step} which is past freeze window ({freeze_steps_total} steps)"
                )
            else:
                freeze_steps_remaining = freeze_steps_total - global_step
                _apply_trainable_mask(freeze_active=True, stage="initial_in_freeze_window")
                logging.info(
                    f"Freezing pretrained weights for first {freeze_steps_total} steps (remaining from step {global_step}: {freeze_steps_remaining}); only scratch heads (action/FT, reinitialized expert) stay trainable"
                )
        else:
            _apply_trainable_mask(freeze_active=False, stage="initial_no_freeze_window")
            logging.info(
                "No backbone freeze window; paligemma base stays frozen, action/FT/gemma_expert train, LoRA unfrozen."
            )

    if is_main:
        _log_lora_modules("after_mask_application")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    def run_validation_step(step: int):
        nonlocal val_iter
        if val_loader is None:
            return None
        try:
            observation, actions = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            observation, actions = next(val_iter)

        observation = jax.tree.map(lambda x: x.to(device), observation)
        actions = actions.to(torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            losses = model(observation, actions)
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)
            val_loss = losses.mean().item()
        model.train()
        if is_main and config.wandb_enabled:
            wandb.log({"val_loss": val_loss}, step=step)
        return val_loss

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # Handle FT staged phase transitions or legacy freeze window.
            if ft_schedule_enabled:
                new_phase = _ft_phase_for_step(global_step)
                if new_phase != current_ft_phase:
                    current_ft_phase = new_phase
                    _apply_trainable_mask(freeze_active=False, ft_phase=current_ft_phase, stage=f"ft_phase_{current_ft_phase}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logging.info("Switched FT phase to %s at step %s", current_ft_phase, global_step)
            elif freeze_steps_remaining > 0 and global_step >= freeze_steps_total:
                _apply_trainable_mask(freeze_active=False, stage="unfreeze_exit_freeze_window")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logging.info(
                    f"Exited freeze window at step {global_step}; LoRA and non-reinitialized action expert params are now trainable, paligemma base still frozen"
                )
                freeze_steps_remaining = 0  # disable further checks

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward pass
            losses = model(observation, actions)
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            # Backward pass
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                # Run a quick val step and log
                run_validation_step(global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, run_checkpoint_dir, config, is_main, data_config)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Safety: ensure backbone not left frozen if training ended early.
    if freeze_steps_total > 0 and freeze_steps_remaining > 0:
        base_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        for _, param in base_model.named_parameters():
            param.requires_grad = True
        logging.warning(
            "Training finished before unfreeze threshold; backbone has been unfrozen now for any further usage." 
            f"(freeze_steps_total={freeze_steps_total}, reached_step={global_step})"
        )

    # Log any remaining metrics that didn't hit a log interval
    if is_main and config.wandb_enabled and len(infos) > 0:
        avg_loss = sum(info["loss"] for info in infos) / len(infos)
        avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
        avg_grad_norm = None
        vals = [info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None]
        if len(vals) > 0:
            avg_grad_norm = sum(vals) / len(vals)
        log_payload = {
            "loss": avg_loss,
            "learning_rate": avg_lr,
            "step": global_step - 1,
            "time_per_step": (time.time() - start_time) / len(infos),
        }
        if avg_grad_norm is not None:
            log_payload["grad_norm"] = avg_grad_norm
        wandb.log(log_payload, step=global_step - 1)

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    host = socket.gethostname()
    print(
        f"[{host}] torch info: {torch.__version__}, cuda available: {torch.cuda.is_available()}, cuda devices: {torch.cuda.device_count()}"
    )
    train_loop(config)


if __name__ == "__main__":
    main()
