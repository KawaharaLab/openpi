import dataclasses
import functools
import logging
import os
import platform
import re
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.models.model_ft as _model_ft
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

# JAX defaults to preallocating most GPU memory, which can cause OOM due to large
# transient allocations during compilation/execution. Keep explicit user settings.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def init_logging():
    """Custom logging format for better readability."""
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
    logger.handlers[0].setFormatter(formatter)


def _global_rank() -> int:
    for key in ("RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
    return 0


def _is_primary_process() -> bool:
    return _global_rank() == 0


def _maybe_initialize_jax_distributed() -> None:
    """Initialize JAX multi-process runtime when launched via torchrun/mpiexec."""
    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")),
        )
    )
    if world_size <= 1:
        return

    rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("SLURM_PROCID", "0"))))
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")),
        )
    )
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")
    if not master_addr or not master_port:
        raise ValueError("WORLD_SIZE>1 requires MASTER_ADDR and MASTER_PORT for jax.distributed.initialize().")

    # torchrun uses MASTER_PORT for c10d rendezvous; use a dedicated JAX coordinator port.
    jax_port = int(os.environ.get("JAX_COORDINATOR_PORT", str(int(master_port) + 1)))
    coordinator = f"{master_addr}:{jax_port}"
    coordinator_bind = f"0.0.0.0:{jax_port}"
    logging.info(
        "Initializing jax.distributed: world_size=%s rank=%s local_rank=%s coordinator=%s bind=%s",
        world_size,
        rank,
        local_rank,
        coordinator,
        coordinator_bind,
    )
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=world_size,
        process_id=rank,
        local_device_ids=[local_rank],
        coordinator_bind_address=coordinator_bind,
    )


def _run_name_sync_file(config: _config.TrainConfig) -> epath.Path:
    base_dir = epath.Path(config.checkpoint_base_dir).resolve() / config.name
    token = config.exp_name or "default"
    token = re.sub(r"[^A-Za-z0-9_.-]", "_", token)
    return base_dir / f".wandb_run_name.{token}.txt"


def _resolve_wandb_run_name(config: _config.TrainConfig, *, enabled: bool) -> str:
    """Resolve run name on rank0 and broadcast to all ranks via shared file."""
    is_primary = _is_primary_process()
    sync_file = _run_name_sync_file(config)
    sync_file.parent.mkdir(parents=True, exist_ok=True)
    sync_start = time.time()

    run_name = config.exp_name

    if is_primary:
        if sync_file.exists():
            sync_file.unlink()
        if enabled:
            # Do not force name so we can use wandb-generated run names.
            wandb.init(config=dataclasses.asdict(config), project=config.name)
            run_name = wandb.run.name or run_name
            if run_name is None:
                run_name = f"run_{int(time.time())}"

            # Rare collision fallback: keep wandb-style name but make directory unique.
            ckpt_candidate = epath.Path(config.checkpoint_base_dir).resolve() / config.name / run_name
            if ckpt_candidate.exists() and not config.overwrite and not config.resume:
                run_name = f"{run_name}-{int(time.time())}"
                wandb.run.name = run_name

            sync_file.write_text(run_name)
        else:
            run_name = run_name or f"run_{int(time.time())}"
            sync_file.write_text(run_name)

    # Wait for rank0 to publish the resolved run name for this launch.
    deadline = time.time() + 120
    while True:
        if sync_file.exists() and sync_file.stat().mtime >= sync_start:
            break
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for run-name sync file: {sync_file}")
        time.sleep(0.2)

    resolved = sync_file.read_text().strip()
    if not resolved:
        raise ValueError(f"Resolved empty run name from sync file: {sync_file}")

    # Non-primary ranks should not create extra wandb runs.
    if (not is_primary) or (not enabled):
        wandb.init(mode="disabled")

    return resolved


def init_wandb_resume(config: _config.TrainConfig, *, enabled: bool):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    run_id_path = ckpt_dir / "wandb_id.txt"
    if not run_id_path.exists():
        raise FileNotFoundError(f"wandb_id.txt not found for resume: {run_id_path}")

    run_id = run_id_path.read_text().strip()
    wandb.init(id=run_id, resume="must", project=config.name)


def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader,
    params_shape: at.Params,
    *,
    check_dtypes: bool = True,
) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=check_dtypes)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Match PyTorch FT defaults: use bf16 training precision unless explicitly set to float32.
        # We keep a float32-loading path for checkpoint compatibility and cast after weight merge.
        if getattr(config, "pytorch_training_precision", "bfloat16") == "bfloat16":
            params = nnx_utils.state_map(params, nnx.Param, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        else:
            # Legacy behavior for float32 training: only cast frozen params to bf16 to save memory.
            params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(
        config.weight_loader,
        train_state_shape.params.to_pure_dict(),
        check_dtypes=(getattr(config, "pytorch_training_precision", "bfloat16") != "bfloat16"),
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation | _model_ft.Observation, _model.Actions],
    grad_mask: at.PyTree,
    *,
    phase_trainable_filter: Any | None = None,
    zero_force_torque: bool = False,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation | _model_ft.Observation, actions: _model.Actions
    ):
        try:
            chunked_loss = model.compute_loss(rng, observation, actions, train=True, zero_force_torque=zero_force_torque)
        except TypeError:
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out baseline frozen params + phase-frozen params.
    active_trainable_filter = config.trainable_filter if phase_trainable_filter is None else phase_trainable_filter
    diff_state = nnx.DiffState(0, active_trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    # Expand sparse grads (phase-only) to full trainable tree for optimizer state compatibility.
    params = state.params.filter(config.trainable_filter)
    grads_flat = grads.flat_state()
    grads = params.map(lambda path, v: grads_flat.get(path, v.replace(value=jnp.zeros_like(v.value))))

    # Apply phase-specific mask on top of the active filter.
    grad_mask_flat = grad_mask.flat_state()
    grads = grads.map(lambda path, g: g.replace(value=g.value * grad_mask_flat[path].value.astype(g.value.dtype)))

    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
    }
    return new_state, info


def _is_ft_param(path: tuple[Any, ...]) -> bool:
    root = str(path[0]) if path else ""
    return root in {"force_torque_axis_cnns", "force_torque_patch_encoders", "force_torque_cnns"}


def _is_action_head_param(path: tuple[Any, ...]) -> bool:
    root = str(path[0]) if path else ""
    return root in {"action_in_proj", "action_out_proj", "state_proj"}


def _build_phase_grad_masks(config: _config.TrainConfig, trainable_params: at.Params) -> dict[str, at.Params]:
    def _enabled(path: tuple[Any, ...], phase: str) -> bool:
        is_ft = _is_ft_param(path)
        is_head = _is_action_head_param(path)

        if phase == "ft_action_head_only":
            return is_head
        if phase == "ft_no_cnn":
            return not is_ft
        if phase == "ft_cnn_only_return":
            return is_ft
        return True

    phases = ["ft_full_train", "ft_action_head_only", "ft_no_cnn", "ft_cnn_only_return"]
    out = {}
    for phase in phases:
        out[phase] = trainable_params.map(
            lambda path, variable: variable.replace(
                value=np.asarray(_enabled(path, phase), dtype=variable.value.dtype)
            )
        )
    return out


def _build_phase_trainable_filters(config: _config.TrainConfig) -> dict[str, Any]:
    base_filter = config.trainable_filter
    ft_filter = nnx_utils.PathRegex(r"(force_torque_axis_cnns|force_torque_patch_encoders|force_torque_cnns)(/.*)?")
    action_head_filter = nnx_utils.PathRegex(r"(action_in_proj|action_out_proj|state_proj)(/.*)?")
    return {
        "ft_full_train": base_filter,
        "ft_action_head_only": nnx.All(base_filter, action_head_filter),
        "ft_no_cnn": nnx.All(base_filter, nnx.Not(ft_filter)),
        "ft_cnn_only_return": nnx.All(base_filter, ft_filter),
    }


def _count_params_in_filter(params: at.Params, filt: Any) -> int:
    return sum(int(np.prod(v.value.shape)) for v in params.filter(filt).flat_state().values())


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    uses_force_torque = "_ft" in config.name
    if hasattr(config.model, "use_force_torque"):
        model_uses_force_torque = getattr(config.model, "use_force_torque")
        if model_uses_force_torque != uses_force_torque:
            config = dataclasses.replace(
                config,
                model=dataclasses.replace(config.model, use_force_torque=uses_force_torque),
            )
    logging.info("Force/torque mode for %s: %s", config.name, uses_force_torque)

    _maybe_initialize_jax_distributed()
    logging.info(
        "JAX distributed: process_index=%s process_count=%s local_device_count=%s global_device_count=%s",
        jax.process_index(),
        jax.process_count(),
        jax.local_device_count(),
        jax.device_count(),
    )
    logging.info(
        "JAX GPU memory config: XLA_PYTHON_CLIENT_PREALLOCATE=%s",
        os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "<unset>"),
    )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    is_primary = _is_primary_process()

    if config.resume:
        checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir,
            keep_period=config.keep_period,
            overwrite=config.overwrite,
            resume=config.resume,
        )
        init_wandb_resume(config, enabled=(config.wandb_enabled and is_primary))
        if not (config.wandb_enabled and is_primary):
            wandb.init(mode="disabled")
    else:
        resolved_run_name = _resolve_wandb_run_name(config, enabled=config.wandb_enabled)
        config = dataclasses.replace(config, exp_name=resolved_run_name)

        if is_primary:
            checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
                config.checkpoint_dir,
                keep_period=config.keep_period,
                overwrite=config.overwrite,
                resume=False,
            )
            if config.wandb_enabled:
                (config.checkpoint_dir / "wandb_id.txt").write_text(wandb.run.id)
        else:
            # Avoid create-race: wait for rank0 to create the run directory, then open in resume mode.
            deadline = time.time() + 120
            while not config.checkpoint_dir.exists():
                if time.time() > deadline:
                    raise TimeoutError(f"Timed out waiting for checkpoint directory: {config.checkpoint_dir}")
                time.sleep(0.2)
            checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
                config.checkpoint_dir,
                keep_period=config.keep_period,
                overwrite=False,
                resume=True,
            )

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    trainable_params = train_state.params.filter(config.trainable_filter)
    phase_grad_masks = _build_phase_grad_masks(config, trainable_params)
    phase_grad_masks = {k: jax.device_put(v, replicated_sharding) for k, v in phase_grad_masks.items()}
    phase_trainable_filters = _build_phase_trainable_filters(config)
    for phase_name, phase_filter in phase_trainable_filters.items():
        logging.info(
            "Phase %s trainable params: %s",
            phase_name,
            _count_params_in_filter(trainable_params, phase_filter),
        )

    action_head_steps = getattr(config, "ft_action_head_steps", 0) if uses_force_torque else 0
    no_cnn_steps = getattr(config, "ft_no_cnn_steps", 0) if uses_force_torque else 0
    cnn_only_steps = getattr(config, "ft_cnn_only_steps", 0) if uses_force_torque else 0
    ft_schedule_enabled = (action_head_steps > 0) or (no_cnn_steps > 0) or (cnn_only_steps > 0)
    if ft_schedule_enabled:
        logging.info(
            "Using FT staged schedule: action_head_only=%s, no_cnn=%s, cnn_only=%s",
            action_head_steps,
            no_cnn_steps,
            cnn_only_steps,
        )

    def _phase_for_step(step: int) -> str:
        if not ft_schedule_enabled:
            return "ft_full_train"
        phase1_end = action_head_steps
        phase2_end = action_head_steps + no_cnn_steps
        phase3_end = action_head_steps + no_cnn_steps + cnn_only_steps
        if step < phase1_end:
            return "ft_action_head_only"
        if step < phase2_end:
            return "ft_no_cnn"
        if step < phase3_end:
            return "ft_cnn_only_return"
        return "ft_full_train"

    def _make_step_fn(phase: str, *, zero_force_torque: bool):
        return jax.jit(
            functools.partial(
                train_step,
                config,
                phase_trainable_filter=phase_trainable_filters[phase],
                zero_force_torque=zero_force_torque,
            ),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding, replicated_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )

    phases = ("ft_full_train", "ft_action_head_only", "ft_no_cnn", "ft_cnn_only_return")
    ptrain_step_by_phase = {phase: _make_step_fn(phase, zero_force_torque=False) for phase in phases}
    ptrain_step_zero_ft_by_phase = {phase: _make_step_fn(phase, zero_force_torque=True) for phase in phases}

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    current_phase = _phase_for_step(start_step)
    for step in pbar:
        phase = _phase_for_step(step)
        if phase != current_phase:
            current_phase = phase
            logging.info("Switched FT phase to %s at step %s", current_phase, step)

        zero_force_torque = uses_force_torque and (phase in {"ft_action_head_only", "ft_no_cnn"})
        phase_grad_mask = phase_grad_masks[phase]

        with sharding.set_mesh(mesh):
            if zero_force_torque:
                train_state, info = ptrain_step_zero_ft_by_phase[phase](train_rng, train_state, batch, phase_grad_mask)
            else:
                train_state, info = ptrain_step_by_phase[phase](train_rng, train_state, batch, phase_grad_mask)

        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            def _fmt_metric(value: Any) -> str:
                if isinstance(value, (str, bytes)):
                    return str(value)
                try:
                    return f"{float(np.asarray(value)):.4f}"
                except Exception:
                    return str(value)

            info_str = ", ".join(f"{k}={_fmt_metric(v)}" for k, v in reduced_info.items())
            pbar.write(f"Step {step} [{phase}]: {info_str}")
            wandb.log({**reduced_info, "ft_phase": phase}, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
