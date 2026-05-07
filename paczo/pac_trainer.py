# coding=utf-8
# PACZero Trainer: Sign-quantized M-subset PAC with exact binary MI.


import math
import os
import random
import re
import shutil
import sys
import time
from typing import TYPE_CHECKING
import json

from src.metrics import f1
import numpy as np

from transformers import Trainer

from transformers.integrations import (
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
)

import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
)
from torch.utils.data.distributed import DistributedSampler

from huggingface_hub import Repository

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import (
    DataCollator,
    DataCollatorWithPadding,
    default_data_collator,
)
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import (
    PreTrainedModel,
    load_sharded_checkpoint,
    unwrap_model,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.optimization import Adafactor, get_scheduler
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_1_10,
    is_torch_less_than_1_11,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    find_labels,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tensorrt_fx_available,
    is_torch_tpu_available,
    is_torchdynamo_available,
    logging,
)
from transformers.utils.generic import ContextManagers

from paczo.pac_utils import (
    binary_channel_mi,
    sigma_for_binary_mi,
    update_p_binary,
    compute_subset_ghats,
)

_is_native_cpu_amp_available = is_torch_greater_or_equal_than_1_10

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
    from .trainer_pt_utils import (
        smp_forward_backward,
        smp_forward_only,
        smp_gather,
        smp_nested_concat,
    )
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)

TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class PACTrainer(Trainer):
    """
    PACZero trainer: sign-quantized M-subset PAC (full-batch, K=1).

    Per step:
        1. Compute per-sample ghat = (L(theta+eps z) - L(theta-eps z)) / (2 eps).
        2. Aggregate M subset means, then sign-quantize to s_m in {-1, +1}.
        3. Under posterior p, q_+ = P(s_secret = +1).
        4. If q_+ in {0, 1}: release = s_secret, MI = 0 (FREE step).
           Else: sigma = binary_MI_inverse(q_+, per_step_mi);
                 release = s_secret + N(0, sigma^2), MI consumed = per_step_mi.
        5. Posterior update using real-valued release, N(s_m, sigma^2).
        6. Quantize: Y = sign(release); update theta with lr_t * Y * z.
        7. Track cumulative MI; break when budget exhausted.

    In diagnostic (no_privacy) mode, uses the raw batch_mean scalar (not
    sign) and logs all PAC statistics WITHOUT adding noise.
    """

    from transformers.trainer_pt_utils import (
        _get_learning_rate,
        log_metrics,
        metrics_format,
        save_metrics,
        save_state,
    )

    def get_train_dataloader(self) -> DataLoader:
        """Full-batch dataloader: all N samples per step.

        Every training step processes the entire dataset.
        This maximizes samples per subset (~N/2) and minimizes subset mean
        variance, which is the key to making M-subset PAC trainable.

        Each sample carries a membership_vector from SubsetAwareDataset.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )

        # Full-batch: process all N samples per step.
        # We use a SequentialSampler and set batch_size = len(dataset).
        # For large datasets this might OOM; for N=1000 it's fine.
        return DataLoader(
            train_dataset,
            batch_size=len(train_dataset),
            shuffle=False,  # order doesn't matter with full batch
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=False,
            worker_init_fn=seed_worker,
        )

    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        """PACZero training loop."""
        self._train_batch_size = batch_size
        train_dataloader = self.get_train_dataloader()

        # --- Training setup (same as dp-aggzo) ---
        total_train_batch_size = (
            args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        )

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = (
                len_dataloader // args.gradient_accumulation_steps
            )
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(
                    args.num_train_epochs * num_update_steps_per_epoch
                )
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = (
                    self.num_examples(train_dataloader) * args.num_train_epochs
                )
        elif args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        if resume_from_checkpoint is not None:
            try:
                self._load_rng_state(resume_from_checkpoint)
            except Exception as _e:
                logger.warning(f"[audit-fix-C3] failed to restore RNG state: {_e}")

        # --- PAC Privacy initialization ---
        no_privacy = getattr(self.args, "no_privacy", False)
        pac_m = self.args.pac_m
        pac_mi = self.args.pac_mi
        pac_clip = self.args.pac_clip
        pac_secret_id = self.args.pac_secret_id
        pac_k = getattr(self.args, "pac_k", 1)
        pac_adaptive_mi = getattr(self.args, "pac_adaptive_mi", False)
        pac_zpl = getattr(self.args, "pac_zpl", False)

        # Mode consistency checks
        if pac_zpl and no_privacy:
            raise ValueError("--pac_zpl and --no_privacy are mutually exclusive")
        if pac_zpl and pac_adaptive_mi:
            # Adaptive MI is irrelevant under ZPL (no MI budget). Warn + ignore.
            print(
                "[v4] WARNING: --pac_adaptive_mi has no effect under --pac_zpl (ZPL has no MI budget)."
            )

        if pac_secret_id < 0:
            pac_secret_id = np.random.randint(pac_m)

        # Posterior over M subsets (uniform initially)
        pac_p = np.ones(pac_m, dtype=np.float64) / pac_m

        # Per-step MI budget. Two modes:
        #   uniform  : beta_t = MI_total / T for all t. Simple, but if many
        #                       steps are free the leftover budget is wasted.
        #   adaptive: beta_t = (MI_total - MI_used) / (T - t).
        per_step_mi_uniform = pac_mi / max_steps
        per_step_mi = per_step_mi_uniform  # will be overwritten per-step if adaptive

        # Cumulative MI consumed (for budget tracking + early stop)
        cum_mi = 0.0
        n_free_steps = 0  # steps where signs were unanimous
        n_noisy_steps = 0  # steps where noise was added

        # Diagnostic log file (always written, even in private mode)
        diag_file = os.path.join(args.output_dir, "diagnostics.jsonl")
        os.makedirs(args.output_dir, exist_ok=True)

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("=" * 60)
        if no_privacy:
            print(f"  MODE: DIAGNOSTIC (no noise, logging M-subset stats)")
        elif pac_zpl:
            print(
                f"  MODE: ZPL (zero privacy loss) — unanimous→s_secret, else→random ±1. MI budget ignored."
            )
        else:
            alloc = (
                "ADAPTIVE (beta_t = remaining/T-t)"
                if pac_adaptive_mi
                else "UNIFORM (beta = MI/T)"
            )
            print(
                f"  MODE: PACZO v4 (sign-quantized, exact binary MI); allocation: {alloc}"
            )
        print(
            f"  K={pac_k}, M={pac_m}, total_MI={pac_mi}, per_step_MI={per_step_mi:.8f}"
        )
        print(f"  secret_id={pac_secret_id}, clip={pac_clip}")
        print(f"  max_steps={max_steps}, trainable_params={n_trainable:,}")
        print("=" * 60)

        # Train!
        logger.info("***** Running PACZero training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(f"  Trainable parameters = {n_trainable:,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = (
                    self.state.global_step % num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0
            logger.info(f"  Resuming from global step {self.state.global_step}")

        if getattr(args, "overwrite_output_dir", False) and os.path.exists(diag_file):
            try:
                # Snapshot the prior diag for forensic reference, then truncate.
                snap_path = (
                    diag_file + ".pre_overwrite_" + time.strftime("%Y%m%d_%H%M%S")
                )
                os.rename(diag_file, snap_path)
                logger.info(
                    f"[pac-resume-patch] --overwrite_output_dir set: archived prior {diag_file}"
                    f" -> {snap_path}; new run starts with empty diagnostics"
                )
            except Exception as e:
                logger.warning(
                    f"[pac-resume-patch] failed to archive stale diag file: {e}"
                )

        if resume_from_checkpoint is not None and not os.path.exists(diag_file):
            raise RuntimeError(
                f"[pac-resume-patch] resume_from_checkpoint={resume_from_checkpoint} "
                f"but {diag_file} missing — cannot replay privacy state. Aborting to "
                f"prevent silent privacy violation."
            )
        if resume_from_checkpoint is not None and os.path.exists(diag_file):
            prior_mi = 0.0
            n_rows = 0
            n_skipped = 0
            try:
                with open(diag_file, "r") as _df:
                    for _line in _df:
                        try:
                            _row = json.loads(_line)
                            prior_mi += float(_row.get("mi_step", 0.0) or 0.0)
                            n_rows += 1
                        except Exception:
                            n_skipped += 1
                cum_mi = float(prior_mi)
                logger.info(
                    f"[pac-resume-patch] Restored cum_mi={cum_mi:.6f} by replaying"
                    f" {n_rows} prior rows in {diag_file}"
                    + (f" ({n_skipped} unparseable rows skipped)" if n_skipped else "")
                )
                # Sanity: warn loudly if budget is already over-spent
                if (not no_privacy) and (not pac_zpl) and cum_mi >= pac_mi:
                    logger.warning(
                        f"[pac-resume-patch] cum_mi={cum_mi:.6f} ALREADY >= pac_mi={pac_mi}"
                        f" at resume; the privacy budget is exhausted by prior released bits."
                        f" The training loop will halt on the first step (line ~864 budget gate)."
                    )
            except Exception as e:
                logger.error(
                    f"[pac-resume-patch] FAILED to replay {diag_file}: {e}."
                    f" Aborting to avoid silent privacy violation."
                )
                raise

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = (
                trial.assignments
                if self.hp_search_backend == HPSearchBackend.SIGOPT
                else trial
            )
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        tr_loss = torch.tensor(0.0).to(args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        assert (
            args.gradient_accumulation_steps == 1
        ), "PACZero requires gradient_accumulation_steps == 1"

        # With full-batch DataLoader, len_dataloader=1 (one batch = all data).
        # So each "epoch" = 1 step. We need max_steps epochs.
        for epoch in range(epochs_trained, max_steps):
            epoch_iterator = train_dataloader

            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            for step, inputs in enumerate(epoch_iterator):
                if not inputs:
                    print("empty batch, skip")
                    continue

                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                self.control = self.callback_handler.on_step_begin(
                    args, self.state, self.control
                )

                # ============================================================
                # PACZero: Full-batch M-subset PAC step (K=1)
                # ============================================================

                # 1. Pop membership vectors
                membership_vectors = inputs.pop("membership_vector", None)
                if membership_vectors is not None:
                    membership_np = membership_vectors.numpy().astype(np.bool_)
                else:
                    raise ValueError(
                        "v4 requires SubsetAwareDataset (membership_vector missing)"
                    )

                # 2. Collect trainable parameters
                self.named_parameters_to_optim = []
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        self.named_parameters_to_optim.append((name, param))

                # 3. K=1 ZO step: compute per-sample directional derivatives
                random_seed = int(np.random.randint(1000000000))

                # Perturb θ → θ + ε*z
                torch.manual_seed(random_seed)
                for name, param in self.named_parameters_to_optim:
                    z = torch.normal(
                        mean=0,
                        std=1,
                        size=param.data.size(),
                        device=param.data.device,
                        dtype=param.data.dtype,
                    )
                    param.data = param.data + self.args.zo_eps * z
                    del z

                loss_plus = self.zo_forward_per_sample(model, inputs)  # (N,)

                # Perturb θ+ε*z → θ-ε*z
                torch.manual_seed(random_seed)
                for name, param in self.named_parameters_to_optim:
                    z = torch.normal(
                        mean=0,
                        std=1,
                        size=param.data.size(),
                        device=param.data.device,
                        dtype=param.data.dtype,
                    )
                    param.data = param.data + (-2.0) * self.args.zo_eps * z
                    del z

                loss_minus = self.zo_forward_per_sample(model, inputs)  # (N,)

                # Float32 finite differencing
                ghat_per_sample = (loss_plus.float() - loss_minus.float()) / (
                    2.0 * self.args.zo_eps
                )
                ghat_np = ghat_per_sample.detach().cpu().numpy().astype(np.float64)

                # Restore θ
                torch.manual_seed(random_seed)
                for name, param in self.named_parameters_to_optim:
                    z = torch.normal(
                        mean=0,
                        std=1,
                        size=param.data.size(),
                        device=param.data.device,
                        dtype=param.data.dtype,
                    )
                    param.data = param.data + self.args.zo_eps * z
                    del z

                del loss_plus, loss_minus

                # 4. Per-sample clipping (stability only; doesn't affect sign)
                sample_abs = np.abs(ghat_np)
                pre_clip_mean = float(sample_abs.mean())
                pre_clip_max = float(sample_abs.max())
                clip_scale = np.minimum(1.0, pac_clip / np.maximum(sample_abs, 1e-12))
                ghat_clipped = ghat_np * clip_scale
                clip_fired_frac = float((sample_abs > pac_clip).mean())

                # 5. Compute M subset means, then sign-quantize
                m_ghats = compute_subset_ghats(
                    ghat_clipped, membership_np, pac_m
                )  # (M,)
                signs = np.where(m_ghats >= 0, 1.0, -1.0).astype(
                    np.float64
                )  # (M,) in {-1,+1}

                # 6. Diagnostic stats (measured on m_ghats pre-quantization)
                batch_mean_scalar = float(ghat_clipped.mean())
                signal_raw = float(np.abs(m_ghats[pac_secret_id]))
                var_subset_raw = float(np.var(m_ghats))
                batch_size_actual = len(ghat_np)

                # Agreement statistics
                q_plus = float(np.sum(pac_p * (signs > 0).astype(np.float64)))
                agreement_frac = max(
                    q_plus, 1.0 - q_plus
                )  # fraction of posterior mass on majority
                unanimous = (q_plus <= 1e-12) or (q_plus >= 1.0 - 1e-12)

                # Posterior entropy
                entropy = float(-np.sum(pac_p * np.log(pac_p + 1e-300)))
                max_entropy = float(np.log(pac_m))

                s_secret = float(signs[pac_secret_id])

                # Per-step MI budget: uniform or adaptive.
                # Adaptive allocation uses only distributional info (cum_mi from Y_<t,
                # and the public step index) — no S-dependence — so the PAC chain-rule
                # bound I(S; Y_1..T) <= sum_t beta_t = pac_mi still holds without new proof.
                per_step_mi_capped = False
                if pac_adaptive_mi:
                    remaining_steps = max(max_steps - self.state.global_step, 1)
                    per_step_mi = max(pac_mi - cum_mi, 0.0) / remaining_steps
                else:
                    per_step_mi = per_step_mi_uniform

                # Entropy-ceiling cap
                if (q_plus > 0.0) and (q_plus < 1.0):
                    h_s_binary = float(
                        -q_plus * np.log(q_plus) - (1.0 - q_plus) * np.log(1.0 - q_plus)
                    )

                    h_s_cap = 0.999 * h_s_binary
                    if per_step_mi > h_s_cap:
                        per_step_mi = h_s_cap
                        per_step_mi_capped = True

                pos_sign_count = int((signs > 0).sum())
                unanimous_true = (pos_sign_count == 0) or (pos_sign_count == pac_m)

                # release_type is logged in the diag: one of
                #   "unanimous"        — deterministic release of s_secret (zero leak)
                #   "noisy_gaussian"   — Gaussian-σ PAC-MI release (E1/E2 private mode)
                #   "zpl_random"       — pure coin-flip ±1 (ZPL, zero leak by independence)
                #   "ablation_*"       — no_privacy ablation variants
                release_type = None

                # 7. PAC mechanism: sign-quantized release
                if no_privacy:
                    # DIAGNOSTIC / ABLATION MODE — no noise, no MI accounting.
                    # `ablation` selects which "part" of the v4 mechanism is active,
                    # for loss-decomposition experiments.
                    ablation = getattr(self.args, "ablation", "none")
                    if ablation == "none":
                        # raw full-batch mean — pure non-private MeZO baseline
                        projected_grad = batch_mean_scalar
                    elif ablation == "quant_full":
                        # sign(full-batch mean) — isolates quantization cost
                        projected_grad = 1.0 if batch_mean_scalar >= 0 else -1.0
                    elif ablation == "raw_half":
                        # raw secret-subset mean (N/2 samples) — isolates subset-size cost
                        projected_grad = float(m_ghats[pac_secret_id])
                    elif ablation == "quant_half":
                        # sign(secret-subset mean) — quantization + subset, no noise
                        projected_grad = s_secret
                    elif ablation == "random_sign":
                        # Pure random ±1, uncorrelated with gradient. Control for
                        # "how much of the full-mechanism result is just from
                        # applying random ±1 updates (stochastic regularization)?"
                        projected_grad = 1.0 if np.random.random() < 0.5 else -1.0
                    else:
                        raise ValueError(f"Unknown ablation: {ablation}")
                    released_bit = s_secret
                    sigma_used = 0.0
                    mi_step = 0.0
                    release_type = f"ablation_{ablation}"
                    # Hypothetical posterior update (for diagnostic logging)
                    if not unanimous:
                        sigma_hyp = sigma_for_binary_mi(q_plus, per_step_mi) or 0.0
                        if sigma_hyp > 0:
                            hyp_noise = float(np.random.randn() * sigma_hyp)
                            hyp_release = s_secret + hyp_noise
                            pac_p = update_p_binary(
                                pac_p, signs, hyp_release, sigma_hyp
                            )
                elif pac_zpl:
                    if unanimous_true:
                        released_bit = s_secret
                        release_type = "unanimous"
                    else:
                        released_bit = 1.0 if np.random.random() < 0.5 else -1.0
                        release_type = "zpl_random"
                    sigma_used = 0.0
                    mi_step = 0.0
                    # Posterior p_t stays uniform for the entire run.
                    projected_grad = released_bit
                    cum_mi += mi_step  # always 0, kept for log uniformity
                    if release_type == "unanimous":
                        n_free_steps += 1
                    else:
                        n_noisy_steps += (
                            1  # repurposed as "random-flip steps" under ZPL
                        )
                else:
                    # PRIVATE MODE (PAC-MI with Gaussian noise — E1 uniform or E2 adaptive)
                    if unanimous:
                        # All subsets agree on sign under current posterior.
                        # Release the sign deterministically; I(S;Y) = 0.
                        released_bit = s_secret
                        sigma_used = 0.0
                        mi_step = 0.0
                        release_type = "unanimous"
                        # Posterior is unchanged (likelihood identical across all m with s_m = s_secret;
                        # any m with differing s_m already has ~0 posterior mass).
                    else:
                        # Solve for sigma such that I(S; s_secret + N(0, sigma^2)) <= per_step_mi.
                        sigma_used = sigma_for_binary_mi(q_plus, per_step_mi)
                        if sigma_used is None or sigma_used <= 0.0:
                            # Shouldn't happen given q_+ in (0,1) and per_step_mi > 0.
                            released_bit = s_secret
                            sigma_used = 0.0
                            mi_step = float(
                                -q_plus * np.log(q_plus)
                                - (1 - q_plus) * np.log(1 - q_plus)
                            )
                            release_type = "unanimous_fallback"
                        else:
                            noise = float(np.random.randn() * sigma_used)
                            released_real = s_secret + noise
                            # Post-process to {-1, +1} for the update
                            released_bit = 1.0 if released_real >= 0 else -1.0
                            # Posterior update on the *real-valued* release (tighter Bayesian update)
                            pac_p = update_p_binary(
                                pac_p, signs, released_real, sigma_used
                            )
                            # Actual MI consumed: target per_step_mi (bisection converges within tol).
                            mi_step = float(per_step_mi)
                            release_type = "noisy_gaussian"

                    cum_mi += mi_step
                    if mi_step > 0.0:
                        n_noisy_steps += 1
                    else:
                        n_free_steps += 1

                    # The trainer applies lr_t * projected_grad * z as the parameter update.
                    # For PACZero, projected_grad = released_bit in {-1, +1}; magnitude is controlled
                    # by lr_t (which decays per args.lr_scheduler_type).
                    projected_grad = released_bit

                # Diagnostics: per-step SNR of the pre-quantization path (for comparison)
                snr_per_step = signal_raw / (sigma_used if sigma_used > 0 else 1e-12)

                # 8. Update parameters
                lr = self._get_learning_rate()
                torch.manual_seed(random_seed)
                for name, param in self.named_parameters_to_optim:
                    z = torch.normal(
                        mean=0,
                        std=1,
                        size=param.data.size(),
                        device=param.data.device,
                        dtype=param.data.dtype,
                    )
                    if (
                        "bias" not in name
                        and "layer_norm" not in name
                        and "layernorm" not in name
                    ):
                        param.data = param.data - lr * (
                            projected_grad * z + args.weight_decay * param.data
                        )
                    else:
                        param.data = param.data - lr * (projected_grad * z)
                    del z

                # Step the LR scheduler (we bypass optimizer.step() for manual ZO update,
                # so the scheduler needs to be stepped explicitly to get decay).
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # 9. Training loss (for internal monitoring)
                tr_loss_step = self.zo_forward(model, inputs).mean()
                tr_loss += tr_loss_step
                torch.cuda.empty_cache()
                self.current_flos += float(self.floating_point_ops(inputs))

                # 10. Write diagnostic log (v4: agreement + binary-MI fields)
                diag_entry = {
                    "step": self.state.global_step,
                    "signal_raw": signal_raw,
                    "batch_mean": batch_mean_scalar,
                    "var_subset_raw": var_subset_raw,
                    "q_plus": q_plus,
                    "agreement_frac": agreement_frac,
                    "unanimous": bool(unanimous),
                    "sigma_used": float(sigma_used),
                    "mi_step": float(mi_step),
                    "per_step_mi_budget": float(per_step_mi),
                    "per_step_mi_capped": bool(per_step_mi_capped),
                    "adaptive_mi": bool(pac_adaptive_mi),
                    "zpl": bool(pac_zpl),
                    "release_type": release_type,
                    "unanimous_true": bool(unanimous_true),
                    "pos_sign_count": pos_sign_count,
                    "cum_mi": float(cum_mi),
                    "n_free_steps": n_free_steps,
                    "n_noisy_steps": n_noisy_steps,
                    "s_secret": float(s_secret),
                    "released_bit": float(released_bit),
                    "snr_per_step_raw": snr_per_step,
                    "entropy": entropy,
                    "entropy_ratio": entropy / max_entropy if max_entropy > 0 else 0,
                    "pre_clip_mean": pre_clip_mean,
                    "pre_clip_max": pre_clip_max,
                    "clip_fired_frac": clip_fired_frac,
                    "batch_size": batch_size_actual,
                    "projected_grad": float(projected_grad),
                    "train_loss": float(tr_loss_step.item()),
                    "lr": lr,
                    "pac_clip": float(pac_clip),
                    "pac_m": pac_m,
                    "mode": "diagnostic" if no_privacy else "private",
                }
                with open(diag_file, "a") as f:
                    f.write(json.dumps(diag_entry) + "\n")

                # 11. HF logging
                if self.state.global_step % args.logging_steps == 0:
                    log_dict = {
                        "signal_raw": signal_raw,
                        "q_plus": q_plus,
                        "agreement_frac": agreement_frac,
                        "sigma_used": float(sigma_used),
                        "cum_mi": float(cum_mi),
                        "free_frac": n_free_steps
                        / max(n_free_steps + n_noisy_steps, 1),
                        "entropy_ratio": (
                            entropy / max_entropy if max_entropy > 0 else 0
                        ),
                        "clip_fired_frac": clip_fired_frac,
                    }
                    self.log(log_dict)

                # 12. Privacy budget early-stop
                if (not no_privacy) and (not pac_zpl) and cum_mi >= pac_mi:
                    print(
                        f"[v4] Privacy budget exhausted at step {self.state.global_step}: "
                        f"cum_mi={cum_mi:.6f} >= pac_mi={pac_mi} "
                        f"(free={n_free_steps}, noisy={n_noisy_steps})"
                    )
                    self.control.should_training_stop = True

                # Step bookkeeping
                self.state.global_step += 1
                self.state.epoch = epoch + 1

                self.control = self.callback_handler.on_step_end(
                    args, self.state, self.control
                )
                self._maybe_log_save_evaluate(
                    torch.tensor(0.0), model, trial, epoch, ignore_keys_for_eval
                )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control
            )
            self._maybe_log_save_evaluate(
                torch.tensor(0.0), model, trial, epoch, ignore_keys_for_eval
            )

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        logger.info("\n\nTraining completed.\n\n")

        train_loss = float(tr_loss.item()) / max(self.state.global_step, 1)
        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False
        self._memory_tracker.stop_and_update_metrics(metrics)
        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=False, output_dir=run_dir
        )

        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
        ):
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
                    )
                    shutil.rmtree(checkpoint)

        # Dev-plateau model selection
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f"[pac_load_best_dev] Loading best-dev checkpoint: "
                f"{self.state.best_model_checkpoint} "
                f"(best metric {self.args.metric_for_best_model}={self.state.best_metric})"
            )
            self._load_best_model()

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )
        return TrainOutput(self.state.global_step, train_loss, metrics)

    ############## ZO Methods ##############

    def _micro_batch_forward(self, model, inputs, micro_batch_size=64):
        """Run forward pass in micro-batches to avoid OOM with full-batch.

        Returns per-sample losses of shape (N,) by chunking the batch into
        micro-batches of size `micro_batch_size`, running each through the
        model, and concatenating the results.
        """
        model.eval()

        # Figure out batch dimension
        batch_size = None
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                batch_size = v.size(0)
                break
        if batch_size is None:
            raise ValueError("Cannot determine batch size from inputs")

        all_losses = []
        for start in range(0, batch_size, micro_batch_size):
            end = min(start + micro_batch_size, batch_size)
            micro_inputs = {
                k: v[start:end] if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            with torch.inference_mode():
                micro_inputs = self._prepare_inputs(micro_inputs)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, micro_inputs)
            # loss should be (micro_batch,) with dpzero=True
            if isinstance(loss, torch.Tensor) and loss.dim() == 0:
                loss = loss.unsqueeze(0)
            all_losses.append(loss.detach())
            del micro_inputs
            torch.cuda.empty_cache()

        return torch.cat(all_losses, dim=0)  # (N,)

    def zo_forward(self, model, inputs):
        """Batch-mean scalar loss for logging. Micro-batched to avoid OOM."""
        model.eval()
        if self.args.non_diff:
            # Route to per-sample nondiff then reduce. Needed for v4 M-subset PAC:
            # we compute per-sample negF1, aggregate into subset means, quantize, etc.
            return self.zo_forward_nondiff_per_sample(model, inputs).mean()
        per_sample = self._micro_batch_forward(model, inputs)
        return per_sample.mean()

    def zo_forward_per_sample(self, model, inputs):
        """Per-sample losses, shape (batch_size,). Micro-batched."""
        if self.args.non_diff:
            return self.zo_forward_nondiff_per_sample(model, inputs)
        return self._micro_batch_forward(model, inputs)

    def zo_forward_nondiff_per_sample(self, model, inputs, gen_micro_batch_size=16):
        """Per-sample negative F1 for generation tasks (SQuAD).

        Returns (N,) tensor of -F1 values, one per sample. Generation is micro-
        batched to avoid OOM on long-context tasks (SQuAD prompts can be 1.5k+
        tokens). The `gold` field is carried through from NondiffCollator as a
        Python list of gold answers.
        """
        model.eval()
        assert (
            self.args.task_name == "SQuAD"
        ), f"zo_forward_nondiff_per_sample is only implemented for SQuAD, got {self.args.task_name}"

        # Determine batch size
        bsz = None
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                bsz = v.size(0)
                break
        if bsz is None:
            raise ValueError("Cannot determine batch size from inputs")

        args = self.args
        all_f1s = []
        for start in range(0, bsz, gen_micro_batch_size):
            end = min(start + gen_micro_batch_size, bsz)
            micro = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    micro[k] = v[start:end]
                elif isinstance(v, list):
                    micro[k] = v[start:end]
                else:
                    micro[k] = v
            with torch.inference_mode():
                micro = self._prepare_inputs(micro)
                outputs = self.model.generate(
                    micro["input_ids"],
                    do_sample=args.sampling,
                    temperature=args.temperature,
                    num_beams=args.num_beams,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=min(
                        args.max_new_tokens,
                        args.max_length - micro["input_ids"].size(1),
                    ),
                    num_return_sequences=1,
                    eos_token_id=[
                        self.tokenizer.encode(args.eos_token, add_special_tokens=False)[
                            -1
                        ],
                        self.tokenizer.eos_token_id,
                    ],
                )
                prompt_len = micro["input_ids"].size(1)
                for i in range(outputs.size(0)):
                    decoded = self.tokenizer.decode(
                        outputs[i][prompt_len:], skip_special_tokens=True
                    ).strip()
                    gold_i = micro["gold"][i]
                    all_f1s.append(f1(decoded, gold_i))
            torch.cuda.empty_cache()
        return -torch.tensor(all_f1s, dtype=torch.float32)

    # Kept as a thin wrapper for backward compatibility with any external callers.
    def zo_forward_nondiff(self, model, inputs):
        """Scalar mean negative F1 (SQuAD). Backward-compat alias."""
        return self.zo_forward_nondiff_per_sample(model, inputs).mean()

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        if labels is not None:
            if (
                unwrap_model(model)._get_name()
                in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values()
            ):
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if not getattr(self.args, "non_diff", False):
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        # Trainer.get_eval_dataloader -> _remove_unused_columns strips the `gold`
        # field (model.forward doesn't take it). gold is needed by
        # zo_forward_nondiff_per_sample for F1 computation. Manually batch the
        # eval_dataset using our PACDataCollator/NondiffCollator (which DO preserve
        # gold) to bypass the column filter.
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        bsz = max(1, int(self.args.per_device_eval_batch_size or 16))
        all_neg_f1 = []
        for start in range(0, len(eval_dataset), bsz):
            end = min(start + bsz, len(eval_dataset))
            raw_features = [eval_dataset[i] for i in range(start, end)]
            batch = self.data_collator(raw_features)
            batch = self._prepare_inputs(batch)
            neg_f1 = self.zo_forward_nondiff_per_sample(self.model, batch)
            all_neg_f1.append(neg_f1.detach().cpu())
        neg_f1_all = torch.cat(all_neg_f1, dim=0)
        f1_mean = float(-neg_f1_all.mean().item())
        eval_loss = float(
            neg_f1_all.mean().item()
        )  # = -mean_F1, so lower=better stays consistent
        metrics = {
            f"{metric_key_prefix}_loss": eval_loss,
            f"{metric_key_prefix}_f1": f1_mean,
            f"{metric_key_prefix}_runtime": 0.0,
            f"{metric_key_prefix}_samples_per_second": 0.0,
            f"{metric_key_prefix}_steps_per_second": 0.0,
        }
        self.log(metrics)
        return metrics
