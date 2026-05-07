"""
PACZero: Sign-quantized M-subset PAC (full-batch, LoRA).

"""

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import time
import src.tasks
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    HfArgumentParser,
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
)
from typing import Union, Optional
import torch
from torch.nn.parameter import Parameter
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict
from tqdm import tqdm
from src.tasks import get_task
import json
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from src.metrics import calculate_metric
from src.utils import *
from paczo.pac_trainer_kvar import PACTrainer
from paczo.pac_utils import SubsetAwareDataset
import random
from functools import partial


@dataclass
class PACZeroArguments(TrainingArguments):
    # dataset and sampling
    task_name: str = "SST2"
    num_train: int = 0
    num_dev: int = None
    num_eval: int = None
    num_train_sets: int = None
    train_set_seed: int = None
    result_file: str = None

    # Model loading
    model_name: str = "facebook/opt-125m"
    load_float16: bool = False
    load_bfloat16: bool = False
    load_int8: bool = False
    max_length: int = 2048
    no_auto_device: bool = False

    # Calibration
    sfc: bool = False
    icl_sfc: bool = False

    # Training
    trainer: str = "none"
    only_train_option: bool = True
    train_as_classification: bool = False

    # MeZO
    zo_eps: float = 1e-3

    # Prefix tuning
    prefix_tuning: bool = False
    num_prefix: int = 5
    no_reparam: bool = True
    prefix_init_by_real_act: bool = True

    # LoRA
    lora: bool = False
    lora_alpha: int = 16
    lora_r: int = 8

    # Generation
    sampling: bool = False
    temperature: float = 1.0
    num_beams: int = 1
    top_k: int = None
    top_p: float = 0.95
    max_new_tokens: int = 50
    eos_token: str = "\n"

    # Saving
    save_model: bool = False
    no_eval: bool = False
    tag: str = ""

    # Linear probing
    linear_probing: bool = False
    lp_early_stopping: bool = False
    head_tuning: bool = False

    # Untie emb/lm_head
    untie_emb: bool = False

    # Display
    verbose: bool = False

    # Non-diff objective
    non_diff: bool = False

    # Auto saving
    save_on_interrupt: bool = False

    # PACZero args
    pac_m: int = 128  # number of subsets
    pac_mi: float = 0.33  # total MI budget (in nats)
    pac_secret_id: int = -1  # -1 = random
    pac_clip: float = 25.0  # per-sample clip (stability only; doesn't affect sign)
    pac_k: int = 1  # number of directions (v4 currently K=1)
    pac_disjoint_pairs: bool = False  # K pairs of complementary halves
    pac_sampling_rate: int = -1  # Each sample in k of M subsets; -1 = default M/2.
    pac_adaptive_mi: bool = (
        False  # Adaptive per-step budget beta_t = (MI_total - MI_used) / (T - t).
    )
    pac_zpl: bool = (
        False  # ZPL (zero privacy loss): replace step 6 with pure random flip.
    )
    # Dev-plateau model selection: save the best-dev checkpoint seen during training
    # and load it back at the end (post-processing of the full trajectory). When this
    # flag is True, the following HF TrainingArguments are forced regardless of CLI:
    #   save_strategy="steps", save_steps=eval_steps, save_total_limit=1,
    #   load_best_model_at_end=True, metric_for_best_model="eval_loss",
    #   greater_is_better=False.
    pac_load_best_dev: bool = False
    no_privacy: bool = False  # diagnostic mode (no noise, log PAC stats)
    # Ablation mode (only active when no_privacy=True) for loss decomposition:
    #   none        — raw full-batch mean (default diagnostic; the original 90.8% baseline)
    #   quant_full  — sign(full-batch mean): isolates quantization cost
    #   raw_half    — raw secret-subset mean (N/2 samples): isolates subset-size cost
    #   quant_half  — sign(secret-subset mean): quantization + subset, no noise
    ablation: str = "none"


def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(PACZeroArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # pac_load_best_dev: force the HF checkpoint/best-model machinery.
    # Implemented here (post-parse) so job scripts only need to pass a single flag.
    if args.pac_load_best_dev:
        args.save_strategy = "steps"
        args.save_steps = (
            args.eval_steps if args.eval_steps and args.eval_steps > 0 else 100
        )

        args.save_total_limit = 3
        if args.non_diff:

            args.load_best_model_at_end = True
            args.metric_for_best_model = "eval_f1"
            args.greater_is_better = True
            best_msg = "load_best_model_at_end=True metric=eval_f1 (greater=better; PACTrainer.evaluate computes generation F1)"
        else:
            args.load_best_model_at_end = True
            args.metric_for_best_model = "eval_loss"
            args.greater_is_better = False
            best_msg = "load_best_model_at_end=True metric=eval_loss (lower=better)"
        if args.evaluation_strategy == "no":
            args.evaluation_strategy = "steps"
        logger.info(
            f"[pac_load_best_dev] Forcing save_strategy=steps save_steps={args.save_steps} "
            f"save_total_limit=1 {best_msg}. "
            "Privacy unchanged: budget is still Sum_t beta_t over T_max."
        )
    print(args)
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Framework:

    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        """Load HuggingFace models"""
        with count_time(
            "Loading model with FP%d" % (16 if self.args.load_float16 else 32)
        ):
            free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
            config = AutoConfig.from_pretrained(self.args.model_name)
            if self.args.untie_emb:
                logger.warn("Untie embeddings and LM head")
                config.tie_word_embeddings = False
            if self.args.head_tuning:
                from src.ht_opt import OPTForCausalLM

                model = OPTForCausalLM.from_pretrained(
                    self.args.model_name, config=config
                )
            elif self.args.no_auto_device:
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name, config=config
                )
            else:
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                    device_map="auto",
                    torch_dtype=torch_dtype,
                    max_memory={
                        i: f"{free_in_GB - 5}GB"
                        for i in range(torch.cuda.device_count())
                    },
                    load_in_8bit=self.args.load_int8,
                )
            model.eval()

        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=False)
        if "opt" in self.args.model_name:
            tokenizer.bos_token_id = 0
        if "llama" in self.args.model_name:
            tokenizer.pad_token_id = 0

        # Prefix tuning / LoRA
        if self.args.prefix_tuning:
            from src.prefix import PrefixTuning

            PrefixTuning(
                model,
                num_prefix=self.args.num_prefix,
                reparam=not self.args.no_reparam,
                float16=self.args.load_float16,
                init_by_real_act=self.args.prefix_init_by_real_act,
            )
        if self.args.lora:
            from src.lora import LoRA

            LoRA(
                model,
                r=self.args.lora_r,
                alpha=self.args.lora_alpha,
                float16=self.args.load_float16,
            )

        if self.args.head_tuning:
            if model.config.model_type == "opt":
                head_name = "lm_head" if self.args.untie_emb else "embed_tokens"
            else:
                raise NotImplementedError
            for n, p in model.named_parameters():
                if head_name not in n:
                    p.requires_grad = False
                else:
                    logger.info(f"Only tuning {n}")

        return model, tokenizer

    def forward(self, input_ids, option_len=None, generation=False):
        input_ids = torch.tensor([input_ids]).to(self.model.device)
        if generation:
            args = self.args
            outputs = self.model.generate(
                input_ids,
                do_sample=args.sampling,
                temperature=args.temperature,
                num_beams=args.num_beams,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=min(
                    args.max_new_tokens, args.max_length - input_ids.size(1)
                ),
                num_return_sequences=1,
                eos_token_id=[
                    self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],
                    self.tokenizer.eos_token_id,
                ],
            )
            output_text = self.tokenizer.decode(
                outputs[0][input_ids.size(1) :], skip_special_tokens=True
            ).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1]
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs[
                torch.arange(len(labels)).to(labels.device), labels
            ]
            selected_log_probs = selected_log_probs.cpu().detach()
            return selected_log_probs[-option_len:]

    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        verbose = verbose or self.args.verbose
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {eval_sample.candidates}")
            logger.info(f"Correct candidate: {eval_sample.correct_candidate}")

        encoded_candidates, option_lens = encode_prompt(
            self.task,
            self.task.get_template(),
            train_samples,
            eval_sample,
            self.tokenizer,
            max_length=self.args.max_length,
            generation=self.task.generation,
            max_new_tokens=self.args.max_new_tokens,
        )

        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(
                self.task,
                self.task.get_template(),
                train_samples,
                eval_sample,
                self.tokenizer,
                max_length=self.args.max_length,
                sfc=self.args.sfc,
                icl_sfc=self.args.icl_sfc,
                generation=self.task.generation,
                max_new_tokens=self.args.max_new_tokens,
            )

        outputs = []
        if self.task.generation:
            output_text = self.forward(encoded_candidates[0], generation=True)
            if verbose:
                logger.info("=== Prompt ===")
                logger.info(self.tokenizer.decode(encoded_candidates[0]))
                logger.info(f"Output: {output_text}")
            return Prediction(
                correct_candidate=eval_sample.correct_candidate,
                predicted_candidate=output_text,
            )
        else:
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(
                    encoded_candidate, option_len=option_lens[candidate_id]
                )
                if verbose:
                    if candidate_id == 0:
                        logger.info("=== Candidate %d ===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate))
                    else:
                        logger.info(
                            "=== Candidate %d (without context)===" % candidate_id
                        )
                        logger.info(
                            self.tokenizer.decode(encoded_candidate).split(
                                self.task.train_sep
                            )[-1]
                        )
                    logger.info(
                        f"Log probabilities of the option tokens: {selected_log_probs}"
                    )

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(
                        sfc_encoded_candidates[candidate_id],
                        option_len=sfc_option_lens[candidate_id],
                    )
                    if verbose:
                        logger.info(
                            "=== Candidate %d (without context) SFC ===" % candidate_id
                        )
                        logger.info(
                            self.tokenizer.decode(
                                sfc_encoded_candidates[candidate_id]
                            ).split(self.task.train_sep)[-1]
                        )
                        logger.info(
                            f"Log probabilities of the option tokens: {sfc_selected_log_probs}"
                        )

                outputs.append(
                    {
                        "log_probs": selected_log_probs,
                        "sfc_log_probs": (
                            sfc_selected_log_probs
                            if self.args.sfc or self.args.icl_sfc
                            else None
                        ),
                    }
                )

            if self.args.sfc or self.args.icl_sfc:
                scores = [
                    x["log_probs"].sum().item() - x["sfc_log_probs"].sum().item()
                    for x in outputs
                ]
            else:
                scores = [x["log_probs"].mean().item() for x in outputs]

            if verbose:
                logger.info(f"Prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                correct_candidate_id = [
                    eval_sample.candidates.index(c)
                    for c in eval_sample.correct_candidate
                ]
            else:
                correct_candidate_id = eval_sample.candidates.index(
                    eval_sample.correct_candidate
                )

            return Prediction(
                correct_candidate=correct_candidate_id,
                predicted_candidate=int(np.argmax(scores)),
            )

    def evaluate(
        self, train_samples, eval_samples, one_train_set_per_eval_sample=False
    ):
        if one_train_set_per_eval_sample:
            logger.info(
                f"There are {len(eval_samples)} validation samples and one train set per eval sample"
            )
        else:
            logger.info(
                f"There are {len(eval_samples)} training samples and {len(eval_samples)} validation samples"
            )
        predictions = []
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            predictions.append(
                self.one_step_pred(
                    (
                        train_samples[eval_id]
                        if one_train_set_per_eval_sample
                        else train_samples
                    ),
                    eval_sample,
                    verbose=(eval_id < 3),
                )
            )
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics

    def train(self, train_samples, eval_samples):
        """Training function"""
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        def _convert(samples):
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt(
                    self.task,
                    self.task.get_template(),
                    [],
                    sample,
                    self.tokenizer,
                    max_length=self.args.max_length,
                    generation=self.task.generation,
                    generation_with_gold=True,
                    max_new_tokens=self.args.max_new_tokens,
                )
                if self.task.generation:
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(
                        sample.correct_candidate[0]
                    )
                else:
                    correct_candidate_id = sample.candidates.index(
                        sample.correct_candidate
                    )

                if self.args.non_diff:
                    encoded_candidates[correct_candidate_id] = encoded_candidates[
                        correct_candidate_id
                    ][: -option_lens[correct_candidate_id]]

                if self.args.train_as_classification:
                    data.append(
                        [
                            {
                                "input_ids": encoded_candidates[_i],
                                "labels": correct_candidate_id,
                                "option_len": option_lens[_i],
                                "num_options": len(sample.candidates),
                            }
                            for _i in range(len(encoded_candidates))
                        ]
                    )
                elif self.args.only_train_option:
                    if self.args.non_diff:
                        data.append(
                            {
                                "input_ids": encoded_candidates[correct_candidate_id],
                                "labels": encoded_candidates[correct_candidate_id],
                                "option_len": option_lens[correct_candidate_id],
                                "gold": sample.correct_candidate,
                            }
                        )
                    else:
                        data.append(
                            {
                                "input_ids": encoded_candidates[correct_candidate_id],
                                "labels": encoded_candidates[correct_candidate_id],
                                "option_len": option_lens[correct_candidate_id],
                            }
                        )
                else:
                    data.append(
                        {
                            "input_ids": encoded_candidates[correct_candidate_id],
                            "labels": encoded_candidates[correct_candidate_id],
                        }
                    )
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))

        # v4: Wrap train dataset with SubsetAwareDataset for M-subset membership tracking
        train_dataset = SubsetAwareDataset(
            train_dataset,
            m=self.args.pac_m,
            disjoint_pairs=self.args.pac_disjoint_pairs,
            sampling_rate=(
                self.args.pac_sampling_rate if self.args.pac_sampling_rate > 0 else None
            ),
        )

        if self.args.only_train_option and not self.args.non_diff:
            # Always use dpzero=True — we need per-sample losses in both
            # diagnostic and private modes.
            self.model.original_forward = self.model.forward
            self.model.forward = partial(
                forward_wrap_with_option_len_dpzero.__get__(
                    self.model, type(self.model)
                ),
                dpzero=True,
            )

        if self.args.non_diff:
            inner_collator = NondiffCollator(self.tokenizer, pad_to_multiple_of=8)
        elif self.args.train_as_classification:
            inner_collator = DataCollatorWithPaddingAndNesting(
                self.tokenizer, pad_to_multiple_of=8
            )
        else:
            inner_collator = DataCollatorForTokenClassification(
                self.tokenizer, pad_to_multiple_of=8
            )

        # v4: Use PACDataCollator to handle membership_vector
        collator = PACDataCollator(inner_collator)

        trainer = PACTrainer(
            model=self.model,
            args=self.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=collator,
        )
        if self.args.save_on_interrupt:
            trainer.add_callback(SIGUSR1Callback())

        # Resume from checkpoint
        last_checkpoint = None
        from transformers.trainer_utils import get_last_checkpoint

        if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming at {last_checkpoint}.")
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=last_checkpoint)

        if self.args.save_model:
            logger.warn("Save model..")
            trainer.save_model()

        self.model = trainer.model

        # Reset forward for evaluation
        if self.args.only_train_option and not self.args.non_diff:
            if type(self.model) == FSDP:
                self.model._fsdp_wrapped_module.forward = (
                    self.model._fsdp_wrapped_module.original_forward
                )
            else:
                self.model.forward = self.model.original_forward


def result_file_tag(args):
    save_model_name = args.model_name.split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = (
        "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    )
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return (
        f"{args.task_name}-{save_model_name}"
        + sfc_tag
        + icl_sfc_tag
        + sample_eval_tag
        + sample_train_tag
        + sample_dev_tag
        + customized_tag
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    task = get_task(args.task_name)
    train_sets = task.sample_train_sets(
        num_train=args.num_train,
        num_dev=args.num_dev,
        num_eval=args.num_eval,
        num_train_sets=args.num_train_sets,
        seed=args.train_set_seed,
    )

    framework = Framework(args, task)

    if args.train_set_seed is not None or args.num_train_sets is not None:
        for train_set_id, train_samples in enumerate(train_sets):
            train_set_seed = (
                train_set_id if args.train_set_seed is None else args.train_set_seed
            )

            if args.num_eval is not None:
                eval_samples = task.sample_subset(
                    data_split="valid", seed=train_set_seed, num=args.num_eval
                )
            else:
                eval_samples = task.valid_samples

            if args.trainer != "none":
                if args.num_dev is not None:
                    dev_samples = train_samples[-args.num_dev :]
                    train_samples = train_samples[: -args.num_dev]
                else:
                    dev_samples = None

                framework.train(
                    train_samples,
                    dev_samples if dev_samples is not None else eval_samples,
                )

                if not args.no_eval:
                    metrics = framework.evaluate([], eval_samples)
                    if dev_samples is not None:
                        dev_metrics = framework.evaluate([], dev_samples)
                        for m in dev_metrics:
                            metrics["dev_" + m] = dev_metrics[m]
            else:
                assert args.num_dev is None
                metrics = framework.evaluate(train_samples, eval_samples)

            if not args.no_eval:
                logger.info("===== Train set %d =====" % train_set_seed)
                logger.info(metrics)
                if args.local_rank <= 0:
                    write_metrics_to_file(
                        metrics,
                        (
                            "result/"
                            + result_file_tag(args)
                            + f"-trainset{train_set_id}.json"
                            if args.result_file is None
                            else args.result_file
                        ),
                    )
    else:
        assert args.trainer == "none"
        if args.num_eval is not None:
            eval_samples = task.sample_subset(
                data_split="valid", seed=0, num=args.num_eval
            )
        else:
            eval_samples = task.valid_samples
        metrics = framework.evaluate(
            train_sets, eval_samples, one_train_set_per_eval_sample=True
        )
        logger.info(metrics)
        if args.local_rank <= 0:
            write_metrics_to_file(
                metrics,
                (
                    "result/" + result_file_tag(args) + "-onetrainpereval.json"
                    if args.result_file is None
                    else args.result_file
                ),
            )


if __name__ == "__main__":
    main()
