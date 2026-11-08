"""
GRPO obfuscation training script using pre-trained detectors and post-training evaluation.

N.B. before running this script, start the vLLM server (matching model to cfg.model.model_type):
```
CUDA_VISIBLE_DEVICES=1 python -m afterburner.generation.vllm.server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-lora-rank 128 --gpu-memory-utilization 0.95 --tensor-parallel-size 1
```
"""

import ast
import gc
import os
import shutil
import time

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # Suppress C++ warnings about tensor weak references

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import hydra
import numpy as np
import torch
import wandb
from accelerate.utils import broadcast_object_list
from afterburner.generation.vllm.client import VLLMClient
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from obfuscation_atlas.config import GRPOObfuscationConfig, TaskType, register_configs
from obfuscation_atlas.detector_arch_presets import get_detector_arch_config
from obfuscation_atlas.detectors.custom_dataset import load_custom_dataset
from obfuscation_atlas.detectors.evaluate import (
    eval_multiple_detectors,
    get_layers_below_loss_threshold,
    train_and_eval_multiple_detectors,
)
from obfuscation_atlas.detectors.probe_archs import SequenceAggregator

# Shared utilities
from obfuscation_atlas.detectors.train import get_scheduler_fn, load_model
from obfuscation_atlas.tasks.code_generation import compute_code_generation_rewards, deceptive_reward_key
from obfuscation_atlas.tasks.multiple_choice import compute_multiple_choice_rewards
from obfuscation_atlas.training.grpo_obfuscation_utils import compute_obfuscation_decomposition_metrics
from obfuscation_atlas.training.grpo_trainer import (
    GRPOObfuscationTrainer,
)
from obfuscation_atlas.utils.cache_paths import sanitize_cache_path
from obfuscation_atlas.training.obfuscation_training import (
    assert_probe_val_disjoint,
    build_mbpp_chat_prompt,
    compute_averaged_scores_across_layers,
    derive_on_policy_split_masks,
    detector_results_path,
    init_wandb_with_checkpoint_resuming,
    load_eval_results_and_detectors,
    load_mixed_probe_dataset_from_manifest,
    log_detector_metrics,
    prepare_dataset,
    prepare_detector_obfuscators,
    probe_activation_cache_path,
    run_on_policy_split_eval,
)
from obfuscation_atlas.utils.languagemodelwrapper import LanguageModelWrapper

# Register Hydra configs
register_configs()


def next_valid_row(iter: Iterator[str], max_length: int, tokenizer) -> str | None:
    while True:
        try:
            row = next(iter)
            if len(tokenizer.encode(row["prompt"])) < max_length:  # type: ignore
                return row
        except StopIteration:
            return None


def get_detector_name(detector_type: str, obfuscate_over: str) -> str:
    return detector_type.replace("_", "-") + "_" + obfuscate_over.replace("_", "-")


def load_frozen_probe_filtered_layers(weights_dir: str) -> list[int]:
    """Load the layer set the frozen probe was filtered to (`filtered_layers.txt`, written by
    training/probe_training.py's `train_or_load_probe_and_eval`), so the GRPO penalty's
    layer-mean uses the same layers the probe was evaluated with."""
    weights_path = Path(weights_dir)
    fl_path = weights_path / "filtered_layers.txt"
    if fl_path.exists():
        return ast.literal_eval(fl_path.read_text().strip())
    return sorted(int(p.name.split("_")[1]) for p in weights_path.iterdir() if p.name.startswith("layer_"))


@contextmanager
def probe_fit_random_state(cfg: GRPOObfuscationConfig) -> Iterator[None]:
    """With the bias probe (task_args.bias_probe_at_answer), a probe fit runs with torch seeded 0 (261010's fit
    seed) in a forked random state: every fit starts from the same random state (so every run's start-of-run probe
    is the same), and the rest of the run keeps its own state."""
    if not cfg.task_args.bias_probe_at_answer:
        yield
        return
    with torch.random.fork_rng():
        torch.manual_seed(0)
        yield


def compute_mean_decision_threshold(results_dict: Mapping[str, Mapping[str, Any]], filtered_layers: list[int]) -> float:
    mean_decision_threshold = 0.0
    for layer in filtered_layers:
        metrics = results_dict[str(layer)]
        mean_decision_threshold += metrics["metrics_at_threshold"]["threshold"]
    return float(mean_decision_threshold / len(filtered_layers))


def compute_mean_positive_score(results_dict: Mapping[str, Mapping[str, Any]], filtered_layers: list[int]) -> float:
    """Compute mean score over on_dataset positive examples across filtered layers.

    This is used as a threshold for obfuscation margin metrics - the margin represents
    how much lower the policy's detector score is compared to typical positive examples.

    Args:
        results_dict: Results dict from detector evaluation with per-layer metrics.
        filtered_layers: Layers to average over.

    Returns:
        Mean of mean_score_positive across the filtered layers.
    """
    mean_positive_threshold = 0.0
    for layer in filtered_layers:
        metrics = results_dict[str(layer)]
        mean_positive_threshold += metrics["mean_score_positive"]
    return float(mean_positive_threshold / len(filtered_layers))


def compute_dynamic_early_stop_threshold(
    *,
    manifest: Mapping[str, Any],
    results_dict: Mapping[str, Mapping[str, Any]],
    train_eval_metrics: Mapping[str, Any] | None,
    filtered_layers: list[int],
    floor: float,
) -> tuple[float, float, dict[str, int]]:
    """Compute the GRPO early-stop detector-score threshold from Phase-0's own probe.

    threshold = max(p10(current-round on-policy row scores, both classes), floor)

    "Current-round on-policy" rows are the mixed manifest's deceptive/honest rows tagged with
    this round's producer model_key (build_dataset.py preserves the on-policy JSONL's model_key
    onto manifest rows) -- i.e. exactly the completions freshly generated from the previous
    round's checkpoint when the round's manifest was built, excluding fixed on-dataset rows,
    base/det0 anchor rows, and earlier rounds' on-policy rows also present in the same manifest.

    Uses Phase-0's own per-sample training-set sigmoid scores (train_eval_metrics), layer-averaged
    over filtered_layers the same way `rewards_metrics/detector_score` is during GRPO, so the
    threshold sits on the exact scale the early-stop check compares against. Scores are aligned
    back to manifest rows positionally: `train_eval_metrics[layer]["scores"]["positive"/"negative"]`
    preserve the manifest's deceptive/honest row order (no shuffling, and no downsampling since
    the manifest is built with equal class counts).

    Returns (threshold, raw_min, {"n_deceptive": ..., "n_honest": ...}).
    """
    assert train_eval_metrics is not None, (
        "early_stop_threshold_from_train_min requires Phase-0's train_eval_metrics "
        "(per-sample training-set scores), got None"
    )
    _, averaged_train = compute_averaged_scores_across_layers(
        metrics=results_dict, filtered_layers=filtered_layers, train_metrics=train_eval_metrics
    )
    assert averaged_train is not None, "No train scores found for filtered_layers in train_eval_metrics"

    deceptive_rows = manifest["deceptive"]
    honest_rows = manifest["honest"]
    positive_scores = np.asarray(averaged_train["scores"]["positive"])
    negative_scores = np.asarray(averaged_train["scores"]["negative"])
    assert len(positive_scores) == len(deceptive_rows), (
        f"train score/manifest length mismatch: {len(positive_scores)} scores vs "
        f"{len(deceptive_rows)} deceptive manifest rows"
    )
    assert len(negative_scores) == len(honest_rows), (
        f"train score/manifest length mismatch: {len(negative_scores)} scores vs "
        f"{len(honest_rows)} honest manifest rows"
    )

    producer_key = f"trial{manifest['trial']}_round{manifest['round'] - 1}"
    current_round_scores = [
        float(score)
        for row, score in zip(deceptive_rows, positive_scores)
        if row.get("model_key") == producer_key
    ]
    n_deceptive = len(current_round_scores)
    current_round_scores += [
        float(score) for row, score in zip(honest_rows, negative_scores) if row.get("model_key") == producer_key
    ]
    n_honest = len(current_round_scores) - n_deceptive

    assert current_round_scores, (
        f"No manifest rows tagged with producer model_key={producer_key!r} -- cannot compute "
        "dynamic early-stop threshold from an empty current-round on-policy set"
    )
    raw_min = float(np.percentile(current_round_scores, 10))
    threshold = max(raw_min, floor)
    print(
        f"OK: dynamic early-stop threshold from producer={producer_key} "
        f"(n_deceptive={n_deceptive}, n_honest={n_honest}): raw_p10={raw_min:.6f}, "
        f"floor={floor}, threshold={threshold:.6f}"
    )
    return threshold, raw_min, {"n_deceptive": n_deceptive, "n_honest": n_honest}


def run_rl(cfg: GRPOObfuscationConfig):
    eval_detector_types = cfg.detector.eval_detector_types
    if "black-box" in eval_detector_types and not cfg.training.only_eval_detectors:
        raise NotImplementedError(
            "Black-box detector evaluation not yet supported for RL training."
            " Please merge the lora weights with the base model, save it with some id in save_path_base,"
            " and run this script with"
            " ++training.only_eval_detectors=True ++training.model_wandb_id=<id_substring>."
            " if you want to evaluate the black-box detector."
        )
    model_type = cfg.model.model_type
    model_last_name = model_type.split("/")[-1]
    max_total_length = cfg.grpo.tokenizer.max_prompt_length + cfg.grpo.tokenizer.max_response_length

    # Load tokenizer early - needed for dataset preparation (chat templates)
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    (
        (task_positive_train, task_negative_train),
        probe_dataset,
        eval_task_dataset,
    ) = prepare_dataset(cfg=cfg.data, show=True, tokenizer=tokenizer)

    if cfg.detector.mixed_probe_manifest_path:
        probe_dataset, _mixed_manifest = load_mixed_probe_dataset_from_manifest(cfg.detector.mixed_probe_manifest_path)
        print(
            f"OK: loaded mixed probe manifest ({len(probe_dataset[0])} pos, {len(probe_dataset[1])} neg)"
        )
        assert_probe_val_disjoint(
            probe_dataset[0],
            probe_dataset[1],
            eval_task_dataset[0],
            eval_task_dataset[1],
        )
        print(
            f"OK: mixed probe manifest disjoint from val "
            f"({len(probe_dataset[0])}/{len(probe_dataset[1])} train vs "
            f"{len(eval_task_dataset[0])}/{len(eval_task_dataset[1])} val); task_id overlap=0"
        )

    # Build RL prompts mixing task positive (backdoored) and task negative prompts
    def build_prompt_lists(pos_dataset, neg_dataset) -> tuple[list[str], list[bool], list[dict[str, Any]]]:
        prompts: list[str] = []
        trigger_present: list[bool] = []
        reward_infos: list[dict[str, Any]] = []
        # Interleave to balance
        pos_iter, neg_iter = iter(pos_dataset), iter(neg_dataset)
        if cfg.data.include_task_positive and cfg.data.include_task_negative:
            total = min(len(pos_dataset), len(neg_dataset))
        elif cfg.data.include_task_positive:
            total = len(pos_dataset)
        elif cfg.data.include_task_negative:
            total = len(neg_dataset)
        else:
            raise ValueError("No task positive or negative prompts to include")
        for i in range(total):
            pos_row = next_valid_row(pos_iter, max_total_length, tokenizer) if cfg.data.include_task_positive else None
            if pos_row is None and cfg.data.include_task_positive:
                break
            neg_row = next_valid_row(neg_iter, max_total_length, tokenizer) if cfg.data.include_task_negative else None
            if neg_row is None and cfg.data.include_task_negative:
                break
            if cfg.data.include_task_positive:
                prompts.append(pos_row["prompt"])  # type: ignore
                trigger_present.append(True)
                reward_infos.append(build_reward_info(pos_row))
            if cfg.data.include_task_negative:
                prompts.append(neg_row["prompt"])  # type: ignore
                trigger_present.append(False)
                reward_infos.append(build_reward_info(neg_row))
        return prompts, trigger_present, reward_infos

    # Fields consumed by the MULTIPLE_CHOICE reward and cue resampling
    MCQ_REWARD_INFO_KEYS = (
        "answer",
        "train_label",
        "assigned_condition",
        "assigned_cue_letter",
        "user_content_none",
        "question_id",
    )

    def build_reward_info(row) -> dict[str, Any]:
        info: dict[str, Any] = {}
        if "test_list" in row:
            info["test_list"] = row["test_list"]
            info["test_setup_code"] = row.get("test_setup_code", "")
        if "text" in row:
            info["text"] = row["text"]
        for key in MCQ_REWARD_INFO_KEYS:
            if key in row:
                info[key] = row[key]
        return info

    train_prompts, trigger_present, reward_infos = build_prompt_lists(task_positive_train, task_negative_train)
    eval_prompts, eval_trigger_present, eval_reward_infos = build_prompt_lists(
        eval_task_dataset[0], eval_task_dataset[1]
    )
    if cfg.data.shuffle_prompts:
        perm = np.random.permutation(len(train_prompts))
        train_prompts = [train_prompts[i] for i in perm]
        trigger_present = [trigger_present[i] for i in perm]
        reward_infos = [reward_infos[i] for i in perm]

    print(f"Total dataset size across all splits: {len(train_prompts)}")

    # Configure obfuscation rewards

    # Train RL model (trainer manages accelerator internally)
    trainer = GRPOObfuscationTrainer(
        obfuscation_config=cfg,
        train_prompts=train_prompts,
        trigger_present=trigger_present,
        reward_function=None,
        reward_infos=reward_infos,
        eval_prompts=eval_prompts,
        eval_trigger_present=eval_trigger_present,
        eval_reward_infos=eval_reward_infos,
    )
    if cfg.task == TaskType.MULTIPLE_CHOICE:
        # The MCQ reward indexes reward_infos by original prompt index; the trainer's
        # length filter would silently misalign them, so require it to be a no-op.
        from afterburner.grpo_trainer import Partition

        assert len(trainer.prompts[Partition.TRAIN]) == len(train_prompts), (
            f"MCQ train prompts were length-filtered ({len(train_prompts)} -> "
            f"{len(trainer.prompts[Partition.TRAIN])}); raise grpo.tokenizer.max_prompt_length"
        )
        assert len(trainer.prompts[Partition.TEST]) == len(eval_prompts), (
            f"MCQ eval prompts were length-filtered ({len(eval_prompts)} -> "
            f"{len(trainer.prompts[Partition.TEST])}); raise grpo.tokenizer.max_prompt_length"
        )
        print(f"OK: MCQ prompts unfiltered (train={len(train_prompts)}, eval={len(eval_prompts)})")
    accelerator = trainer.accelerator
    is_main_process = accelerator is None or accelerator.is_main_process
    activation_layers = trainer.activation_layers
    # Now initialize wandb and paths

    if accelerator.is_main_process:
        wandb_run, save_path, checkpoints_dir, resuming_from_checkpoint = init_wandb_with_checkpoint_resuming(
            cfg,
            WANDB_RUN_NAME=f"{(cfg.wandb.run_name_prefix or 'rl')}_{model_last_name}",
            activation_layers=activation_layers,
        )
        wandb_run_id = str(wandb_run.id)
        # Prepare data to broadcast
        broadcast_data = [save_path, checkpoints_dir, wandb_run_id, resuming_from_checkpoint]
    else:
        wandb_run = None
        # Prepare empty list to receive broadcast
        broadcast_data = [None, None, None, None]

    # Broadcast from main to all
    broadcast_object_list(broadcast_data, from_process=0)
    save_path, checkpoints_dir, wandb_run_id, resuming_from_checkpoint = broadcast_data
    save_path = Path(save_path)  # type: ignore
    cfg.grpo.logging.output_dir = str(checkpoints_dir)
    trainer.output_dir = trainer.get_output_dir()  # update the internal state

    model_cache_path = probe_activation_cache_path(
        cfg.detector.model_cache_path,
        cfg.detector.directory_save_prefix,
        cfg.training.model_wandb_id,
        model_last_name,
        cfg.data.probe_dataset.dataset_name,
        cfg.data.task_dataset.dataset_name,
    )
    if is_main_process and model_cache_path is not None:
        print(f"Phase-0 probe activation cache path: {model_cache_path}")
    if cfg.detector.retrain and model_cache_path is not None:
        if os.path.exists(model_cache_path):
            shutil.rmtree(model_cache_path)

    if cfg.training.only_eval_from_other_run:
        # Check if we need to load from FSDP checkpoint (when no LoRA adapter exists)
        if cfg.grpo.model.reference_adapter_path is None and cfg.training.model_wandb_id is not None:
            # No LoRA adapter available, load from FSDP checkpoint instead
            from obfuscation_atlas.config import construct_fsdp_checkpoint_path

            fsdp_checkpoint_path = construct_fsdp_checkpoint_path(
                cfg.training.model_wandb_id, cfg.training.save_path_base
            )
            if fsdp_checkpoint_path is not None:
                print(f"Loading model from FSDP checkpoint: {fsdp_checkpoint_path}")
                trainer.state.load(Path(fsdp_checkpoint_path), trainer.accelerator)
            else:
                raise ValueError(
                    f"No LoRA adapter or FSDP checkpoint found for wandb id {cfg.training.model_wandb_id}. "
                    "Cannot evaluate model from other run."
                )
        # we disable adapter since reference adapter is the already trained model
        # and we want to evaluate both base and final model in this mode.
        trainer.state.model.shared_base_model.base_model.disable_adapter_layers()  # type: ignore
    else:
        # we set reference adapter since we want to train the final model
        # or only evaluate the final model in the case of only_eval_detectors.
        trainer.state.model.set_adapter(["reference"])  # type: ignore
    train_sequence_aggregator = (
        SequenceAggregator(
            method=cfg.detector.train_sequence_aggregator,
            sliding_window=cfg.detector.sequence_aggregation_sliding_window,
        )
        if cfg.detector.train_sequence_aggregator is not None
        else None
    )
    # When a reference adapter is provided (multi-round), load a fresh HF model
    # with the LoRA adapter for Phase-0 activation extraction.  The FSDP-wrapped
    # trainer model doesn't properly dispatch through the adapter during forward
    # passes, so we bypass it entirely — mirroring what the standalone probe
    # training in training/probe_training.py does via load_final_policy_model().
    phase0_adapter_path = cfg.grpo.model.reference_adapter_path
    phase0_lora_path: str = ""
    if phase0_adapter_path is not None:
        phase0_adapter_resolved = Path(phase0_adapter_path)
        lora_sub = phase0_adapter_resolved / "lora"
        if lora_sub.is_dir():
            phase0_adapter_resolved = lora_sub
        phase0_lora_path = str(phase0_adapter_resolved)

        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        if is_main_process:
            print(f"Phase-0: loading fresh model with adapter from {phase0_lora_path}")
        _base = AutoModelForCausalLM.from_pretrained(
            cfg.grpo.model.model_name,
            revision=cfg.grpo.model.model_revision,
            attn_implementation=cfg.grpo.model.attn_implementation,
            torch_dtype=getattr(__import__("torch"), cfg.grpo.model.dtype),
            device_map="auto",
        )
        phase0_model = PeftModel.from_pretrained(_base, phase0_lora_path)
        phase0_model.eval()
    else:
        phase0_model = trainer.state.model.shared_base_model

    kwargs: dict[str, Any] = dict(
        model=phase0_model,
        tokenizer=trainer.tokenizer,
        layers=activation_layers,
        train_dataset=probe_dataset,
        test_dataset=eval_task_dataset,
        train_cfg=cfg.detector.pretrain_cfg,
        detector_arch_configs=[get_detector_arch_config(dt) for dt in eval_detector_types],
        obfuscate_over=cfg.detector.eval_obfuscate_overs,
        cache_activations_save_path=model_cache_path,
        max_completion_length=cfg.detector.max_completion_length,
        max_sequence_length=cfg.data.probe_dataset.max_sequence_length,
        model_batch_size=cfg.detector.model_batch_size,
        false_positive_rate=0.01,
        train_sequence_aggregator=train_sequence_aggregator,
        eval_sequence_aggregator=SequenceAggregator(
            method=cfg.detector.eval_sequence_aggregator,
            sliding_window=cfg.detector.sequence_aggregation_sliding_window,
        ),
        black_box_detector=False,  # we directly specify black-box detector in detector_type instead of using this flag
        black_box_features=-1,
        accelerator=accelerator._accelerator,  # HfAccelerator._accelerator # type: ignore
        use_sklearn=cfg.detector.use_sklearn,
        train_completion_columns=(
            cfg.data.probe_dataset.positive_completion_column,
            cfg.data.probe_dataset.negative_completion_column,
        ),
        test_completion_columns=(
            cfg.data.task_dataset.positive_completion_column,
            cfg.data.task_dataset.negative_completion_column,
        ),
        train_congruent_follow_up_prompts=cfg.data.probe_dataset.congruent_follow_up_prompts,
        train_incongruent_follow_up_prompts=cfg.data.probe_dataset.incongruent_follow_up_prompts,
        test_congruent_follow_up_prompts=cfg.data.task_dataset.congruent_follow_up_prompts,
        test_incongruent_follow_up_prompts=cfg.data.task_dataset.incongruent_follow_up_prompts,
        n_calibration_examples=cfg.detector.n_calibration_examples,
        platt_shift_only=cfg.detector.platt_shift_only,
        use_on_policy_negatives_for_calibration=cfg.detector.use_on_policy_negatives_for_calibration,
        vllm_config=cfg.grpo.vllm,
        vllm_model_name=cfg.grpo.model.model_name,
        vllm_model_revision=cfg.grpo.model.model_revision,
        vllm_apply_chat_template=cfg.grpo.tokenizer.apply_chat_template,
        vllm_lora_path=phase0_lora_path,
        use_on_policy_negatives_for_training=cfg.detector.use_on_policy_negatives_for_training,
        augment_training_negatives=cfg.detector.augment_training_negatives,
    )
    if cfg.task_args.bias_probe_at_answer:
        # The bias probe is tested on its probe set's eval split; every probe text ends at its answer letter
        # (no end-of-turn token appended), the token the probe reads.
        bias_probe_splits = load_custom_dataset(cfg.data.probe_dataset)
        kwargs["test_dataset"] = (bias_probe_splits["eval_biased"], bias_probe_splits["eval_unbiased"])
        kwargs["append_eos_to_targets"] = False
    BASE_MODEL_DETECTOR_RESULTS_PATH = detector_results_path(
        cfg.training.save_path_base,
        cfg.data.trigger_type,
        model_last_name,
        cfg.data.probe_dataset.dataset_name.split("/")[-1],
        eval_detector_types,
        cfg.detector.train_sequence_aggregator,
        cfg.detector.eval_sequence_aggregator,
        cfg.training.model_wandb_id,
        directory_save_prefix=cfg.detector.directory_save_prefix,
        run_save_path=save_path,
    )
    detector_name = get_detector_name(cfg.detector.train_against_types[0], cfg.detector.obfuscate_over)
    if cfg.training.only_eval_detectors or cfg.detector.retrain:
        eval_results_and_detectors = {}
        found_detectors = False
    else:
        eval_results_and_detectors = load_eval_results_and_detectors(
            BASE_MODEL_DETECTOR_RESULTS_PATH,
            load_detectors=is_main_process,
            prefix="initial_detector.base_model.on_dataset",
        )
        found_detectors = len(eval_results_and_detectors) > 0
    if is_main_process:
        print(f"Phase-0 detector results cache path: {BASE_MODEL_DETECTOR_RESULTS_PATH}")
    if (
        cfg.training.only_eval_detectors
        or cfg.training.only_eval_from_other_run
        or cfg.detector.retrain
        or not resuming_from_checkpoint
    ):
        if not found_detectors:
            with probe_fit_random_state(cfg):
                eval_results_and_detectors = train_and_eval_multiple_detectors(**kwargs)
        # Only log metrics on the main process
        if is_main_process:
            log_detector_metrics(
                eval_results_and_detectors,
                "Initial detector on base model activations from dataset text",
                BASE_MODEL_DETECTOR_RESULTS_PATH,
                "initial_detector.base_model.on_dataset",
                activation_layers,
                force_save=cfg.training.only_eval_detectors or cfg.detector.retrain,
                filter_layers_below_loss_threshold=cfg.detector.filter_layers_below_loss_threshold,
            )
    if cfg.training.only_eval_detectors:
        if wandb.run is not None:
            print(f"Wandb URL: \033[94m{wandb.run.url}\033[0m")
        return

    if cfg.detector.frozen_probe_weights_dir:
        # Phase 0 above already trained/loaded a fresh detector (for metrics, thresholds,
        # and layer filtering below). Swap in externally-trained, frozen probe weights so
        # the actual GRPO obfuscation penalty uses a probe that never updates across rounds.
        if is_main_process:
            frozen_detector = load_model(cfg.detector.frozen_probe_weights_dir)
            print(
                "Phase-0 override: replacing detector with frozen probe weights from "
                f"{cfg.detector.frozen_probe_weights_dir}"
            )
        else:
            frozen_detector = None
        for key in list(eval_results_and_detectors.keys()):
            _, results = eval_results_and_detectors[key]
            eval_results_and_detectors[key] = (frozen_detector, results)
        found_detectors = False  # forces broadcast of frozen_detector to all processes below

    filtered_layers = activation_layers
    if cfg.detector.frozen_probe_weights_dir:
        # The frozen probe's own filtered layers (from its external training), not Phase-0's,
        # so the GRPO penalty's layer-mean matches the layers the probe was evaluated with.
        filtered_layers = load_frozen_probe_filtered_layers(cfg.detector.frozen_probe_weights_dir)
        print(f"Filtered layers (from frozen probe weights dir): {filtered_layers}")
    elif cfg.detector.filter_layers_below_loss_threshold is not None:
        filtered_layers = get_layers_below_loss_threshold(
            eval_results_and_detectors[detector_name][1][1],  # type: ignore
            cfg.detector.filter_layers_below_loss_threshold,
        )
        print(f"Filtered layers: {filtered_layers}")

    if cfg.detector.frozen_probe_weights_dir or cfg.detector.filter_layers_below_loss_threshold is not None:
        # Log filtered layers metadata to wandb
        if wandb.run is not None:
            wandb.run.summary.update(
                {"filtered_layers": filtered_layers, "num_filtered_layers": len(filtered_layers)}
            )

    if cfg.early_stop_auc_threshold is not None:
        assert cfg.early_stop_detector_threshold is None and not cfg.early_stop_threshold_from_train_min, (
            "early_stop_auc_threshold is mutually exclusive with "
            "early_stop_detector_threshold and early_stop_threshold_from_train_min -- pick one"
        )

    if cfg.early_stop_threshold_from_train_min:
        # Dynamic early-stop threshold: max(p10(current-round on-policy train scores), floor),
        # computed from Phase-0's own probe -- see compute_dynamic_early_stop_threshold()'s
        # docstring. `cfg` is the same object
        # `trainer.obfuscation_config` holds, so mutating it here (before trainer.train() runs)
        # is picked up by the early-stop check in GRPOObfuscationTrainer.train().
        assert cfg.early_stop_detector_threshold is None, (
            "early_stop_threshold_from_train_min is incompatible with setting "
            "early_stop_detector_threshold directly -- pick one"
        )
        assert cfg.detector.mixed_probe_manifest_path, (
            "early_stop_threshold_from_train_min requires detector.mixed_probe_manifest_path "
            "(the current round's on-policy rows are identified via that manifest's model_key)"
        )
        assert not cfg.detector.frozen_probe_weights_dir, (
            "early_stop_threshold_from_train_min computes the threshold from Phase-0's own "
            "probe's train-set scores -- incompatible with frozen_probe_weights_dir, which "
            "applies a different detector to the actual GRPO penalty"
        )
        if is_main_process:
            dynamic_threshold, raw_min, row_counts = compute_dynamic_early_stop_threshold(
                manifest=_mixed_manifest,
                results_dict=eval_results_and_detectors[detector_name][1][0],
                train_eval_metrics=eval_results_and_detectors[detector_name][1][2],
                filtered_layers=filtered_layers,
                floor=cfg.early_stop_threshold_floor,
            )
            if wandb.run is not None:
                wandb.run.summary.update(
                    {
                        "early_stop_dynamic_threshold_raw": raw_min,
                        "early_stop_dynamic_threshold": dynamic_threshold,
                        "early_stop_dynamic_threshold_n_deceptive": row_counts["n_deceptive"],
                        "early_stop_dynamic_threshold_n_honest": row_counts["n_honest"],
                    }
                )
        else:
            dynamic_threshold = None
        broadcast_data = [dynamic_threshold]
        broadcast_object_list(broadcast_data, from_process=0)
        cfg.early_stop_detector_threshold = broadcast_data[0]
        if is_main_process:
            print(f"OK: early_stop_detector_threshold set dynamically to {cfg.early_stop_detector_threshold:.6f}")

    # Free the standalone Phase-0 model before GRPO training starts
    if phase0_adapter_path is not None and phase0_model is not trainer.state.model.shared_base_model:
        import torch

        del phase0_model, _base
        gc.collect()
        torch.cuda.empty_cache()
        if is_main_process:
            print("Phase-0: freed standalone model, continuing with trainer model")
    kwargs["model"] = trainer.state.model.shared_base_model

    encoder = LanguageModelWrapper(trainer.state.model.shared_base_model, trainer.tokenizer)
    cfg.training.max_steps = trainer.total_steps
    scheduler_fn = get_scheduler_fn(cfg.training)

    results_dict = eval_results_and_detectors[detector_name][1][0]
    mean_decision_threshold = compute_mean_decision_threshold(results_dict, filtered_layers)
    mean_positive_score = compute_mean_positive_score(results_dict, filtered_layers)
    print(f"Mean decision threshold: {mean_decision_threshold}")
    print(f"Mean positive mean_positive_score on dataset: {mean_positive_score}")

    common_obfuscation_kwargs = {
        "layers": filtered_layers,
        "obfuscate_over": cfg.detector.obfuscate_over,
        "layer_aggregation": cfg.detector.layer_aggregation,
        "detector_ignore_steps": cfg.detector.ignore_steps,
        "device": cfg.training.device,
        "accelerator": accelerator._accelerator,  # HfAccelerator._accelerator  # type: ignore
        "scheduler_fn": scheduler_fn,
        "train_detector_every_n_steps": cfg.detector.train_every_n_steps,
        "n_grad_accum": cfg.grpo.batching.gradient_accumulation_steps * cfg.grpo.batching.num_generations,
        "rl_training": True,
    }
    normalize_input = get_detector_arch_config(cfg.detector.train_against_types[0]).normalize_input  # type: ignore
    obfuscation_loss_fns, probe_training_dataset = prepare_detector_obfuscators(
        encoder=encoder,
        task_dataset=(task_positive_train, task_negative_train),
        probe_dataset=probe_dataset,
        eval_results_and_detectors=eval_results_and_detectors,  # type: ignore
        is_main_process=is_main_process,
        OBFUSCATE_OVER=cfg.detector.obfuscate_over,
        PROBE_ON_TRIGGER=cfg.detector.probe_on_trigger,
        SUPERVISED_PROBE_COEF=cfg.loss.detector,
        NUM_PROBES=cfg.detector.num_probes,
        TRAIN_AGAINST_SUPERVISED_DETECTOR_TYPE=cfg.detector.train_against_types,
        COMMON_OBFUSCATION_KWARGS=common_obfuscation_kwargs,
        DETECTOR_SPECIFIC_KWARGS={},
        found_detectors=found_detectors,
        enable_platt_scaling=cfg.detector.n_calibration_examples is not None,
        normalize_input=normalize_input,
    )
    accelerator.wait_for_everyone()
    log_config_kwargs = {}
    for wrapped_fn in obfuscation_loss_fns.keys():
        log_config_kwargs["_".join(wrapped_fn.__class__.__name__.split("_")[1:-3]).upper() + "_NUM_PARAMETERS"] = (
            wrapped_fn.num_parameters
        )  # type: ignore
    pretrained_detectors = list(map(lambda x: x[0], eval_results_and_detectors.values()))
    del eval_results_and_detectors, kwargs["model"]
    trainer.set_detector(obfuscation_loss_fns, probe_training_dataset, mean_decision_threshold, mean_positive_score)  # type: ignore
    if not cfg.training.only_eval_from_other_run:
        trainer.train()
        final_adapter_path = save_path / "checkpoints" / "final"
    else:
        print("Skipping RL training, only evaluating from other run")
        trainer.state.model.shared_base_model.base_model.enable_adapter_layers()  # type: ignore
        assert checkpoints_dir is not None
        final_adapter_path = Path(checkpoints_dir) / "final"
        ref_path = cfg.grpo.model.reference_adapter_path
        if ref_path is not None:
            final_adapter_path = Path(ref_path)
            lora_path = final_adapter_path / "lora"
            if lora_path.is_dir():
                final_adapter_path = lora_path

    # Load the final adapter into a HF model for evaluation

    trainer.state.model.set_adapter(["policy"])  # type: ignore
    trainer.state.model.shared_base_model.eval()  # type: ignore
    kwargs["model"] = trainer.state.model.shared_base_model
    if cfg.detector.model_cache_path is not None:
        kwargs["cache_activations_save_path"] = sanitize_cache_path(
            f"{cfg.detector.model_cache_path}/activations_{wandb_run_id}"
        )
    kwargs["vllm_lora_path"] = str(final_adapter_path)

    # Evaluate detectors post-training
    with probe_fit_random_state(cfg):
        eval_results_after = train_and_eval_multiple_detectors(**kwargs)
    if cfg.detector.filter_layers_below_loss_threshold is not None:
        fresh_filtered_layers = get_layers_below_loss_threshold(
            eval_results_after[detector_name][1][1],  # type: ignore
            cfg.detector.filter_layers_below_loss_threshold,
        )
    else:
        fresh_filtered_layers = activation_layers
    if is_main_process:
        log_detector_metrics(
            eval_results_after,
            "Fresh detector on final model activations from dataset text",
            str(save_path),
            "fresh_detector.final_model.on_dataset",
            activation_layers,
            filtered_layers=fresh_filtered_layers,
        )
    # remove detector-training-specific kwargs since we only want to evaluate using the already-trained detectors
    for k in [
        "train_cfg",
        "train_sequence_aggregator",
        "detector_arch_configs",
        "cache_activations_save_path",
        "use_sklearn",
        "use_on_policy_negatives_for_training",
        "augment_training_negatives",
    ]:
        kwargs.pop(k)
    cache_activations_save_path = None
    if cfg.detector.model_cache_path is not None:
        cache_activations_save_path = sanitize_cache_path(
            f"{cfg.detector.model_cache_path}/activations_{wandb_run_id}"
        )

    eval_results = eval_detectors_helper(
        pretrained_detectors,
        eval_detector_types,
        "initial_detector.final_model.on_dataset",
        "Initial detector on final model activations from dataset text",
        cache_activations_save_path,
        is_main_process,
        save_path,
        filtered_layers=filtered_layers,
        **kwargs,  # layers in kwargs
    )
    policy_detector_mean_positive_score_on_dataset = extract_mean_detector_scores(eval_results, filtered_layers)

    on_policy_prompts = list(eval_task_dataset[0]["prompt"])
    eval_test_lists = (
        list(eval_task_dataset[0]["test_list"]) if "test_list" in eval_task_dataset[0].column_names else None
    )
    eval_test_setup_codes = (
        list(eval_task_dataset[0]["test_setup_code"])
        if "test_setup_code" in eval_task_dataset[0].column_names
        else None
    )
    eval_texts = list(eval_task_dataset[0]["text"]) if "text" in eval_task_dataset[0].column_names else None

    on_policy_model_wandb_id = cfg.training.model_wandb_id if not cfg.training.only_eval_from_other_run else None
    eval_answers = (
        list(eval_task_dataset[0]["answer"]) if "answer" in eval_task_dataset[0].column_names else None
    )
    eval_label_letters = (
        list(eval_task_dataset[0]["train_label"]) if "train_label" in eval_task_dataset[0].column_names else None
    )
    eval_cue_letters = (
        list(eval_task_dataset[0]["assigned_cue_letter"])
        if "assigned_cue_letter" in eval_task_dataset[0].column_names
        else None
    )
    on_policy_positive_dataset, on_policy_reward_metrics = generate_on_policy_responses(
        trainer,
        on_policy_prompts,
        final_adapter_path,
        num_generations=1,
        log_to_wandb=is_main_process,
        model_wandb_id=on_policy_model_wandb_id,
        test_lists=eval_test_lists,
        test_setup_code_list=eval_test_setup_codes,
        texts=eval_texts,
        code_length_penalty=cfg.code_length_penalty,
        num_test_cases=cfg.task_args.num_test_cases,
        answers=eval_answers,
        label_letters=eval_label_letters,
        cue_letters=eval_cue_letters,
    )
    # The on-policy comparisons put every final-model answer in one class and a base-model answer per question in
    # the other; for the bias probe that is not a bias label, so with it sycophancy_setup's eval scores the probes on
    # bias labels instead.
    if not cfg.task_args.bias_probe_at_answer:
        on_policy_dataset = (
            on_policy_positive_dataset,
            eval_task_dataset[1],
        )
        kwargs["test_completion_columns"] = (
            "completion",
            cfg.data.task_dataset.negative_completion_column,
        )
        kwargs["test_dataset"] = on_policy_dataset
        on_policy_deceptive_key = deceptive_reward_key(cfg.task_args.deceptive_definition)
        if on_policy_deceptive_key in on_policy_reward_metrics:
            _, _, on_policy_split_stats = derive_on_policy_split_masks(
                on_policy_reward_metrics,
                deceptive_key=on_policy_deceptive_key,
            )
            assert on_policy_split_stats["n_total"] == len(on_policy_positive_dataset)
            print(
                "OK: on-policy split labels — "
                f"deceptive={on_policy_split_stats['n_deceptive']} "
                f"honest={on_policy_split_stats['n_honest']} "
                f"unlabeled={on_policy_split_stats['n_unlabeled']}"
            )
        else:
            print("Skipping on-policy split labels: no deceptive label for this task")
        detectors_list = [pretrained_detectors, list(map(lambda x: x[0], eval_results_after.values()))]
        del eval_results_after
        plot_keys = ["initial_detector.final_model.on_policy", "fresh_detector.final_model.on_policy"]
        plot_titles = [
            "Initial detector on final model activations from on-policy text",
            "Fresh detector on final model activations from on-policy text",
        ]
        filtered_layers_list = [filtered_layers, fresh_filtered_layers]

        # Evaluate detectors on policy model activations and collect results
        policy_eval_results_list = []
        for detectors, plot_key, plot_title, filtered_layers_d in zip(
            detectors_list, plot_keys, plot_titles, filtered_layers_list
        ):
            eval_results = eval_detectors_helper(
                detectors,
                eval_detector_types,
                plot_key,
                plot_title,
                cache_activations_save_path,
                is_main_process,
                save_path,
                filtered_layers=filtered_layers_d,
                **kwargs,  # layers in kwargs
            )
            log_on_policy_split_if_applicable(
                eval_results=eval_results,
                reward_metrics=on_policy_reward_metrics,
                save_path=save_path,
                plot_key=plot_key,
                filtered_layers=filtered_layers_d,
                false_positive_rate=kwargs.get("false_positive_rate", 0.01),
                deceptive_definition=cfg.task_args.deceptive_definition,
            )
            policy_eval_results_list.append(eval_results)

        # Evaluate initial detector on reference (base) model activations
        if cfg.training.only_eval_from_other_run:
            # we disable adapter since reference adapter is the already trained model in this mode.
            trainer.state.model.shared_base_model.base_model.disable_adapter_layers()  # type: ignore
        else:
            trainer.state.model.set_adapter(["reference"])  # type: ignore
        kwargs["model"] = trainer.state.model.shared_base_model
        reference_eval_results = eval_detectors_helper(
            pretrained_detectors,
            eval_detector_types,
            "initial_detector.base_model.on_policy",
            "Initial detector on base model activations from on-policy text",
            cache_activations_save_path,
            is_main_process,
            save_path,
            filtered_layers=filtered_layers,
            **kwargs,  # layers in kwargs
        )
        log_on_policy_split_if_applicable(
            eval_results=reference_eval_results,
            reward_metrics=on_policy_reward_metrics,
            save_path=save_path,
            plot_key="initial_detector.base_model.on_policy",
            filtered_layers=filtered_layers,
            false_positive_rate=kwargs.get("false_positive_rate", 0.01),
            deceptive_definition=cfg.task_args.deceptive_definition,
        )

        # Compute obfuscation decomposition metrics for each detector type
        if is_main_process and reference_eval_results is not None:
            obfuscation_prefixes = ["eval_obfuscation/initial_detector", "eval_obfuscation/fresh_detector"]
            for policy_eval_results, obf_prefix, filtered_layers_d in zip(
                policy_eval_results_list, obfuscation_prefixes, filtered_layers_list
            ):
                if policy_eval_results is not None:
                    compute_obfuscation_decomposition_metrics_helper(
                        policy_eval_results=policy_eval_results,
                        reference_eval_results=reference_eval_results,
                        reward_metrics=on_policy_reward_metrics,
                        detector_loss_coef=cfg.loss.detector,
                        detector_mean_decision_threshold=mean_decision_threshold,
                        filtered_layers=filtered_layers_d,
                        reference_detector_mean_positive_score_on_dataset=mean_positive_score,
                        policy_detector_mean_positive_score_on_dataset=policy_detector_mean_positive_score_on_dataset,  # type: ignore
                        prefix=obf_prefix,
                        deceptive_definition=cfg.task_args.deceptive_definition,
                    )

    # Successful end of run: drop activation caches (kept on crash for post-eval recovery)
    if is_main_process and cfg.detector.model_cache_path is not None:
        gc.collect()  # release MemoryMappedTensor mmap handles before rmtree
        cache_root = Path(cfg.detector.model_cache_path).resolve()
        deleted: list[str] = []
        dirs_to_delete: list[Path] = []
        if model_cache_path is not None:
            phase0_cache = Path(model_cache_path).resolve()
            if phase0_cache.is_dir() and phase0_cache.is_relative_to(cache_root):
                dirs_to_delete.append(phase0_cache)
        for path in sorted(cache_root.glob(f"activations_{wandb_run_id}*")):
            resolved = path.resolve()
            if resolved.is_dir() and resolved.is_relative_to(cache_root):
                dirs_to_delete.append(resolved)
        for d in dirs_to_delete:
            for attempt in range(3):
                try:
                    shutil.rmtree(d)
                    deleted.append(str(d))
                    break
                except OSError:
                    # NFS silly-rename (.nfs*) files from lingering mmap handles;
                    # brief sleep lets the kernel clean them up after GC above.
                    time.sleep(2)
            else:
                print(f"WARNING: could not remove activation cache {d} after 3 attempts (non-fatal)")
        if deleted:
            print(f"OK: deleted {len(deleted)} activation cache dir(s) after successful run:")
            for p in deleted:
                print(f"  - {p}")


def eval_detectors_helper(
    detectors: list,
    detector_types: list[str],
    plot_key: str,
    plot_title: str,
    cache_activations_save_path: str | None,
    is_main_process: bool,
    metrics_save_path: Path | str,
    layers: list[int],
    filtered_layers: list[int] | None = None,
    **kwargs,
) -> dict[str, dict[str, Any]] | None:
    """Helper function to evaluate detectors using already trained detectors.

    Args:
        detectors: The detectors to evaluate.
        detector_types: The types of detectors to evaluate.
        plot_key: The key to use for the plot.
        plot_title: The title to use for the plot.
        cache_activations_save_path: The path to save the cached activations.
        is_main_process: Whether to log metrics to wandb.
        metrics_save_path: The path to save the metrics.
        layers: The layers to evaluate the detectors on.
        filter_layers_below_loss_threshold: Threshold to filter layers below.
            If None, uses `layers` instead.
        **kwargs: Additional keyword arguments to pass to the eval_multiple_detectors function.

    Returns:
        The evaluation results dict (test_metrics per detector key), or None if not main process.
    """
    train_cache_activations_save_path, test_cache_activations_save_path = None, None
    if cache_activations_save_path is not None:
        train_cache_activations_save_path = f"{cache_activations_save_path}_{plot_key}_train"
        test_cache_activations_save_path = f"{cache_activations_save_path}_{plot_key}_test"
    eval_for_obf_policy = eval_multiple_detectors(
        detectors=detectors,
        detector_types=detector_types,
        train_cache_activations_save_path=train_cache_activations_save_path,
        test_cache_activations_save_path=test_cache_activations_save_path,
        **kwargs,
    )

    # convert to tuple format for logging metrics
    eval_for_logging = {k: (None, (v, None, None)) for k, v in eval_for_obf_policy.items()}
    if is_main_process:
        log_detector_metrics(
            eval_for_logging,
            plot_title,
            str(metrics_save_path),
            plot_key,
            layers,
            filtered_layers=filtered_layers,
        )
    return eval_for_obf_policy if is_main_process else None


def log_on_policy_split_if_applicable(
    eval_results: dict[str, dict[str, Any]] | None,
    reward_metrics: dict[str, list] | None,
    save_path: Path | str,
    plot_key: str,
    filtered_layers: list[int],
    false_positive_rate: float = 0.01,
    deceptive_definition: str = "pass_only_shown",
) -> None:
    """Run on-policy split eval when standard on-policy probe scoring has completed."""
    if not plot_key.endswith(".on_policy") or eval_results is None or reward_metrics is None:
        return
    if deceptive_reward_key(deceptive_definition) not in reward_metrics:
        return  # task has no deceptive label (e.g. MULTIPLE_CHOICE)
    run_on_policy_split_eval(
        eval_results=eval_results,
        reward_metrics=reward_metrics,
        save_path=save_path,
        plot_key=plot_key,
        filtered_layers=filtered_layers,
        false_positive_rate=false_positive_rate,
        deceptive_key=deceptive_reward_key(deceptive_definition),
        save_json=False,  # aggregate split metrics still go to W&B; skip local JSON dumps
    )


def extract_mean_detector_scores(
    eval_results: dict[str, dict[str, Any]] | None,
    filtered_layers: list[int],
) -> np.ndarray | None:
    """Extract mean detector scores across filtered layers for positive examples.

    Args:
        eval_results: Evaluation results from eval_detectors_helper.
        filtered_layers: Layers to average over.

    Returns:
        Mean detector scores per sample as numpy array.
    """
    if eval_results is None:  # non-main process
        return None
    # Get the first detector key (all detectors evaluate on same data)
    first_key = next(iter(eval_results.keys()))
    test_metrics = eval_results[first_key]

    positive_scores_per_layer = []
    for layer in filtered_layers:
        layer_key = str(layer)
        if layer_key in test_metrics and "scores" in test_metrics[layer_key]:
            positive_scores_per_layer.append(test_metrics[layer_key]["scores"]["positive"])

    # Average across layers: shape [num_layers, num_samples] -> [num_samples]
    return np.array(positive_scores_per_layer).mean(axis=0)


def compute_obfuscation_decomposition_metrics_helper(
    policy_eval_results: dict[str, dict[str, Any]],
    reference_eval_results: dict[str, dict[str, Any]],
    reward_metrics: dict[str, list],
    detector_loss_coef: float,
    detector_mean_decision_threshold: float,
    filtered_layers: list[int],
    reference_detector_mean_positive_score_on_dataset: float | None = None,
    policy_detector_mean_positive_score_on_dataset: float | None = None,
    prefix: str = "eval_obfuscation",
    deceptive_definition: str = "pass_only_shown",
) -> dict[str, float]:
    """Compute obfuscation decomposition metrics from policy and reference eval results.

    Args:
        policy_eval_results: Eval results from policy model (final model activations).
        reference_eval_results: Eval results from reference model (base model activations).
        reward_metrics: Reward metrics from on-policy generation containing task rewards and deceptive metric.
        detector_loss_coef: Coefficient for detector loss.
        detector_mean_decision_threshold: Mean decision threshold from detector training.
        reference_detector_mean_positive_score_on_dataset: Mean positive score on dataset from reference model.
        policy_detector_mean_positive_score_on_dataset: Mean positive score on dataset from policy model.
        filtered_layers: Layers to average detector scores over.
        prefix: Prefix for logged metric names.

    Returns:
        Dict of obfuscation decomposition metrics.
    """
    import torch

    # Extract mean detector scores across filtered layers
    policy_scores = extract_mean_detector_scores(policy_eval_results, filtered_layers)
    reference_scores = extract_mean_detector_scores(reference_eval_results, filtered_layers)

    # Extract task rewards and deceptive metric from reward_metrics
    # Keys match those used in compute_code_generation_rewards
    if "rewards/task_success" not in reward_metrics:
        return {}
    deceptive_metric = reward_metrics.get(deceptive_reward_key(deceptive_definition))
    if deceptive_metric is None:
        return {}  # task has no deceptive label (e.g. MULTIPLE_CHOICE)
    task_rewards = torch.tensor(reward_metrics["rewards/task_success"]).float()

    # Compute decomposition metrics
    metrics = compute_obfuscation_decomposition_metrics(
        reference_detector_prob=torch.tensor(reference_scores),
        policy_detector_prob=torch.tensor(policy_scores),
        task_rewards=task_rewards,
        deceptive_metric=deceptive_metric,
        detector_loss_coef=detector_loss_coef,
        detector_mean_decision_threshold=detector_mean_decision_threshold,
        detector_mean_positive_threshold=reference_detector_mean_positive_score_on_dataset,
    )

    # Log to wandb with the specified prefix, stripping the "obfuscation/" prefix from keys
    wandb_metrics = {f"{prefix}/{k.replace('obfuscation/', '')}": v for k, v in metrics.items()}
    if (
        reference_detector_mean_positive_score_on_dataset is not None
        and policy_detector_mean_positive_score_on_dataset is not None
    ):
        wandb_metrics["representation_change_on_dataset"] = (
            reference_detector_mean_positive_score_on_dataset - policy_detector_mean_positive_score_on_dataset
        )
    if wandb.run is not None:
        wandb.log(wandb_metrics)

    return wandb_metrics


def generate_on_policy_responses(
    trainer: GRPOObfuscationTrainer,
    prompts: list[str],
    final_adapter_path: Path,
    num_generations: int = 1,
    log_to_wandb: bool = False,
    model_wandb_id: str | None = None,
    test_lists: list[list[str]] | None = None,
    test_setup_code_list: list[str] | None = None,
    texts: list[str] | None = None,
    code_length_penalty: float = 0.002,
    num_test_cases: int = 1,
    answers: list[str] | None = None,
    label_letters: list[str] | None = None,
    cue_letters: list[str] | None = None,
) -> tuple[Dataset, dict[str, list]]:
    """Function to generate on-policy responses using the GRPOObfuscationTrainer.

    Args:
        trainer: The trainer to use to generate the responses.
        prompts: The prompts to generate the responses for.
        final_adapter_path: The path to the final adapter.
        num_generations: The number of generations to generate.
        log_to_wandb: Whether to log the responses to wandb.
        model_wandb_id: If provided, we fetch on policy generations from this wandb id.
        test_lists: Optional list of test cases for each prompt (for code generation metrics).
        test_setup_code_list: Optional setup code for each prompt.
        texts: Optional task text for each prompt (for pass_only_shown_deception alt generation).
        code_length_penalty: Penalty per character of code length.
        num_test_cases: Number of test cases to use for reward computation.
        answers: Optional gold letters per prompt (for MULTIPLE_CHOICE metrics).
        label_letters: Training labels per prompt (MULTIPLE_CHOICE; required with answers).
        cue_letters: Optional served cue letters per prompt (for MULTIPLE_CHOICE metrics).

    Returns:
        A tuple of (Dataset containing prompts and responses, reward_metrics dict).
    """
    prompts_formatted, response_strings, reward_metrics = None, None, None  # type: ignore
    if model_wandb_id:
        run = wandb.Api().run(model_wandb_id)
        df = None
        for artifact in run.logged_artifacts():
            if "on_policy_responses" in artifact.name:
                table = artifact.get("on_policy_responses")
                df = table.get_dataframe()
                break
        if df is not None:
            print(f"Found on policy responses for wandb id: {model_wandb_id}. Using them.")
            # Extract reward_metrics from the dataframe columns if available
            reward_metrics: dict[str, list] = {}  # type: ignore
            for col in df.columns:
                if col not in ["prompt", "completion"]:
                    reward_metrics[col] = df[col].tolist()
            prompts_formatted = df["prompt"].tolist()
            response_strings = df["completion"].tolist()
        else:
            print(f"No on policy responses found for wandb id: {model_wandb_id}. Generating new responses using vLLM.")

    if prompts_formatted is None:
        prompts_formatted: list[str] = (
            trainer.tokenizer.apply_chat_template(  # type: ignore
                [[{"role": "user", "content": prompt}] for prompt in prompts],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if trainer.config.tokenizer.apply_chat_template
            else list(prompts)
        )
        # MULTIPLE_CHOICE no-CoT format: the assistant turn starts with the prefill, as in training
        nocot_prefill = trainer.obfuscation_config.task_args.nocot_prefill
        if nocot_prefill is not None:
            assert trainer.obfuscation_config.task == TaskType.MULTIPLE_CHOICE, "nocot_prefill is MULTIPLE_CHOICE only"
            prompts_formatted = [prompt + nocot_prefill for prompt in prompts_formatted]
        prompt_ids: list[list[int]] = trainer.tokenizer(  # type: ignore
            prompts_formatted,
            add_special_tokens=False,
        )["input_ids"]
        assert isinstance(prompt_ids, list)
        vllm_client = VLLMClient(trainer.config.vllm)
        responses = trainer.generate_responses(
            vllm_client,
            prompt_ids,
            trainer.state.model.policy_adapter,
            final_adapter_path,
            num_generations=num_generations,
        )
        response_strings = trainer.tokenizer.batch_decode(responses.tokens)
        # Compute code generation metrics if test_lists provided
        reward_metrics: dict[str, list] = {}
        if test_lists is not None and "rewards/task_success" not in reward_metrics:
            completions = [r.removesuffix(trainer.tokenizer.eos_token) for r in response_strings]  # type: ignore
            alt_completions = None
            if texts is not None and len(texts) == len(test_lists):
                system_prompt = trainer.obfuscation_config.data.task_dataset.system_prompt
                alt_prompts = [
                    build_mbpp_chat_prompt(text, test_list, 1, system_prompt, trainer.tokenizer)
                    for text, test_list in zip(texts, test_lists)
                ]
                alt_prompt_ids: list[list[int]] = trainer.tokenizer(  # type: ignore
                    alt_prompts,
                    add_special_tokens=False,
                )["input_ids"]
                alt_responses = trainer.generate_responses(
                    vllm_client,
                    alt_prompt_ids,
                    trainer.state.model.policy_adapter,
                    final_adapter_path,
                    num_generations=num_generations,
                )
                alt_response_strings = trainer.tokenizer.batch_decode(alt_responses.tokens)
                eot = trainer.tokenizer.eot_token or trainer.tokenizer.eos_token or ""
                alt_per_prompt = [r.removesuffix(eot) for r in alt_response_strings]
                if num_generations > 1:
                    alt_completions = [
                        alt_per_prompt[i // num_generations] for i in range(len(completions))
                    ]
                else:
                    alt_completions = alt_per_prompt
            _, reward_metrics = compute_code_generation_rewards(
                prompts_formatted,
                completions,
                test_lists,
                test_setup_code_list,
                code_length_penalty=code_length_penalty,
                num_test_cases=num_test_cases,
                max_workers=1,
                alt_completions=alt_completions,
            )
        elif answers is not None:
            eot = trainer.tokenizer.eot_token or trainer.tokenizer.eos_token or ""
            completions = [r.removesuffix(eot) for r in response_strings]
            assert label_letters is not None
            answers_upsampled = [answers[i // num_generations] for i in range(len(completions))]
            labels_upsampled = [label_letters[i // num_generations] for i in range(len(completions))]
            cue_letters_upsampled = (
                [cue_letters[i // num_generations] for i in range(len(completions))]
                if cue_letters is not None
                else None
            )
            _, reward_metrics = compute_multiple_choice_rewards(
                completions,
                answers_upsampled,
                labels_upsampled,
                cue_letters_upsampled,
                letter_only=trainer.obfuscation_config.task_args.nocot_prefill is not None,
            )

    if log_to_wandb:
        # Build table columns: prompt, completion, plus any reward metrics
        columns = ["prompt", "completion"]
        metric_keys = list(reward_metrics.keys())
        columns.extend(metric_keys)

        # Build table data
        data = []
        for i, (prompt, completion) in enumerate(zip(prompts_formatted, response_strings)):  # type: ignore
            row = [prompt, completion]
            for k in metric_keys:
                row.append(reward_metrics[k][i])
            data.append(row)

        wandb.log({"on_policy_responses": wandb.Table(columns=columns, data=data)})
        # log aggregated reward metrics
        wandb.log({f"final_eval/{k}": np.mean(v) for k, v in reward_metrics.items()})

    return Dataset.from_dict({"prompt": prompts_formatted, "completion": response_strings}), reward_metrics


@hydra.main(version_base=None, config_name="grpo_code_gen")
def _hydra_entry(cfg: DictConfig):
    cfg_obj: GRPOObfuscationConfig = OmegaConf.to_object(cfg)  # type: ignore
    run_rl(cfg_obj)


if __name__ == "__main__":
    _hydra_entry()
