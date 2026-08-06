import numpy as np
import pytest

from obfuscation_atlas.config import TaskArgsConfig
from obfuscation_atlas.tasks.code_generation import apply_mixed_reward_regime


def test_task_args_config_validates_mixed_reward_pass_all_prob() -> None:
    TaskArgsConfig(mixed_reward_pass_all_prob=0.4)
    TaskArgsConfig(mixed_reward_pass_all_prob=0.0)
    TaskArgsConfig(mixed_reward_pass_all_prob=1.0)
    with pytest.raises(ValueError, match="mixed_reward_pass_all_prob"):
        TaskArgsConfig(mixed_reward_pass_all_prob=1.5)
    with pytest.raises(ValueError, match="mixed_reward_pass_all_prob"):
        TaskArgsConfig(mixed_reward_pass_all_prob=-0.1)


def test_mixed_reward_noop_when_prob_zero() -> None:
    sequence_reward = [1.0, 0.0, 1.0, 1.0]
    prompt_indices = [10, 10, 20, 20]
    passes_all_tests = [False, False, True, False]
    length_penalties = [-0.01, -0.01, -0.02, -0.02]
    rng = np.random.default_rng(0)

    new_rewards, fraction = apply_mixed_reward_regime(
        sequence_reward, prompt_indices, passes_all_tests, length_penalties, 0.0, rng
    )

    assert new_rewards == sequence_reward
    assert fraction == 0.0


def test_mixed_reward_all_pass_all_regime_when_prob_one() -> None:
    # 2 prompts x 4 completions each (G=4), mimicking upsampled prompt_indices.
    prompt_indices = [1, 1, 1, 1, 2, 2, 2, 2]
    # Prompt 1: passes first test (shown) on all completions, but only completion 0 passes ALL tests.
    # Prompt 2: fails the first test everywhere, so passes_first_n_tests would be False for all;
    # here we only need passes_all_tests to determine the pass-all score.
    passes_all_tests = [True, False, False, False, False, False, False, False]
    length_penalties = [-0.01] * 8
    sequence_reward = [1.0 - 0.01, 1.0 - 0.01, 1.0 - 0.01, 1.0 - 0.01, 0.0, 0.0, 0.0, 0.0]
    rng = np.random.default_rng(0)

    new_rewards, fraction = apply_mixed_reward_regime(
        sequence_reward, prompt_indices, passes_all_tests, length_penalties, 1.0, rng
    )

    assert fraction == 1.0
    # Prompt 1, completion 0: passes_all_tests -> 1.0 + length_penalty (length_penalty already negative)
    assert new_rewards[0] == pytest.approx(1.0 - 0.01)
    # Prompt 1, completions 1-3: fails passes_all_tests -> 0.0 even though original reward was 1-penalty
    assert new_rewards[1:4] == [0.0, 0.0, 0.0]
    # Prompt 2: all fail passes_all_tests -> 0.0 (same as before, since original reward was already 0.0)
    assert new_rewards[4:] == [0.0, 0.0, 0.0, 0.0]


def test_mixed_reward_shares_regime_within_prompt_group() -> None:
    prompt_indices = [1, 1, 1, 1, 2, 2, 2, 2]
    passes_all_tests = [True, True, True, True, True, True, True, True]
    length_penalties = [0.0] * 8
    sequence_reward = [1.0] * 8

    rng = np.random.default_rng(123)
    new_rewards, fraction = apply_mixed_reward_regime(
        sequence_reward, prompt_indices, passes_all_tests, length_penalties, 0.5, rng
    )

    # Regardless of which prompts were drawn into the pass-all regime, since every completion
    # here passes all tests and length_penalty is 0, all rewards should be unchanged (1.0).
    assert new_rewards == [1.0] * 8
    assert fraction in (0.0, 0.5, 1.0)


def test_mixed_reward_regime_is_per_prompt_not_per_completion() -> None:
    # A prompt drawn into the pass-all regime must apply consistently to every one of its G
    # completions, even when their passes_all_tests outcomes differ.
    prompt_indices = [1, 1, 1, 1]
    passes_all_tests = [True, False, True, False]
    length_penalties = [-0.05, -0.05, -0.05, -0.05]
    sequence_reward = [1.0 - 0.05, 1.0 - 0.05, 1.0 - 0.05, 1.0 - 0.05]

    rng = np.random.default_rng(0)
    new_rewards, fraction = apply_mixed_reward_regime(
        sequence_reward, prompt_indices, passes_all_tests, length_penalties, 1.0, rng
    )

    assert fraction == 1.0
    assert new_rewards == [1.0 - 0.05, 0.0, 1.0 - 0.05, 0.0]


def test_mixed_reward_deterministic_given_same_rng_state() -> None:
    prompt_indices = list(range(20)) * 8
    passes_all_tests = [True] * len(prompt_indices)
    length_penalties = [0.0] * len(prompt_indices)
    sequence_reward = [1.0] * len(prompt_indices)

    rewards_a, fraction_a = apply_mixed_reward_regime(
        sequence_reward, prompt_indices, passes_all_tests, length_penalties, 0.4, np.random.default_rng(42)
    )
    rewards_b, fraction_b = apply_mixed_reward_regime(
        sequence_reward, prompt_indices, passes_all_tests, length_penalties, 0.4, np.random.default_rng(42)
    )

    assert rewards_a == rewards_b
    assert fraction_a == fraction_b


def test_mixed_reward_rejects_mismatched_lengths() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        apply_mixed_reward_regime([1.0, 0.0], [1, 1], [True], [0.0, 0.0], 0.5, rng)
