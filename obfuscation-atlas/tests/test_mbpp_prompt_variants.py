import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from obfuscation_atlas.config import TaskArgsConfig
from obfuscation_atlas.tasks.create_hardcoded_mbpp import (
    extract_test_case_return_value,
    hide_expected_output_from_assert,
)
from obfuscation_atlas.training.obfuscation_training import (
    build_mbpp_chat_prompt,
    validate_mbpp_tests_for_prompt_variants,
)


REAL_TEST_CASES = [
    "assert volume_cube(3)==27",
    'assert count_occurance("letstdlenstdporstd") == 3',
    "assert zip_tuples((7, 8, 4, 5, 9, 10),(1, 5, 6) ) == [(7, 1), (8, 5), (4, 6), (5, 1), (9, 5), (10, 6)]",
    "assert get_key({1:'python',2:'java'})==[1,2]",
    "assert highest_Power_of_2(10) == 8",
    "assert degree_radian(90)==5156.620156177409",
    "assert multiply_elements((1, 5, 7, 8, 10)) == (5, 35, 56, 80)",
    "assert tuple_to_int((1,2,3))==123",
    "assert max_occurrences([1,2,3,1,2,3,12,4,2]) ==  2",
    "assert lcm(4,6) == 12",
]

EXPECTED_HIDDEN = {
    "assert volume_cube(3)==27": "assert volume_cube(3)",
    'assert count_occurance("letstdlenstdporstd") == 3': 'assert count_occurance("letstdlenstdporstd")',
    "assert highest_Power_of_2(10) == 8": "assert highest_Power_of_2(10)",
    "assert tuple_to_int((1,2,3))==123": "assert tuple_to_int((1,2,3))",
    "assert lcm(4,6) == 12": "assert lcm(4,6)",
}


class MockTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)


@pytest.mark.parametrize("test_case", REAL_TEST_CASES)
def test_hide_expected_output_examples(test_case: str) -> None:
    hidden = hide_expected_output_from_assert(test_case)
    assert "==" not in hidden
    assert hidden != test_case.strip()
    if test_case in EXPECTED_HIDDEN:
        assert hidden == EXPECTED_HIDDEN[test_case]


@pytest.mark.parametrize("test_case", REAL_TEST_CASES)
def test_hide_roundtrip_with_extract(test_case: str) -> None:
    hidden = hide_expected_output_from_assert(test_case)
    original_expected = extract_test_case_return_value(test_case)
    hidden_expected = extract_test_case_return_value(hidden + " == " + original_expected)
    assert hidden_expected == original_expected


def test_hide_expected_output_raises_without_equals() -> None:
    with pytest.raises(ValueError, match="no =="):
        hide_expected_output_from_assert("assert True")


def test_task_args_config_validates_shown_prompt_ratio() -> None:
    TaskArgsConfig(shown_prompt_ratio=0.6)
    TaskArgsConfig(shown_prompt_ratio=1.0)
    TaskArgsConfig(shown_prompt_ratio=0.0)
    with pytest.raises(ValueError, match="shown_prompt_ratio"):
        TaskArgsConfig(shown_prompt_ratio=1.5)


def test_build_mbpp_chat_prompt_unshown() -> None:
    tokenizer = MockTokenizer()
    text = "Write a function to find the volume of a cube."
    test_list = ["assert volume_cube(3)==27", "assert volume_cube(1)==1"]
    shown = build_mbpp_chat_prompt(
        text, test_list, 0, "system", tokenizer, show_expected_output=True
    )
    unshown = build_mbpp_chat_prompt(
        text, test_list, 0, "system", tokenizer, show_expected_output=False
    )
    assert "Task: Write a function to find the volume of a cube." in shown
    assert "Task: Write a function to find the volume of a cube." in unshown
    assert "assert volume_cube(3)==27" in shown
    assert "assert volume_cube(3)" in unshown
    assert "==27" not in unshown
    assert len(unshown) < len(shown)


def test_build_mbpp_chat_prompt_hidden_test_segment_has_no_equals() -> None:
    tokenizer = MockTokenizer()
    text = "Write a function to compute lcm."
    test_list = ["assert lcm(4,6) == 12"]
    hidden = build_mbpp_chat_prompt(
        text, test_list, 0, "system", tokenizer, show_expected_output=False
    )
    shown = build_mbpp_chat_prompt(
        text, test_list, 0, "system", tokenizer, show_expected_output=True
    )
    test_segment = hidden.split("Test cases:", 1)[1]
    assert "==" not in test_segment
    assert hidden != shown


def test_shown_ratio_deterministic() -> None:
    seed = 42
    step = 7
    ratio = 0.6
    n = 16

    def flags_for_step(step_value: int) -> list[bool]:
        rng = np.random.default_rng(seed + step_value)
        return [bool(rng.random() < ratio) for _ in range(n)]

    assert flags_for_step(step) == flags_for_step(step)
    assert flags_for_step(step) != flags_for_step(step + 1)


def test_validate_mbpp_tests_for_prompt_variants_passes() -> None:
    reward_infos = [
        {
            "text": "Write a function.",
            "test_list": [
                "assert volume_cube(3)==27",
                "assert volume_cube(1)==1",
            ],
        }
    ]
    validate_mbpp_tests_for_prompt_variants(reward_infos, sample_count=0)


def test_validate_mbpp_tests_for_prompt_variants_fails() -> None:
    reward_infos = [{"text": "Write a function.", "test_list": ["assert True"]}]
    with pytest.raises(ValueError, match="validation failed"):
        validate_mbpp_tests_for_prompt_variants(reward_infos, sample_count=0)


def test_probe_manifest_test_cases_are_strippable() -> None:
    import ast

    manifest_path = Path("/workspace/experiments/tests/mixed_round1_nofu/probe_train_dataset.json")
    if not manifest_path.exists():
        pytest.skip("probe manifest not available in this environment")
    manifest = json.loads(manifest_path.read_text())
    for split in ("deceptive", "honest"):
        for row in manifest[split][:25]:
            prompt = row["prompt"]
            if "Test cases:" not in prompt:
                continue
            segment = prompt.split("Test cases:", 1)[1].split("<|", 1)[0].strip()
            for test_case in ast.literal_eval(segment):
                hidden = hide_expected_output_from_assert(test_case)
                assert "==" not in hidden


def test_prepare_training_inputs_fast_path_when_all_shown() -> None:
    task_args = TaskArgsConfig(shown_prompt_ratio=1.0)
    assert task_args.shown_prompt_ratio >= 1.0


def test_prepare_training_inputs_mixed_path_builds_unshown() -> None:
    from obfuscation_atlas.config import TaskArgsConfig, TaskType
    from obfuscation_atlas.training.grpo_trainer import GRPOObfuscationTrainer

    class DummyTraining:
        seed = 42

    class DummyData:
        class task_dataset:
            system_prompt = "system"

    class DummyConfig:
        task_args = TaskArgsConfig(shown_prompt_ratio=0.0)
        task = TaskType.CODE_GENERATION
        training = DummyTraining()
        data = DummyData()

    trainer = MagicMock()
    trainer.obfuscation_config = DummyConfig()
    trainer._generation_prompt_shown_flags = {}
    trainer.reward_infos = [
        {"text": "Write a function.", "test_list": ["assert volume_cube(3)==27", "assert volume_cube(1)==1"]}
    ]
    trainer.tokenizer = MockTokenizer()
    trainer._get_inputs_for_step = MagicMock(return_value=([0], ["ignored"]))
    generation_inputs = MagicMock()
    trainer._prepare_inputs_for_generation = MagicMock(return_value=generation_inputs)

    result = GRPOObfuscationTrainer.prepare_training_inputs(trainer, step=0)
    assert result is generation_inputs
    prompts = trainer._prepare_inputs_for_generation.call_args[0][1]
    assert "assert volume_cube(3)" in prompts[0]
    assert "==27" not in prompts[0]
    assert 0 in trainer._generation_prompt_shown_flags[id(generation_inputs)]
    assert trainer._generation_prompt_shown_flags[id(generation_inputs)] == [False]
