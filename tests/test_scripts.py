import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


def load_script(script_name: str):
    path = Path(__file__).parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


@pytest.mark.parametrize(
    ("script_name", "work_function"),
    [
        ("generate.py", "generate"),
        ("prepare_data.py", "prepare_data"),
        ("train.py", "train"),
        ("train_tokenizer.py", "train_tokenizer"),
    ],
)
def test_script_help_exits_without_running_work(
    script_name: str,
    work_function: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, module = load_script(script_name)

    monkeypatch.setattr(module, work_function, Mock(side_effect=AssertionError("work started")))
    monkeypatch.setattr(sys, "argv", [str(path), "--help"])

    with pytest.raises(SystemExit) as exit_info:
        module.main()

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("usage:")


def test_tokenizer_output_directory_is_forwarded(tmp_path, monkeypatch, capsys):
    path, module = load_script("train_tokenizer.py")
    train_tokenizer = Mock(return_value=tmp_path / "candidate")
    monkeypatch.setattr(module, "train_tokenizer", train_tokenizer)
    monkeypatch.setattr(sys, "argv", [str(path), "--output-dir", str(tmp_path / "candidate")])

    module.main()

    train_tokenizer.assert_called_once_with(tmp_path / "candidate")
    assert capsys.readouterr().out.strip() == str(tmp_path / "candidate")


def test_train_help_shows_training_defaults(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, module = load_script("train.py")
    monkeypatch.setattr(sys, "argv", [str(path), "--help"])

    with pytest.raises(SystemExit) as exit_info:
        module.main()

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--architecture {vanilla,rope}" in help_text
    assert "--steps STEPS" in help_text
    assert "(default: 16384)" in help_text
    assert "--validation-interval VALIDATION_INTERVAL" in help_text
    assert "--checkpoint-interval" not in help_text
    assert "--milestone-interval CHECKPOINTS" in help_text
    assert "(default: 4)" in help_text
    assert "--wandb-project PROJECT" in help_text
    assert "--no-wandb" in help_text
    assert "--resume-checkpoint CHECKPOINT" in help_text
    assert "--continue-run" in help_text


def test_train_reports_to_wandb_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    path, module = load_script("train.py")
    train = Mock(return_value=Path("latest.pt"))
    monkeypatch.setattr(module, "train", train)
    monkeypatch.setattr(sys, "argv", [str(path), "--architecture", "vanilla"])

    module.main()

    run_conf = train.call_args.kwargs["run_conf"]
    assert run_conf.wandb_project == "ventris"
    assert run_conf.validation_interval_steps == 250
    assert run_conf.milestone_interval_checkpoints == 4
    assert train.call_args.kwargs["continue_run"] is False


def test_train_continue_run_requires_resume_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    path, module = load_script("train.py")
    train = Mock(return_value=Path("latest"))
    monkeypatch.setattr(module, "train", train)
    monkeypatch.setattr(sys, "argv", [str(path), "--architecture", "vanilla", "--continue-run"])

    with pytest.raises(SystemExit) as exit_info:
        module.main()

    assert exit_info.value.code == 2
    train.assert_not_called()


def test_train_continue_run_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    path, module = load_script("train.py")
    train = Mock(return_value=Path("latest"))
    monkeypatch.setattr(module, "train", train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--architecture",
            "vanilla",
            "--resume-checkpoint",
            "run/step-001000",
            "--continue-run",
        ],
    )

    module.main()

    assert train.call_args.kwargs["resume"] == Path("run/step-001000")
    assert train.call_args.kwargs["continue_run"] is True


def test_train_can_explicitly_disable_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    path, module = load_script("train.py")
    train = Mock(return_value=Path("latest.pt"))
    monkeypatch.setattr(module, "train", train)
    monkeypatch.setattr(sys, "argv", [str(path), "--architecture", "vanilla", "--no-wandb"])

    module.main()

    assert train.call_args.kwargs["run_conf"].wandb_project is None


def test_train_rejects_unlisted_argument_abbreviations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, module = load_script("train.py")
    train = Mock(return_value=Path("latest.pt"))
    monkeypatch.setattr(module, "train", train)
    monkeypatch.setattr(sys, "argv", [str(path), "--architecture", "vanilla", "--wandb"])

    with pytest.raises(SystemExit) as exit_info:
        module.main()

    assert exit_info.value.code == 2
    train.assert_not_called()


def test_fresh_training_requires_explicit_architecture(monkeypatch, capsys):
    path, module = load_script("train.py")
    train = Mock()
    monkeypatch.setattr(module, "train", train)
    monkeypatch.setattr(sys, "argv", [str(path)])

    with pytest.raises(SystemExit) as error:
        module.main()

    assert error.value.code == 2
    assert "--architecture or --resume-checkpoint is required" in capsys.readouterr().err
    train.assert_not_called()


def test_resumed_training_can_read_architecture_from_checkpoint(monkeypatch):
    path, module = load_script("train.py")
    train = Mock(return_value=Path("latest"))
    monkeypatch.setattr(module, "train", train)
    monkeypatch.setattr(sys, "argv", [str(path), "--resume-checkpoint", "run/latest"])

    module.main()

    assert train.call_args.kwargs["resume"] == Path("run/latest")
    assert train.call_args.kwargs["architecture"] is None


@pytest.mark.parametrize("architecture", ["vanilla", "rope"])
def test_train_forwards_architecture_selection(monkeypatch, architecture):
    path, module = load_script("train.py")
    train = Mock(return_value=Path("latest"))
    monkeypatch.setattr(module, "train", train)
    monkeypatch.setattr(sys, "argv", [str(path), "--architecture", architecture])

    module.main()

    assert train.call_args.kwargs["architecture"] == architecture


def test_train_rejects_unsupported_architecture_choice(monkeypatch):
    path, module = load_script("train.py")
    train = Mock()
    monkeypatch.setattr(module, "train", train)
    monkeypatch.setattr(sys, "argv", [str(path), "--architecture", "mla"])

    with pytest.raises(SystemExit) as error:
        module.main()

    assert error.value.code == 2
    train.assert_not_called()
