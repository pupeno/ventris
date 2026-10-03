import pytest
import torch

from ventris.models.rope import ModelConfig as RopeConfig
from ventris.models.rope import Ventris as RopeVentris
from ventris.models.vanilla import ModelConfig, Ventris


@pytest.mark.parametrize(
    "model_class,config_class", [(Ventris, ModelConfig), (RopeVentris, RopeConfig)]
)
def test_default_models_own_independent_configurations(model_class, config_class):
    with torch.device("meta"):
        first = model_class()
        second = model_class()

    first.config.max_position_embeddings = 4

    assert isinstance(first.config, config_class)
    assert first.config is not second.config
    assert second.config.max_position_embeddings == 1_024


def test_factory_creates_selected_vanilla():
    from ventris.models import create_model

    with torch.device("meta"):
        model = create_model("vanilla")

    assert isinstance(model, Ventris)
    assert model.config.model_type == "ventris-vanilla-v1"
    assert model.config.shape_dict() == {
        "vocab_size": 50_257,
        "max_position_embeddings": 1_024,
        "num_hidden_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "intermediate_size": 2_048,
    }


def test_loader_uses_saved_identity_and_dimensions(tmp_path):
    import json

    from tests.helpers import tiny_model
    from ventris.models import load_model

    model = tiny_model().eval()
    input_ids = torch.tensor([[1, 2, 3]])
    model.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    saved_config = json.loads(config_path.read_text())
    saved_config["architectures"] = ["UnrelatedModel"]
    config_path.write_text(json.dumps(saved_config))

    restored = load_model(tmp_path, expected_architecture="vanilla")

    assert isinstance(restored, Ventris)
    assert restored.config.shape_dict() == model.config.shape_dict()
    torch.testing.assert_close(restored(input_ids).logits, model(input_ids).logits)


@pytest.mark.parametrize("model_type", [None, "ventris-vanilla-v2", "ventris-rope-v2", "unknown"])
def test_loader_rejects_missing_or_unsupported_saved_identity(tmp_path, model_type):
    import json

    from tests.helpers import tiny_model
    from ventris.models import load_model

    tiny_model().save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    saved_config = json.loads(config_path.read_text())
    if model_type is None:
        del saved_config["model_type"]
    else:
        saved_config["model_type"] = model_type
    config_path.write_text(json.dumps(saved_config))

    with pytest.raises(ValueError, match="model_type"):
        load_model(tmp_path)


@pytest.mark.parametrize("architecture", ["unknown", "Vanilla", "ventris-vanilla-v1"])
def test_factory_and_loader_reject_unsupported_architecture_selections(tmp_path, architecture):
    from ventris.models import create_model, load_model

    with pytest.raises(ValueError, match="unsupported architecture"):
        create_model(architecture)
    with pytest.raises(ValueError, match="unsupported expected architecture"):
        load_model(tmp_path, expected_architecture=architecture)


@pytest.mark.parametrize("workflow", ["generation", "resume"])
def test_shared_workflows_reject_unsupported_saved_identity(tmp_path, workflow):
    import json

    from tests.helpers import tiny_model
    from ventris.checkpoint import load_checkpoint
    from ventris.common import TrainingConfig
    from ventris.generate import generate

    tiny_model().save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    saved_config = json.loads(config_path.read_text())
    saved_config["model_type"] = "ventris-vanilla-v2"
    config_path.write_text(json.dumps(saved_config))

    with pytest.raises(ValueError, match="model_type"):
        if workflow == "generation":
            generate(tmp_path, "prompt")
        else:
            load_checkpoint(tmp_path, torch.device("cpu"), TrainingConfig())


def test_rope_factory_and_custom_configuration_round_trip(tmp_path):
    from safetensors.torch import load_file

    from ventris.models import create_model, load_model
    from ventris.models.rope import ModelConfig as RopeConfig
    from ventris.models.rope import Ventris as RopeVentris

    with torch.device("meta"):
        default = create_model("rope")
    assert isinstance(default, RopeVentris)
    assert default.config.shape_dict() == ModelConfig().shape_dict()
    assert default.config.rope_theta == 10_000.0
    assert "rope_theta" not in default.config.shape_dict()
    model = RopeVentris(
        RopeConfig(
            vocab_size=32,
            hidden_size=16,
            num_attention_heads=2,
            num_hidden_layers=1,
            intermediate_size=32,
            max_position_embeddings=8,
            rope_theta=625.0,
        )
    ).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    expected = model(input_ids).logits
    model.save_pretrained(tmp_path)
    restored = load_model(tmp_path, expected_architecture="rope")
    assert isinstance(restored, RopeVentris)
    assert restored.config.rope_theta == 625.0
    assert restored.config.shape_dict() == model.config.shape_dict()
    torch.testing.assert_close(restored(input_ids).logits, expected)
    # Saved state consists of learned parameters; no derived positional tensors.
    assert sum(
        tensor.numel() for tensor in load_file(tmp_path / "model.safetensors").values()
    ) == sum(parameter.numel() for parameter in model.parameters())


@pytest.mark.parametrize("saved,requested", [("vanilla", "rope"), ("rope", "vanilla")])
def test_loader_rejects_conflict_before_reading_weights(tmp_path, saved, requested):
    import json

    from ventris.models import load_model

    (tmp_path / "config.json").write_text(json.dumps({"model_type": f"ventris-{saved}-v1"}))
    with pytest.raises(ValueError, match=f"{requested}.*{saved}"):
        load_model(tmp_path, expected_architecture=requested)
