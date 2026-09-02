import ast
import json
from pathlib import Path

import torch

from models.pidnet_multiclass import PIDNetSmall, load_pidnet_imagenet_weights


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pidnet_small_full_resolution_output_and_gradients():
    model = PIDNetSmall(in_channels=5, classes=5).train()
    inputs = torch.randn(2, 5, 64, 64)
    logits = model(inputs)
    assert logits.shape == (2, 5, 64, 64)
    logits.mean().backward()
    assert model.conv1[0].weight.grad is not None
    assert 7_000_000 < sum(parameter.numel() for parameter in model.parameters()) < 8_000_000


def test_pidnet_imagenet_loader_mean_initializes_extra_channels(tmp_path: Path):
    model = PIDNetSmall(in_channels=5, classes=5)
    source = {}
    for key, value in model.state_dict().items():
        if key.startswith("final_layer."):
            continue
        if key == "conv1.0.weight":
            source[key] = torch.randn(value.shape[0], 3, *value.shape[2:])
        else:
            source[key] = torch.randn_like(value) if value.is_floating_point() else value.clone()
    checkpoint = tmp_path / "pidnet_s_imagenet.pth.tar"
    torch.save({"state_dict": source}, checkpoint)

    result = load_pidnet_imagenet_weights(model, checkpoint)
    loaded = model.conv1[0].weight.detach()
    assert result["loaded_tensors"] >= 50
    assert result["adapted_input_tensors"] == 1
    assert torch.equal(loaded[:, :3], source["conv1.0.weight"])
    expected_extra = source["conv1.0.weight"].mean(dim=1)
    assert torch.allclose(loaded[:, 3], expected_extra)
    assert torch.allclose(loaded[:, 4], expected_extra)


def test_pidnet_seed_configs_are_paired_and_keep_test_locked():
    configs = []
    for seed in (42, 1337):
        path = PROJECT_ROOT / "configs" / f"v6_overlap40_pidnet_s_full_seed{seed}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["seed"] == seed
        assert payload["model"] == "PIDNet-S"
        assert payload["encoder"] == "native_pidnet_s"
        assert payload["batch_size"] == 4 and payload["accum_steps"] == 1
        assert payload["selection_metric"] == "val_mIoU_fg"
        assert payload["automatic_test_evaluation"] is False
        configs.append(payload)

    ignored = {"experiment", "hypothesis", "comparison_scope", "seed", "run_name"}
    assert {k: v for k, v in configs[0].items() if k not in ignored} == {
        k: v for k, v in configs[1].items() if k not in ignored
    }


def test_pidnet_notebooks_are_valid_and_never_request_test_evaluation():
    for seed in (42, 1337):
        path = (
            PROJECT_ROOT
            / "notebooks"
            / f"kaggle_v6_overlap40_pidnet_s_seed{seed}.ipynb"
        )
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        assert "--skip-test-evaluation" in code
        assert "--eval-only" not in code
        for cell in notebook["cells"]:
            source = "".join(cell.get("source", []))
            if cell.get("cell_type") == "code" and not source.lstrip().startswith("!"):
                ast.parse(source)
