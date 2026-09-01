import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def notebook_source(name: str) -> str:
    notebook = load_json(PROJECT_ROOT / "notebooks" / name)
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )


def assert_seed_pair(seed42: dict, seed1337: dict, allowed_differences: set[str]) -> None:
    assert set(seed42) == set(seed1337)
    for key in seed42:
        if key not in allowed_differences:
            assert seed42[key] == seed1337[key], key
    assert [seed42["seed"], seed1337["seed"]] == [42, 1337]
    assert seed42["automatic_test_evaluation"] is False
    assert seed1337["automatic_test_evaluation"] is False


def test_swinv2_base_seed1337_changes_only_seed_identity():
    config_dir = PROJECT_ROOT / "configs"
    seed42 = load_json(config_dir / "v6_overlap40_stdl_swinv2_base_full_seed42.json")
    seed1337 = load_json(
        config_dir / "v6_overlap40_stdl_swinv2_base_full_seed1337.json"
    )
    assert_seed_pair(
        seed42,
        seed1337,
        {"experiment", "hypothesis", "comparison_scope", "seed", "run_name"},
    )
    assert seed1337["module"] == "stdl_swinv2_base"
    assert seed1337["batch_size"] == 4
    assert seed1337["accum_steps"] == 1


def test_dlinknet_seed_pair_uses_locked_full_protocol():
    config_dir = PROJECT_ROOT / "configs"
    seed42 = load_json(config_dir / "v6_overlap40_dlinknet_full_seed42.json")
    seed1337 = load_json(config_dir / "v6_overlap40_dlinknet_full_seed1337.json")
    assert_seed_pair(
        seed42,
        seed1337,
        {"experiment", "hypothesis", "comparison_scope", "seed", "run_name"},
    )
    assert seed1337["model"] == "DLinkNet"
    assert seed1337["channel_mode"] == "full"
    assert seed1337["batch_size"] * seed1337["accum_steps"] == 4


def test_seed1337_kaggle_notebooks_never_evaluate_test():
    dlinknet = notebook_source("kaggle_v6_overlap40_dlinknet_seed1337.ipynb")
    swin = notebook_source("kaggle_v6_overlap40_stdl_swinv2_base_seed1337.ipynb")
    assert "--skip-test-evaluation" in dlinknet
    assert "'--split', 'val'" in dlinknet
    assert "metrics['test'] is None" in dlinknet
    assert "automatic_test_evaluation'] is False" in dlinknet
    assert "automatic_test_evaluation'] is False" in swin
    assert "run_autodl_stdl_swin.py" in swin


def test_baseline_trainer_exposes_explicit_test_lock():
    source = (PROJECT_ROOT / "scripts" / "train_baseline.py").read_text(
        encoding="utf-8"
    )
    assert "--skip-test-evaluation" in source
    assert "Test evaluation is locked and was not executed." in source
    assert "'automatic_test_evaluation': not args.skip_test_evaluation" in source
