import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_rezero_cmcr_lr5e5_configs_are_controlled_optimizer_variants():
    config_dir = PROJECT_ROOT / "configs"
    pairs = (
        (
            "v6_overlap40_frozen_gated_rezero_cmcr_batch4_seed42.json",
            "v6_overlap40_frozen_gated_rezero_cmcr_lr5e5_batch4_seed42.json",
        ),
        (
            "v6_overlap40_frozen_gated_rezero_cmcr_batch4_seed1337.json",
            "v6_overlap40_frozen_gated_rezero_cmcr_lr5e5_batch4_seed1337.json",
        ),
    )
    allowed_differences = {
        "experiment",
        "hypothesis",
        "unique_variable",
        "epochs",
        "early_stopping_patience",
        "learning_rate",
        "run_name",
    }
    for control_name, optimized_name in pairs:
        control = json.loads((config_dir / control_name).read_text(encoding="utf-8"))
        optimized = json.loads((config_dir / optimized_name).read_text(encoding="utf-8"))
        assert set(control) == set(optimized)
        for key in control:
            if key not in allowed_differences:
                assert control[key] == optimized[key], key
        assert control["learning_rate"] == 1e-4
        assert optimized["learning_rate"] == 5e-5
        assert optimized["epochs"] == 60
        assert optimized["early_stopping_patience"] == 12
        assert optimized["automatic_test_evaluation"] is False
