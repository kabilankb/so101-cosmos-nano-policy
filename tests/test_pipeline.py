"""The serving contract, asserted directly.

Every flag checked here fails *silently* in production: omit one and the server
starts happily and returns wrong actions. These tests are the only place that
mismatch is caught cheaply.
"""

from __future__ import annotations

import pytest

from so101_cosmos import pipeline
from so101_cosmos.config import Settings


@pytest.fixture
def cfg(tmp_path):
    """A settings object rooted in a tmp tree with one exported checkpoint."""
    framework = tmp_path / "cosmos-framework"
    run = framework / "outputs/run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "model_export_3750").mkdir(parents=True)
    (framework / ".venv/bin").mkdir(parents=True)
    (framework / ".venv/lib/python3.13/site-packages/nvidia/cu13/lib").mkdir(parents=True)
    bench = tmp_path / "so101_bench"
    (bench / "tasks/layouts").mkdir(parents=True)
    return Settings(framework=framework, bench=bench, run_dir="outputs/run",
                    kit_python=tmp_path / "kit/python.sh")


def flags(argv: list[str]) -> dict[str, str | None]:
    """Flatten ``--flag value`` pairs; a bare flag maps to None."""
    out: dict[str, str | None] = {}
    for i, token in enumerate(argv):
        if token.startswith("--"):
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            out[token] = None if nxt is None or nxt.startswith("--") else nxt
    return out


class TestServe:
    def test_requires_an_export(self, cfg):
        with pytest.raises(FileNotFoundError, match="run merge, then export"):
            pipeline.serve(cfg, 250)

    def test_carries_every_domain_flag(self, cfg):
        argv, cwd, _ = pipeline.serve(cfg, 3750)
        f = flags(argv)
        assert f["--domain-name"] == "so101"
        assert f["--action-dim"] == "6"
        assert f["--arm-joint-dim"] == "5"  # not DROID's 7
        assert f["--action-space"] == "joint_pos"
        assert f["--conditioning-fps"] == "30"
        assert cwd == cfg.framework

    def test_gripper_flip_is_disabled(self, cfg):
        # The default 1.0-x flip is DROID's [0,1] convention; SO-101 trains on
        # raw LeRobot .pos [0,100], so the flip would send 1.0 - 87.3 = -86.3.
        argv, _, _ = pipeline.serve(cfg, 3750)
        assert "--no-flip-gripper" in argv
        assert "--flip-gripper" not in argv

    def test_normalization_is_never_half_configured(self, cfg):
        # Both or neither. With only one, normalized outputs (~[-1,1]) are
        # returned verbatim and every joint is commanded to about zero.
        argv, _, _ = pipeline.serve(cfg, 3750)
        f = flags(argv)
        assert f["--action-normalization"] == "minmax"
        assert f["--normalizer-stats-path"].endswith("so101_lerobot_stats.json")

    def test_view_description_names_two_views(self, cfg):
        # The server default is DROID's wording, which describes three views.
        argv, _, _ = pipeline.serve(cfg, 3750)
        desc = flags(argv)["--view-description"]
        assert "wrist" in desc and "overhead" in desc

    def test_device_memory_bytes_is_not_passed(self, cfg):
        # A no-op on a single GPU: passed only to _build_model_parallelism,
        # which ignores the value.
        argv, _, _ = pipeline.serve(cfg, 3750)
        assert "--device-memory-bytes" not in argv


class TestMergeAndExport:
    def test_merge_passes_the_runs_lora_scale(self, cfg):
        argv, _, _ = pipeline.merge(cfg, 3750)
        f = flags(argv)
        assert f["--lora-alpha"] == "32"
        assert f["--lora-rank"] == "16"
        assert f["--output"].endswith("iter_000003750_merged")

    def test_export_reads_the_merged_directory(self, cfg):
        argv, _, _ = pipeline.export(cfg, 3750)
        assert flags(argv)["--checkpoint-path"].endswith("_merged")

    def test_export_uses_the_registered_experiment_not_the_run_config(self, cfg):
        # The run's config.yaml still declares lora_enabled = true and demands
        # adapter keys the merge folded away.
        argv, _, _ = pipeline.export(cfg, 3750)
        f = flags(argv)
        assert f["--experiment"] == "action_policy_so101_nano_focus5"
        assert f["--config-file"] == "cosmos_framework/configs/base/config.py"
        assert "config.yaml" not in " ".join(argv)

    def test_export_disables_pretrained_and_ema(self, cfg):
        argv, _, _ = pipeline.export(cfg, 3750)
        joined = " ".join(argv)
        assert "model.config.ema.enabled=false" in joined
        assert "model.config.vlm_config.pretrained_weights.enabled=False" in joined


class TestEvaluate:
    def test_runs_under_kit_python_from_the_bench_checkout(self, cfg):
        argv, cwd, env = pipeline.evaluate(cfg)
        assert argv[0] == str(cfg.kit_python)
        assert cwd == cfg.bench
        assert "scripts/cosmos3_eval.py" in argv

    def test_pythonpath_shadows_the_editable_install(self, cfg):
        _, _, env = pipeline.evaluate(cfg)
        assert env["PYTHONPATH"] == str(cfg.bench / "source/so101_bench")

    def test_headless_unless_gui(self, cfg):
        assert "--headless" in pipeline.evaluate(cfg)[0]
        assert "--headless" not in pipeline.evaluate(cfg, gui=True)[0]

    def test_gui_sets_a_display(self, cfg):
        _, _, env = pipeline.evaluate(cfg, gui=True)
        assert env.get("DISPLAY")

    def test_layouts_are_passed_relative_to_the_bench(self, cfg):
        saved = cfg.bench / "tasks/layouts/focus5_layouts_20260821_132325.jsonl"
        saved.write_text("{}")
        argv, _, _ = pipeline.evaluate(cfg)
        assert flags(argv)["--episode_layouts_jsonl"] == \
            "tasks/layouts/focus5_layouts_20260821_132325.jsonl"

    def test_no_layouts_flag_when_none_saved(self, cfg):
        assert "--episode_layouts_jsonl" not in pipeline.evaluate(cfg)[0]


class TestEnvironment:
    def test_cuda_lib_is_prepended_to_ld_library_path(self, cfg, monkeypatch):
        # An inherited ROS profile puts system CUDA first, which loads
        # libcublasLt from one install and libcublas from another.
        monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64")
        env = pipeline.base_env(cfg)
        first = env["LD_LIBRARY_PATH"].split(":")[0]
        assert first.endswith("nvidia/cu13/lib")

    def test_so101_root_is_set_for_transform_metadata(self, cfg):
        env = pipeline.base_env(cfg)
        assert env["SO101_ROOT"].endswith("so101_bench_sim_6")

    def test_every_gpu_stage_carries_the_allocator_setting(self, cfg):
        for stage in ("merge", "export", "serve"):
            _, _, env = pipeline.build(cfg, stage, 3750)
            assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


class TestDispatch:
    def test_stage_needing_an_iteration_says_so(self, cfg):
        with pytest.raises(ValueError, match="needs an iteration"):
            pipeline.build(cfg, "serve")

    def test_unknown_stage_lists_the_known_ones(self, cfg):
        with pytest.raises(ValueError, match="unknown stage"):
            pipeline.build(cfg, "deploy")

    def test_sides_partition_the_stages(self):
        assert not pipeline.SERVER_STAGES & pipeline.CLIENT_STAGES
        assert set(pipeline.STAGES) == pipeline.SERVER_STAGES | pipeline.CLIENT_STAGES


class TestGuiDisplay:
    def test_empty_display_is_replaced_not_kept(self, cfg, monkeypatch):
        # An SSH session carries DISPLAY="" -- present, so setdefault would keep
        # it, and Isaac Sim would have no display to open a window on.
        monkeypatch.setenv("DISPLAY", "")
        _, _, env = pipeline.evaluate(cfg, gui=True)
        assert env["DISPLAY"] == ":0"

    def test_a_real_display_is_respected(self, cfg, monkeypatch):
        monkeypatch.setenv("DISPLAY", ":1")
        _, _, env = pipeline.evaluate(cfg, gui=True)
        assert env["DISPLAY"] == ":1"

    def test_headless_run_sets_no_display(self, cfg, monkeypatch):
        monkeypatch.delenv("DISPLAY", raising=False)
        _, _, env = pipeline.evaluate(cfg, gui=False)
        assert "DISPLAY" not in env
