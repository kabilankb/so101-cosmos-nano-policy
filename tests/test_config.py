"""Settings resolution: defaults, TOML, environment, and derived paths."""

from __future__ import annotations

import pytest

from so101_cosmos.config import ENV_PREFIX, Settings


class TestLoad:
    def test_defaults_need_no_files(self):
        cfg = Settings.load()
        assert cfg.policy_port == 8000
        assert cfg.arm_joint_dim == 5

    def test_toml_overrides_defaults(self, tmp_path):
        f = tmp_path / "s.toml"
        f.write_text('[so101]\npolicy_port = 8123\nrun_dir = "outputs/other"\n')
        cfg = Settings.load(f)
        assert cfg.policy_port == 8123
        assert cfg.run_dir == "outputs/other"

    def test_env_overrides_toml(self, tmp_path, monkeypatch):
        f = tmp_path / "s.toml"
        f.write_text("[so101]\npolicy_port = 8123\n")
        monkeypatch.setenv(ENV_PREFIX + "POLICY_PORT", "8999")
        assert Settings.load(f).policy_port == 8999

    def test_paths_are_expanded(self, monkeypatch):
        monkeypatch.setenv(ENV_PREFIX + "FRAMEWORK", "~/somewhere")
        assert "~" not in str(Settings.load().framework)

    def test_unknown_key_is_rejected(self, tmp_path):
        f = tmp_path / "s.toml"
        f.write_text('[so101]\ngpu_count = 8\n')
        with pytest.raises(ValueError, match="gpu_count"):
            Settings.load(f)

    def test_missing_file_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Settings.load(tmp_path / "absent.toml")

    def test_env_prefix_does_not_collide_with_cosmos_so101_root(self, monkeypatch):
        # cosmos-framework reads SO101_ROOT itself; setting it must not reach us.
        monkeypatch.setenv("SO101_ROOT", "/somewhere/else")
        assert Settings.load().framework == Settings().framework


class TestDerived:
    def test_export_and_checkpoint_paths(self):
        cfg = Settings()
        assert cfg.checkpoint_path(3750).name == "iter_000003750"
        assert cfg.export_path(3750).name == "model_export_3750"

    def test_latest_iteration_reads_the_marker(self, tmp_path):
        cfg = Settings(framework=tmp_path, run_dir="run")
        cfg.ckpt_dir.mkdir(parents=True)
        (cfg.ckpt_dir / "latest_checkpoint.txt").write_text("iter_000004000\n")
        assert cfg.latest_iteration() == 4000

    def test_latest_iteration_absent_is_none(self, tmp_path):
        assert Settings(framework=tmp_path, run_dir="run").latest_iteration() is None

    def test_cuda_lib_found_by_glob_not_a_hardcoded_version(self, tmp_path):
        cfg = Settings(framework=tmp_path)
        (tmp_path / ".venv/lib/python3.14/site-packages/nvidia/cu13/lib").mkdir(parents=True)
        assert cfg.cuda_lib is not None
        assert "python3.14" in str(cfg.cuda_lib)

    def test_cuda_lib_missing_is_none(self, tmp_path):
        assert Settings(framework=tmp_path).cuda_lib is None

    def test_newest_layouts_picks_the_last(self, tmp_path):
        cfg = Settings(bench=tmp_path)
        (tmp_path / "tasks/layouts").mkdir(parents=True)
        for stamp in ("20260819_100000", "20260821_132325"):
            (tmp_path / f"tasks/layouts/focus5_layouts_{stamp}.jsonl").write_text("{}")
        assert "20260821_132325" in cfg.newest_layouts().name

    def test_retired_test_layouts_are_not_matched(self, tmp_path):
        # test_layouts_* carry trial ids for the retired Move trials; the eval
        # script hard-errors on any requested id missing from the file.
        cfg = Settings(bench=tmp_path)
        (tmp_path / "tasks/layouts").mkdir(parents=True)
        (tmp_path / "tasks/layouts/test_layouts_20260818_105329.jsonl").write_text("{}")
        assert cfg.newest_layouts() is None
