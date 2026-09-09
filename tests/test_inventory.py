"""Checkpoint inventory: what is merged, what is exported, what it costs."""

from __future__ import annotations

import json

from so101_cosmos import inventory
from so101_cosmos.config import Settings


def make_run(tmp_path):
    cfg = Settings(framework=tmp_path, run_dir="run")
    cfg.ckpt_dir.mkdir(parents=True)
    return cfg


def add_checkpoint(cfg, n, merged=False, exported=None):
    cfg.checkpoint_path(n).mkdir(parents=True, exist_ok=True)
    if merged:
        (cfg.ckpt_dir / f"iter_{n:09d}_merged").mkdir(exist_ok=True)
    if exported is not None:
        d = cfg.run_path / f"model_export_{exported}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "checkpoint.json").write_text(
            json.dumps({"checkpoint_path": f"/x/checkpoints/iter_{n:09d}_merged"})
        )


class TestExportIndex:
    def test_maps_iteration_from_provenance_not_the_directory_name(self, tmp_path):
        # An export from iter_250 sitting in model_export_500 is exactly the
        # mistake reading checkpoint.json catches.
        cfg = make_run(tmp_path)
        add_checkpoint(cfg, 250, exported=500)
        assert set(inventory.export_index(cfg)) == {250}

    def test_directory_without_provenance_is_ignored(self, tmp_path):
        cfg = make_run(tmp_path)
        (cfg.run_path / "model_export_999").mkdir(parents=True)
        assert inventory.export_index(cfg) == {}


class TestCheckpoints:
    def test_merged_intermediates_are_not_listed_as_checkpoints(self, tmp_path):
        cfg = make_run(tmp_path)
        add_checkpoint(cfg, 3750, merged=True)
        rows = inventory.checkpoints(cfg)
        assert [r["iteration"] for r in rows] == [3750]
        assert rows[0]["merged"] is True

    def test_servable_requires_an_export(self, tmp_path):
        cfg = make_run(tmp_path)
        add_checkpoint(cfg, 3500, merged=True)
        add_checkpoint(cfg, 3750, merged=True, exported=3750)
        by_iter = {r["iteration"]: r for r in inventory.checkpoints(cfg)}
        assert by_iter[3500]["servable"] is False
        assert by_iter[3750]["servable"] is True

    def test_epochs_use_the_configured_denominator(self, tmp_path):
        # 899 iterations per epoch, from the 28,760 train-split windows -- not
        # the dataset's total_frames, which would read ~2.7% at ~96% done.
        cfg = make_run(tmp_path)
        add_checkpoint(cfg, 3750)
        assert inventory.checkpoints(cfg)[0]["epochs"] == 4.17


class TestReclaim:
    def test_only_merged_dirs_with_an_export_are_reclaimable(self, tmp_path):
        cfg = make_run(tmp_path)
        add_checkpoint(cfg, 3500, merged=True)                    # not exported yet
        add_checkpoint(cfg, 3750, merged=True, exported=3750)     # safe to delete
        assert [m["iteration"] for m in inventory.merged_intermediates(cfg)] == [3750]

    def test_disk_reports_headroom(self, tmp_path):
        cfg = make_run(tmp_path)
        d = inventory.disk(cfg)
        assert d["free"] > 0
        assert d["per_checkpoint_estimate"] == 88 * 1024**3


class TestFormatting:
    def test_human_bytes(self):
        assert inventory.human_bytes(None) == "—"
        assert inventory.human_bytes(512) == "512B"
        assert inventory.human_bytes(88 * 1024**3) == "88.0G"


class TestLegacyHistory:
    def test_reads_the_old_monitors_job_records(self, tmp_path):
        # tools/train_monitor.py keyed the stage as `action` and the checkpoint
        # as a padded string. Its history predates this package and is worth keeping.
        cfg = make_run(tmp_path)
        legacy = tmp_path / "outputs/monitor_jobs"
        legacy.mkdir(parents=True)
        log = legacy / "eval_iter_000003750_20260821.log"
        log.write_text(
            "[INFO]: Episode 1/2: success=True, reason=success, length=11.67s\n"
            "[INFO]: Episode 2/2: success=False, reason=time_out, length=90.00s\n"
        )
        (legacy / "eval_iter_000003750_20260821.meta.json").write_text(
            json.dumps({"id": "x", "action": "eval", "checkpoint": "iter_000003750",
                        "status": "done", "log": str(log)})
        )
        history = inventory.eval_history(cfg)
        assert history[3750]["episodes"] == 2
        assert history[3750]["successes"] == 1

    def test_non_eval_legacy_jobs_are_ignored(self, tmp_path):
        cfg = make_run(tmp_path)
        legacy = tmp_path / "outputs/monitor_jobs"
        legacy.mkdir(parents=True)
        (legacy / "serve.meta.json").write_text(
            json.dumps({"id": "s", "action": "serve", "checkpoint": "iter_000003750",
                        "status": "done", "log": "/nowhere.log"})
        )
        assert inventory.eval_history(cfg) == {}

    def test_missing_legacy_directory_is_not_an_error(self, tmp_path):
        cfg = make_run(tmp_path)
        assert inventory.eval_history(cfg) == {}
