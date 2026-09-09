"""Log parsing, against the real line shapes both tools emit."""

from __future__ import annotations

from so101_cosmos import logparse

EARLY = "[2026-08-18 20:41:03] Iteration 26: Hit counter: 26/50 | Loss: 1.7040 | Time: 35.58s"
SPEED = "[2026-08-19 02:11:44] 763 : iter_speed 35.80 seconds per iteration | Loss: 0.2592"

MOVE_EPISODE = (
    "[INFO]: Episode 1/4: success=False, reason=time_out, length=25.00s, "
    "failure_type=failed_grasp, live_failure_reason=none "
    "[target=object_3 (pink eraser), target_lift=0.00in, wrong_object=none, "
    "max_distractor_lift=0.01in, lift_threshold=0.50in]"
)
BIN_EPISODE = (
    "[INFO]: Episode 12/20: success=False, reason=time_out, length=90.00s, "
    "live_failure_reason=none"
)
WIN = "[INFO]: Episode 20/20: success=True, reason=success, length=11.67s, live_failure_reason=none"


def write(tmp_path, *lines):
    p = tmp_path / "run.log"
    p.write_text("\n".join(lines) + "\n")
    return p


class TestTraining:
    def test_reads_both_line_formats(self, tmp_path):
        # Grepping only the first makes a live run look stalled at iteration 550.
        pts = logparse.parse_training(write(tmp_path, EARLY, SPEED))
        assert [p["iter"] for p in pts] == [26, 763]
        assert pts[0]["loss"] == 1.7040
        assert pts[1]["sec"] == 35.80

    def test_later_lines_win_for_the_same_iteration(self, tmp_path):
        dup = "[x] 763 : iter_speed 40.00 seconds per iteration | Loss: 0.1000"
        pts = logparse.parse_training(write(tmp_path, SPEED, dup))
        assert len(pts) == 1
        assert pts[0]["loss"] == 0.1000

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert logparse.parse_training(tmp_path / "nope.log") == []


class TestEpisodes:
    def test_move_episode_carries_lift_telemetry(self, tmp_path):
        (ep,) = logparse.parse_episodes(write(tmp_path, MOVE_EPISODE))
        assert ep["success"] is False
        assert ep["reason"] == "time_out"
        assert ep["failure_type"] == "failed_grasp"
        assert ep["lift"] == 0.0
        assert ep["distractor"] == 0.01

    def test_bin_episode_yields_none_not_a_false_zero(self, tmp_path):
        # Lift is a move-task metric. A bin episode reporting 0.00 would read as
        # "measured, and it never moved" rather than "not measured".
        (ep,) = logparse.parse_episodes(write(tmp_path, BIN_EPISODE))
        assert ep["lift"] is None
        assert ep["failure_type"] == "none"
        assert ep["total"] == 20

    def test_summary_counts_successes(self, tmp_path):
        eps = logparse.parse_episodes(write(tmp_path, BIN_EPISODE, WIN))
        s = logparse.summarize(eps)
        assert s == {"episodes": 2, "successes": 1, "rate": 0.5, "total": 20}

    def test_summary_of_nothing_has_no_rate(self):
        assert logparse.summarize([])["rate"] is None


class TestErrors:
    def test_finds_a_poisoned_cuda_context(self, tmp_path):
        log = write(tmp_path, "fine", "RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED")
        assert "CUBLAS" in logparse.has_error(log)

    def test_clean_log_reports_nothing(self, tmp_path):
        assert logparse.has_error(write(tmp_path, EARLY, SPEED)) is None
