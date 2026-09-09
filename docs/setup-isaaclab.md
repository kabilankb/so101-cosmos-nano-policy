# Setting up the Isaac Lab side (the client)

The environment that runs the digital twin and drives the policy: `so101_bench` on Isaac
Lab, under Isaac Sim's Kit python. This is the **client** half — the server half is
[setup-cosmos.md](setup-cosmos.md).

The single most common way to lose an afternoon here is picking the wrong interpreter, so
that comes first.

- [Use the Kit python](#use-the-kit-python)
- [One-time install](#one-time-install)
- [Scene assets](#scene-assets)
- [Move-task footprint metadata](#move-task-footprint-metadata)
- [Smoke test: task registration](#smoke-test-task-registration)
- [Full scene test: cameras and objects](#full-scene-test-cameras-and-objects)
- [Unit tests](#unit-tests)
- [The shadowed-checkout trap](#the-shadowed-checkout-trap)
- [Patches cosmos3_eval.py needs](#patches-cosmos3_evalpy-needs)
- [Episode layouts](#episode-layouts)
- [Real policy round-trip](#real-policy-round-trip)

---

## Use the Kit python

```
/path/to/IsaacLab/_isaac_sim/python.sh
```

On the verified box **no conda environment has `omni` importable**. What makes this
expensive is that `isaaclab` *alone* does import in at least one of them, so an environment
looks viable right up until Isaac Sim fails to boot. The Kit python has `omni`, `isaacsim`,
`isaaclab` and `so101_bench` together.

`pipeline.evaluate()` hardcodes this interpreter via the `kit_python` setting; there is no
code path that runs the client under anything else.

**On aarch64** (DGX Spark) Isaac Sim aborts at boot — a hard abort before your script runs,
not a hang — without:

```shell
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
```

Not needed on x86_64.

## One-time install

```shell
cd /path/to/so101_bench

/path/to/IsaacLab/_isaac_sim/python.sh -m pip install -e source/so101_bench
/path/to/IsaacLab/_isaac_sim/python.sh -m pip install openpi-client
```

`openpi-client` is the WebSocket client the eval uses to reach the policy server. It must
be installed **into the Kit python**, not into whatever environment is active in your shell.

## Scene assets

About 430 MB of USD — room, arm, plastic bin, ~50 household objects:

```shell
hf download 5hadytru/so101_bench_assets so101_bench_usd_assets.tar.gz \
  --repo-type dataset --local-dir /tmp/so101_assets

tar -xzf /tmp/so101_assets/so101_bench_usd_assets.tar.gz \
  -C source/so101_bench/so101_bench/assets/
```

## Move-task footprint metadata

```shell
PYTHONPATH= python3 -u scripts/generate_object_move_footprints.py
```

The repo README says these JSONs ship committed. **They do not** in a fresh clone. Skip this
and every episode referencing a move task fails with `Missing generated move-task footprint
metadata`.

Two details worth knowing if you go looking at them: the JSON key is `boxes`, not
`move_footprint_boxes`, and `load_object_move_footprint_boxes` is called with
`required=False`, so missing files degrade *silently* rather than erroring. Check the count
matches the USD count:

```shell
ls source/so101_bench/so101_bench/assets/objects/*.json | wc -l
```

For the focus5 bin-placement evaluation these are not on the critical path — they matter for
the `Move` task family.

## Smoke test: task registration

No rendering, no scene load, ~10 s:

```shell
PYTHONPATH= python3 -u scripts/list_envs.py
```

Expect all **11** `So101Bench-*` tasks: `-Bin-v0`, `-NamedBin-v0`, `-Bin-SingleObject-v0`,
four `-Bin-ObjectN-v0` variants, `-NextTo-v0`, `-Between-v0`, `-Move-v0`, `-Mixed-v0`.

## Full scene test: cameras and objects

Boots the renderer, loads the scene and arm, resets episode 0 and prints where everything
landed — without stepping physics or querying a policy:

```shell
timeout 120 python3 -u scripts/cosmos3_eval.py \
  --task So101Bench-Bin-v0 \
  --episodes_jsonl tasks/test.jsonl \
  --inspect_initial_scene --headless
```

Expect both cameras found (`wrist`, `overhead`, 640×480), the episode's objects and bin
printed with real coordinates, then:

```
Inspecting initial scene. Close the Isaac app window to exit
```

**That message is the pass condition.** Headless mode has no window to close, so the process
idles there until `timeout` kills it — **exit code 124 is expected, not a failure**.

## Unit tests

```shell
PYTHONPATH= python3 -m pytest tests/ -v
```

16 cases, under a second, no Isaac Sim boot: `SO101JointMapper` calibration math and
round-trips, plus `Cosmos3RemotePolicy` against a mocked WebSocket client — request shape,
the raw-unflipped-gripper contract, action-chunk queuing and error paths. A `conftest.py`
registers a synthetic `so101_bench.utils` namespace package so the pure-Python policy client
imports without triggering the full Isaac Sim runtime.

## The shadowed-checkout trap

```
ModuleNotFoundError: so101_bench.utils.cosmos3
```

`so101_bench` may be pip-installed **editable pointing at a different checkout**. Check:

```shell
/path/to/IsaacLab/_isaac_sim/python.sh -m pip show so101_bench | grep Editable
```

On the verified box it resolved to a GR00T-era checkout whose `utils/` has `groot.py` and
`molmoact2.py` but no `cosmos3.py`, and which has no `scripts/cosmos3_eval.py` at all.

`pipeline.evaluate()` sets `PYTHONPATH=<bench>/source/so101_bench` on every client launch,
which shadows the wrong checkout **without disturbing that other project**. That is why the
prefix is mandatory rather than cosmetic. To fix it permanently instead — this breaks
whatever project depends on the other checkout:

```shell
cd /path/to/so101_bench
/path/to/IsaacLab/_isaac_sim/python.sh -m pip install -e source/so101_bench
```

> Testing `import so101_bench` from the Isaac Lab directory gives a **false pass** — the
> `so101_bench/` directory there is picked up as an implicit namespace package. Check from a
> different working directory, or check `pip show`.

`so101 doctor` runs this check (skip it with `--quick`; it boots the Kit python and takes a
minute).

## Patches `cosmos3_eval.py` needs

`scripts/cosmos3_eval.py` is **untracked** in the `so101_bench` repo and was written against
a newer `layouts.py` than the checkout ships. Out of the box:

```
TypeError: generate_episode_layout() got an unexpected keyword argument 'placement_history'
```

The tracked `scripts/groot_eval.py` is the reference for how the current library expects to
be called. Three changes:

1. **Imports** — add `object_metadata`, `object_usd_stem` from `so101_bench.benchmark`;
   `DEFAULT_BIN_FOOTPRINT_HALF_EXTENTS`, `DEFAULT_OBJECT_FOOTPRINT_HALF_EXTENTS` from
   `so101_bench.layouts`; `ASSETS_PATH`, `MOVE_STRAIGHTNESS_TOLERANCE_M`, `TABLE_BOUNDS`
   from `so101_bench_env_cfg`. Add `MULTI_RIGID_BODY_BIN_CLEARANCE_MARGIN_M = 0.5 * INCH`.
2. **Port the footprint helpers** from `groot_eval.py` — `_usd_footprint`,
   `_object_footprint_half_extents`, `_bin_footprint_half_extents`,
   `_episode_object_footprints`. `cosmos3_eval.py` shipped with none of them.
3. **Rebuild the `generate_episode_layout(...)` callsite** to match `groot_eval.py`: drop
   `placement_history`, add the required `object_footprint_half_extents` plus
   `bin_footprint_half_extents`, `table_bounds`, `move_straightness_tolerance_m` and
   `sample_random_valid_spatial_layout`.

Dropping `placement_history` loses the intended inter-episode placement spread; variance now
comes from the shared `layout_rng` alone, exactly as `groot_eval.py` does it. The tracked
`layouts.py` has no such parameter, so preserving the feature would mean writing new
rejection-sampling logic into library code.

Two further local changes are already in place on the verified box and worth re-applying to
a fresh clone: `target_lift` telemetry emitted on plain timeouts (otherwise the dominant
failure mode logs no lift at all, and you cannot tell a near-grasp from no movement), and a
`--layout_max_attempts` flag wired to `max_attempts=`.

Also note `--record_dataset` calls `so101_bench.utils.lerobot_dataset`, which is a
placeholder in this checkout and raises `NotImplementedError` on first use. Leave it off;
plain pass/fail evaluation does not need it.

## Episode layouts

```
LayoutGenerationError: Could not sample a non-overlapping layout after 144 attempts.
Rejections: {'object_placement_failed': 10, 'bin_clearance_failed': 0,
             'object_clearance_failed': 0, 'task_feasibility_failed': 134}
```

144 is `DEFAULT_LAYOUT_MAX_ATTEMPTS` — the sampler ran out of budget on a genuinely tight
table (0.39 m × 0.255 m with a 3.5 inch move clear-path gap). Raise it:

```shell
--layout_max_attempts 4096
```

This is **pure rejection-sampling budget**: more tries, identical acceptance criteria, so
accepted layouts come from the same distribution. Nothing is relaxed.

A successful run writes `tasks/layouts/<name>_layouts_<timestamp>.jsonl`. Feed that back via
`--episode_layouts_jsonl` to skip sampling entirely — worth doing, because generation costs
~22 s per episode with a black viewport before the first frame renders, since every layout
is built up front.

**It must be a `focus5_layouts_*` file.** The older `test_layouts_*` files carry trial ids
for the retired `Move` trials only, and the script hard-errors on any requested trial id
missing from the layout file. `Settings.newest_layouts()` globs for exactly that prefix.

## Real policy round-trip

With a server already running (see [setup-cosmos.md](setup-cosmos.md) and the
[README](../README.md)):

```shell
so101 warmup      # one synthetic request -- proves the wire contract
so101 eval        # headless rollout
so101 eval --gui  # with the viewer
```

`--gui` needs a display. On a remote-desktop session the window appears on **that** desktop,
not in your SSH terminal; `DISPLAY=:0` is set automatically when the variable is unset or
empty. In the window, `P` saves the camera images actually being sent to the policy and `N`
skips an episode.

Order matters: the client connects once at startup and dies on refusal, and the server's
first request compiles for minutes. Serve, warm up, then evaluate.
