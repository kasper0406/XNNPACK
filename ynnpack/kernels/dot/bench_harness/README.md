# dot schedule_bench harness

Scripts for evaluating the `schedule_dot` change in `kn/dot-schedule-l2`.

## What we're comparing

The scheduling-only PR (`kn/dot-schedule-l2`) changes two things in
`schedule_dot`:

1. **K-block sizing formula.** Old: `kc` sized so that a per-kernel-call
   `kc × block_n` stripe fits in the cache budget. New: sized so that the
   cross-m-iteration `kc × n` stripe fits.
2. **Fixed cache guess.** Old: 128 KiB (`cache_size_l2` in `dot.cc`). New:
   1 MiB. Still a fixed guess, not `cpuinfo`-derived — tiling must be
   consistent across hardware for non-associative float math to agree.

To isolate the two effects (formula vs. cache size), we drive
`schedule_bench` with the `auto:<cache_size>` CLI flag introduced on
`kn/dot-schedule-l2`. This lets one binary produce schedules for arbitrary
cache sizes; the formula is baked into whichever branch the binary was
built from.

## Configurations being benchmarked

| label           | binary branch         | `auto:<cache>` | represents                                    |
|-----------------|-----------------------|----------------|-----------------------------------------------|
| `master(128K)`  | master + CLI patch    | `auto:131072`  | current production (old formula, L1 guess)   |
| `sched(1M)`     | `kn/dot-schedule-l2`  | `auto:1048576` | proposed scheduling-only PR default          |
| `sched(2M)`     | `kn/dot-schedule-l2`  | `auto:2097152` | 2 MiB fixed guess variant                    |
| `sched(3M)`     | `kn/dot-schedule-l2`  | `auto:3145728` | what `cpuinfo` reports on Apple M-series      |

The `auto:` CLI did not exist on master, so to benchmark master's old formula
you apply `master_auto_cli.patch` to a master checkout before building.
`schedule.cc` is not touched by that patch, so the binary uses master's
scheduling logic.

## One-time setup

```sh
# 1. Build scheduling-only binary
git checkout kn/dot-schedule-l2
bazel build -c opt //ynnpack/kernels/dot:schedule_bench
cp bazel-bin/ynnpack/kernels/dot/schedule_bench /tmp/xnn-bench/schedule_bench.sched-only

# 2. Build master binary with auto: CLI.
#    Use a worktree so you don't disturb your working branch.
git worktree add /tmp/xnn-master master
cd /tmp/xnn-master
git apply ynnpack/kernels/dot/bench_harness/master_auto_cli.patch
bazel build -c opt //ynnpack/kernels/dot:schedule_bench
cp bazel-bin/ynnpack/kernels/dot/schedule_bench /tmp/xnn-bench/schedule_bench.master
cd -
```

`/tmp/xnn-bench` is the hard-coded binary path in `run_sweep.sh` — override
with `BENCH_DIR=/path/to/dir` if you prefer somewhere else.

## Running

```sh
mkdir -p /tmp/xnn-bench
N=3 ynnpack/kernels/dot/bench_harness/run_sweep.sh | tee results.txt
```

`N` is the best-of count (default 3). Each individual kernel run
self-calibrates for ~0.5 s of wall time inside the binary, so a full sweep
takes roughly 2 kernels × 11 shapes × 4 cache sizes × N × 0.5 s ≈ 2 min per
N, plus warm-up overhead.

### Avoiding thermal throttling on laptops

Apple M-series throttles aggressively under sustained GEMM load — a 1-min
unbroken sweep drops bf16 throughput to ~20 % of nominal, which silently
corrupts the comparison. Two mitigations:

- **Space out runs**: set `SLEEP=30` (seconds between shape/kernel
  combinations). Extends wall time but keeps the baseline stable.
- **Plug in + cold start**: warm-weather laptop benchmarks need the chassis
  cool to begin with.

If you see the same shape report radically different GFLOPS on repeat runs
(e.g. 60 GFLOPS → 13 GFLOPS on the bf16 kernel) the CPU is throttled.

A known-good sanity check before trusting results:

```sh
/tmp/xnn-bench/schedule_bench.master \
  dot_bf16_bf16_fp32_8x8x8_2x4x4_neonbf16 1024x1024x1024 auto:131072
```

Should be ~60 GFLOPS on Apple M; under thermal throttling drops to ~13.

## Shapes

Chosen to cover the ranges where the old/new formulas diverge:

- **Small/medium square** (`512³`, `1024³`, `1024x1024x4096`): B fits in
  modest cache either way — expect near-neutral.
- **Large square** (`2048³`, `4096x4096x1024`): old formula's kc is too big
  for any real cache, new formula actually hits L2 reuse — expect wins.
- **Small M, large N+K** (`128x4096x4096`, `256x8192x2048`): few m-iters
  per B stripe → sensitive to fixed-guess size.
- **LLM FFN widths** (`512x11008x4096`, `2048x11008x4096`,
  `512x4096x4096`): very wide B, stripe doesn't fit in L2 at any
  reasonable kc — the "kc gets squeezed to ~20 elements" failure mode
  of the new formula shows up here.
- **Wide M + large K** (`4096x1024x4096`): stresses the m-outer vs n-outer
  decision.

## Kernels

- `dot_fp32_4x16x4_1x4x1_neon` — plain NEON f32; the simplest case.
- `dot_bf16_bf16_fp32_8x8x8_2x4x4_neonbf16` — NEON bf16 with
  `transpose_a`; different block shape exercises `block_n`, `block_k`
  denominators differently in the kc formula.
