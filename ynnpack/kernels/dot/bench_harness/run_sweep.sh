#!/bin/bash
# Compare master vs kn/dot-schedule-l2 (new formula) across 4 cache-size
# guesses (128 K / 1 M / 2 M / 3 M) on a set of GEMM shapes.
#
# See README.md for setup. Expects these binaries to exist:
#
#   $BENCH_DIR/schedule_bench.master      # master + auto: CLI patch
#   $BENCH_DIR/schedule_bench.sched-only  # kn/dot-schedule-l2
#
# Env vars:
#   BENCH_DIR  — binary dir (default /tmp/xnn-bench)
#   N          — best-of-N per config (default 3)
#   SLEEP      — seconds to pause between shape/kernel combinations,
#                helpful on thermally-throttled laptops (default 0)

set -u
BENCH_DIR=${BENCH_DIR:-/tmp/xnn-bench}
MASTER=$BENCH_DIR/schedule_bench.master
SCHED=$BENCH_DIR/schedule_bench.sched-only
N=${N:-3}
SLEEP=${SLEEP:-0}

for b in "$MASTER" "$SCHED"; do
  if [ ! -x "$b" ]; then
    echo "Missing binary: $b" >&2
    echo "See README.md for build instructions." >&2
    exit 1
  fi
done

KERNELS=(
  dot_fp32_4x16x4_1x4x1_neon
  dot_bf16_bf16_fp32_8x8x8_2x4x4_neonbf16
)

SHAPES=(
  # small / medium square
  512x512x512
  1024x1024x1024
  2048x2048x2048
  # tall / wide
  4096x4096x1024
  1024x1024x4096
  # small M, large N+K
  128x4096x4096
  256x8192x2048
  # LLM FFN widths
  512x11008x4096
  512x4096x4096
  # wider
  4096x1024x4096
  2048x11008x4096
)

bestof() {
  local best=0 g
  for _ in $(seq 1 "$N"); do
    g=$("$@" 2>/dev/null | awk '/GFLOPS/ {print $(NF-1)}' | tail -1)
    best=$(awk -v a="$best" -v b="$g" 'BEGIN { print (b+0 > a+0 ? b : a) }')
  done
  printf '%s\n' "$best"
}

pct() {
  awk -v m="$1" -v s="$2" 'BEGIN { if (m+0>0) printf "%+.1f%%", 100*(s/m-1); else print "-" }'
}

printf '%-50s %-20s %12s %12s %12s %12s %12s %9s %9s %9s\n' \
  kernel shape 'master(128K)' 'sched(1M)' 'sched(2M)' 'sched(3M)' 'sched(512K)' '%1M/m' '%2M/m' '%3M/m'

for K in "${KERNELS[@]}"; do
  for S in "${SHAPES[@]}"; do
    M=$(bestof  "$MASTER" "$K" "$S" "auto:131072")
    S1=$(bestof "$SCHED"  "$K" "$S" "auto:1048576")
    S2=$(bestof "$SCHED"  "$K" "$S" "auto:2097152")
    S3=$(bestof "$SCHED"  "$K" "$S" "auto:3145728")
    S05=$(bestof "$SCHED" "$K" "$S" "auto:524288")
    printf '%-50s %-20s %12s %12s %12s %12s %12s %9s %9s %9s\n' \
      "$K" "$S" "$M" "$S1" "$S2" "$S3" "$S05" "$(pct "$M" "$S1")" "$(pct "$M" "$S2")" "$(pct "$M" "$S3")"
    [ "$SLEEP" -gt 0 ] && sleep "$SLEEP"
  done
done
