#!/bin/sh
# Controlled timing on the TWR-CLOUD-Bern evaluation corpus, for speed only. Each named method and mode runs alone,
# SOLID_TIMING_WORKERS single-threaded CLI processes, pinned by SOLID_TIMING_PIN when it is set, into a timing folder of
# its own so nothing is taken from a cache. The paper's runs took 90 processes pinned to the first hardware thread of 90
# physical cores, SOLID_TIMING_WORKERS=90 and SOLID_TIMING_PIN="taskset -c 0-89" (CPUs 0 to 89, whose siblings 96 to
# 185 stayed idle). Before every run the script waits until the machine is idle, a load below 2 over the last minute and
# no process of any user above half a core, for two checks a minute apart. It writes conditions.json beside the run, with
# the start and end in UTC and the load and the busiest processes before and after. The process that measures, and the
# ps it starts, are left out of every check and of the busiest processes, since a process that has just started can show
# a whole core. The accuracy of every method stays with the evaluation runs. The checks read /proc/loadavg and GNU ps, so
# the script runs on Linux. Writes only inside SOLID_TIMING.
#
# Settings, from the environment:
#   SOLID_TIMING          the folder of the runs (bern/METHOD/MODE), the log and the temporary files, out/timing of the
#                         checkout unless set
#   SOLID_TIMING_WORKERS  the processes of a run, every online processor unless set
#   SOLID_TIMING_PIN      the command that pins them, for example "taskset -c 0-89", none unless set
#   PYTHONPATH            as it is set, for the Python packages of the runner
#
# Usage: sh experiments/timing/timing_bern.sh METHOD:MODE [METHOD:MODE ...]
#        for example ade-bp-2015:anc
ROOT=$(cd "$(dirname "$0")/../.." && pwd) || exit 1
R=${SOLID_TIMING:-$ROOT/out/timing}
WORKERS=${SOLID_TIMING_WORKERS:-$(getconf _NPROCESSORS_ONLN)}
PIN=${SOLID_TIMING_PIN:-}
LOG=$R/timing-bern.log
[ "$#" -ge 1 ] || { echo "usage: sh $0 METHOD:MODE [METHOD:MODE ...]" >&2; exit 2; }
cd "$ROOT" || exit 1
export RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TMPDIR=$R/tmp TMP=$R/tmp TEMP=$R/tmp
mkdir -p "$TMPDIR" "$R/bern"
note() { echo "$(date -u '+%F %H:%M:%S') $*" >> "$LOG"; }
busy() {
  # the load over the last minute, and the busiest processes of every user other than this measurement, as JSON, or
  # with the argument idle no output and the exit status 0 when the machine is idle
  python3 - "$@" << 'PY'
import json, os, subprocess, sys
load = [float(x) for x in open("/proc/loadavg").read().split()[:3]]
lines = subprocess.run(["ps", "-eo", "pcpu,user,pid,ppid,comm", "--sort=-pcpu", "--no-headers"], capture_output=True, text=True).stdout.splitlines()
me = os.getpid()
rows = [dict(zip(("pcpu", "user", "pid", "ppid", "command"), line.split(None, 4))) for line in lines]
top = [dict(pcpu=r["pcpu"], user=r["user"], pid=r["pid"], command=r["command"]) for r in rows if me not in (int(r["pid"]), int(r["ppid"]))][:8]
if sys.argv[1:] == ["idle"]:
    sys.exit(0 if load[0] < 2 and all(float(p["pcpu"]) <= 50 for p in top) else 1)
print(json.dumps(dict(load=load, busiest=top)))
PY
}
idle() {
  busy idle
}
for pair in "$@"; do
  method=${pair%%:*}
  mode=${pair#*:}
  out=$R/bern/$method/$mode
  if [ -e "$out" ]; then
    note "$method $mode refused, its timing folder exists"
    exit 1
  fi
  waited=0
  while ! { idle && sleep 60 && idle; }; do
    waited=$((waited + 1))
    [ "$waited" -le 360 ] || { note "$method $mode refused, the machine was not idle for six hours"; exit 1; }
    sleep 60
  done
  before=$(busy)
  started=$(date -u '+%FT%TZ')
  note "$method $mode starts on an idle machine"
  $PIN python3 research/benchmarks/bern/run_bern.py --method "$method" --mode "$mode" --output "$out" --workers "$WORKERS" >> "$LOG" 2>&1
  status=$?
  ended=$(date -u '+%FT%TZ')
  after=$(busy)
  note "$method $mode exit $status"
  mkdir -p "$out"
  printf '{"method": "%s", "mode": "%s", "started": "%s", "ended": "%s", "exit": %s, "before": %s, "after": %s}\n' \
    "$method" "$mode" "$started" "$ended" "$status" "$before" "$after" > "$out/conditions.json"
  [ "$status" -eq 0 ] || exit "$status"
done
