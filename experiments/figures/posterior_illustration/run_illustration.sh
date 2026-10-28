#!/bin/sh
# The posterior illustration, built in plain copies of the frozen sources, on Linux.
#
# The script copies the sources both builds need from the checkout $REPO into $WORK/figures/posterior-illustration/src. It
# adds illustration.rs of this folder to the runtime copy as engine/src/illustration.rs, declared at the end of the
# copy's lib.rs under cfg(test), and dump_problem.rs to the research copy as the example
# cli/examples/illustration_dump.rs, which cargo finds without a change to any manifest, so both lock files stay as they
# are. It builds with the toolchain rust-toolchain.toml pins and with --locked. The list command prints the anchor only
# scenarios of an archive. The run command writes the input of one anchor only unit with the dump and then the record of
# its adaptive posteriors with the engine test, under RAYON_NUM_THREADS=1 and TMPDIR=$WORK/tmp.
#
# The script writes its files nowhere but in $WORK/figures/posterior-illustration, the base. Every path it writes is
# resolved with realpath first, symbolic links included, and one that resolves outside the base stops the script before
# anything is written. It deletes only the previous copy of the sources, a stale record and its own lock, all inside the
# base and after the same check. The copy must hold no symbolic link. The compiler and the linker keep their temporary
# files in $WORK/tmp, which must exist already, cargo and rustup keep using their own caches as for every build, and
# cargo's automatic cleaning of its cache is off while the script runs. A lock in the base keeps two runs apart.
#
# Usage
#   sh run_illustration.sh list ARCHIVE [SAMPLES [ITERATION]]
#   sh run_illustration.sh run ARCHIVE SCENARIO [SAMPLES [ITERATION [PROFILE [AGENT]]]]
#
# REPO names the checkout, the one that holds this script unless set, and WORK the folder the script writes in, ./work
# unless set. ARCHIVE is best given as an absolute path. SAMPLES is 10 and ITERATION 1 unless given. PROFILE is
# $REPO/research/artifacts/solid/profiles/bern-anchor-only.json unless given, and AGENT keeps the decisions of that agent
# alone in the record. The names of the
# outputs carry the name of the archive's folder, scan-01 for example, and the stem of the profile. The list goes to
# BASE/list-FOLDER-samples-SAMPLES-iteration-ITERATION.tsv and a run to
# BASE/runs/FOLDER-scenario-SCENARIO-samples-SAMPLES-iteration-ITERATION-PROFILE, with input.json, record.json and logs.
set -eu

unset CDPATH
here=$(cd "$(dirname "$0")" && pwd -P)
REPO=${REPO:-$(cd "$here/../../.." && pwd -P)}
WORK=${WORK:-./work}

refuse() {
  echo "run_illustration.sh stops, $*" >&2
  exit 1
}

usage() {
  echo "usage  sh $0 list ARCHIVE [SAMPLES [ITERATION]]" >&2
  echo "       sh $0 run ARCHIVE SCENARIO [SAMPLES [ITERATION [PROFILE [AGENT]]]]" >&2
  exit 2
}

# Stops unless $1 is a whole number, positive unless $3 is "zero".
whole() {
  case "$1" in
    '' | *[!0-9]*) refuse "$2 must be a whole number, not '$1'" ;;
  esac
  if [ "${3:-}" != zero ] && [ "$1" -eq 0 ]; then
    refuse "$2 must be positive"
  fi
}

[ "$#" -ge 2 ] || usage
command=$1
archive=$2
scenario=
profile=
agent=
case "$command" in
  list)
    [ "$#" -le 4 ] || usage
    samples=${3:-10}
    iteration=${4:-1}
    ;;
  run)
    [ "$#" -ge 3 ] && [ "$#" -le 7 ] || usage
    scenario=$3
    samples=${4:-10}
    iteration=${5:-1}
    profile=${6:-}
    agent=${7:-}
    whole "$scenario" SCENARIO zero
    if [ -n "$agent" ]; then
      whole "$agent" AGENT zero
    fi
    ;;
  *)
    usage
    ;;
esac
whole "$samples" SAMPLES
whole "$iteration" ITERATION

# the folders the script reads, resolved before it changes directory
[ -d "$WORK" ] || refuse "the work folder $WORK does not exist"
root=$(realpath -e -- "$WORK")
[ -d "$REPO" ] || refuse "the checkout $REPO does not exist"
repo=$(realpath -e -- "$REPO")
[ -d "$repo/runtime/engine/src" ] && [ -f "$repo/research/Cargo.toml" ] || refuse "$repo does not hold the runtime and research sources"
[ -f "$archive" ] || refuse "the archive $archive does not exist"
archive=$(realpath -e -- "$archive")
if [ "$command" = run ]; then
  profile=${profile:-$repo/research/artifacts/solid/profiles/bern-anchor-only.json}
  [ -f "$profile" ] || refuse "the deployment profile $profile does not exist"
  profile=$(realpath -e -- "$profile")
fi
[ -f "$here/illustration.rs" ] && [ -f "$here/dump_problem.rs" ] || refuse "$here lacks illustration.rs or dump_problem.rs"
[ -d "$root/tmp" ] || refuse "$root/tmp does not exist, and the script creates nothing outside its own folder"

# the base, which no symbolic link may lead elsewhere
base=$root/figures/posterior-illustration
[ "$(realpath -m -- "$base")" = "$base" ] || refuse "$base resolves elsewhere through a symbolic link"

# Whether $1 resolves inside the base, symbolic links resolved and missing parts taken as they are.
inside() {
  resolved=$(realpath -m -- "$1") || return 1
  case "$resolved/" in
    "$base"/*) return 0 ;;
    *) return 1 ;;
  esac
}

# Stops unless every argument resolves inside the base.
writable() {
  for candidate in "$@"; do
    inside "$candidate" || refuse "$candidate resolves outside $base"
  done
}

# the name of a file as a safe part of an output's name
tag() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'
}

src=$base/src
targets=$base/target
logs=$base/logs
lock=$base/lock
folder=$(tag "$(basename "$(dirname "$archive")")")
if [ "$command" = list ]; then
  listing=$base/list-$folder-samples-$samples-iteration-$iteration.tsv
  writable "$base" "$lock" "$src" "$targets/research" "$logs" "$logs/build-dump.log" "$logs/sources.sha256" "$listing"
else
  run=$base/runs/$folder-scenario-$scenario-samples-$samples-iteration-$iteration-$(tag "$(basename "$profile" .json)")
  writable "$base" "$lock" "$src" "$targets/research" "$targets/runtime" "$logs" "$logs/build-dump.log" "$logs/build-test.log" \
    "$logs/sources.sha256" "$run" "$run/input.json" "$run/record.json" "$run/dump.log" "$run/test.log"
fi

if ! command -v cargo > /dev/null 2>&1 && [ -x "$HOME/.cargo/bin/cargo" ]; then
  PATH=$HOME/.cargo/bin:$PATH
  export PATH
fi
command -v cargo > /dev/null 2>&1 || refuse "cargo is not on PATH"
TMPDIR=$root/tmp
RAYON_NUM_THREADS=1
CARGO_CACHE_AUTO_CLEAN_FREQUENCY=never
export TMPDIR RAYON_NUM_THREADS CARGO_CACHE_AUTO_CLEAN_FREQUENCY
unset ILLUSTRATION_INPUT ILLUSTRATION_OUTPUT ILLUSTRATION_AGENT

mkdir -p "$base"
# one run at a time, since every run refreshes the one copy of the sources
if ! mkdir "$lock" 2> /dev/null; then
  refuse "$lock exists, so another run is active, or an interrupted one left it and it may be removed by hand"
fi
trap 'rmdir "$lock"' EXIT
trap 'exit 1' HUP INT TERM
mkdir -p "$logs"

# a fresh copy of the sources on every run, the previous one deleted inside the base only
if [ -L "$src" ]; then
  refuse "$src is a symbolic link"
fi
if [ -e "$src" ]; then
  rm -rf -- "$src"
fi
mkdir -p "$src"

# Copies one file or folder of the checkout to the same place in the copy.
copy() {
  [ -e "$repo/$1" ] || refuse "$repo/$1 is missing"
  writable "$src/$1"
  mkdir -p "$(dirname "$src/$1")"
  cp -a "$repo/$1" "$src/$1"
}

# the toolchain, the cargo settings of OpenBLAS, and every manifest and source cargo reads to load and build both
# workspaces, without their target, data and artifact folders
for item in rust-toolchain.toml \
  runtime/Cargo.toml runtime/Cargo.lock runtime/engine/Cargo.toml runtime/engine/src runtime/ffi/Cargo.toml runtime/ffi/src \
  research/Cargo.toml research/Cargo.lock research/cli/Cargo.toml research/cli/build.rs research/cli/src \
  research/evaluation/Cargo.toml research/evaluation/src research/io/Cargo.toml research/io/src \
  research/methods/Cargo.toml research/methods/src \
  research/benchmarks/bern/calibration/fitter/Cargo.toml research/benchmarks/bern/calibration/fitter/src; do
  copy "$item"
done
# the viewer, a member of the workspace where the checkout holds it
for item in .cargo/config.toml runtime/engine/tests research/viewer/src-tauri/Cargo.toml research/viewer/src-tauri/src \
  research/viewer/src-tauri/build.rs; do
  if [ -e "$repo/$item" ]; then
    copy "$item"
  fi
done
# the executables of solid-cli live beside their drivers, and its manifest names them
for file in "$repo"/research/benchmarks/datasets/*.rs "$repo"/research/solid/*.rs; do
  if [ -f "$file" ]; then
    copy "${file#"$repo"/}"
  fi
done
if [ -n "$(find "$src" -type l -print | head -n 1)" ]; then
  refuse "the copy holds a symbolic link, $(find "$src" -type l -print | head -n 1)"
fi
(cd "$src" && find . -type f -exec sha256sum {} + | LC_ALL=C sort -k 2) > "$logs/sources.sha256"
# illustration.rs copies its helpers from these two files of commit 06c1f58. Other files may still hold the same helpers,
# and the test's assertions decide, so a difference is reported and not refused.
for pair in "3f44b547364a9ea9d220ceb6deade9bb2ff0660364e8f7cfdab22bd08de91b55 runtime/engine/src/anchor_only.rs" \
  "5c968ff5501bab709542677c8b2f5a881211747d2399b1971434501e2697fece runtime/engine/src/posterior.rs"; do
  expected=${pair%% *}
  file=${pair#* }
  if [ "$(sha256sum "$src/$file" | cut -d ' ' -f 1)" != "$expected" ]; then
    echo "note, $file of $repo differs from the one illustration.rs copies its helpers from, the test's assertions will show whether they still agree" >&2
  fi
done

# the two new files, the only additions to the copy
lib=$src/runtime/engine/src/lib.rs
module=$src/runtime/engine/src/illustration.rs
example=$src/research/cli/examples/illustration_dump.rs
writable "$lib" "$module" "$example"
[ ! -e "$module" ] && [ ! -e "$example" ] || refuse "the copy already holds illustration.rs or illustration_dump.rs"
if grep -q 'mod illustration' "$lib"; then
  refuse "the copy's lib.rs already declares a module illustration"
fi
cp "$here/illustration.rs" "$module"
printf '\n#[cfg(test)]\nmod illustration;\n' >> "$lib"
mkdir -p "$(dirname "$example")"
cp "$here/dump_problem.rs" "$example"

# the pinned toolchain, which rustup takes from rust-toolchain.toml in the directory cargo runs from
cd "$src"
pinned=$(sed -n 's/^channel *= *"\([^"]*\)".*/\1/p' rust-toolchain.toml)
[ -n "$pinned" ] || refuse "rust-toolchain.toml pins no channel"
active=$(rustc --version) || refuse "rustc does not run in $src"
case "$active" in
  "rustc $pinned "*) ;;
  *) refuse "rustc is '$active', not the pinned $pinned" ;;
esac
echo "sources of $repo copied to $src, digests in $logs/sources.sha256, $active"

echo "building the dump, log $logs/build-dump.log"
if ! cargo build --release --locked --manifest-path research/Cargo.toml --target-dir "$targets/research" \
  -p solid-cli --example illustration_dump > "$logs/build-dump.log" 2>&1; then
  tail -n 30 "$logs/build-dump.log" >&2
  refuse "the build of the dump failed, log $logs/build-dump.log"
fi
dump=$targets/research/release/examples/illustration_dump

if [ "$command" = list ]; then
  if ! "$dump" --list "$archive" "$samples" "$iteration" > "$listing"; then
    refuse "the list of $archive failed, and $listing holds the part written"
  fi
  echo "wrote $listing, $(($(wc -l < "$listing") - 1)) anchor only scenarios at $samples samples, iteration $iteration"
  exit 0
fi

mkdir -p "$run"
if [ -e "$run/record.json" ]; then
  rm -f -- "$run/record.json"
fi
echo "writing the input of scenario $scenario, log $run/dump.log"
if ! "$dump" "$archive" "$scenario" "$samples" "$iteration" "$profile" "$run/input.json" 2> "$run/dump.log"; then
  cat "$run/dump.log" >&2
  refuse "the dump failed, log $run/dump.log"
fi
cat "$run/dump.log"

echo "building the engine test, log $logs/build-test.log"
if ! cargo test --release --locked --manifest-path runtime/Cargo.toml --target-dir "$targets/runtime" -p solid-core --lib --no-run \
  > "$logs/build-test.log" 2>&1; then
  tail -n 30 "$logs/build-test.log" >&2
  refuse "the build of the engine test failed, log $logs/build-test.log"
fi

echo "running the engine test, log $run/test.log"
ILLUSTRATION_INPUT=$run/input.json
ILLUSTRATION_OUTPUT=$run/record.json
export ILLUSTRATION_INPUT ILLUSTRATION_OUTPUT
if [ -n "$agent" ]; then
  ILLUSTRATION_AGENT=$agent
  export ILLUSTRATION_AGENT
fi
if ! cargo test --release --locked --manifest-path runtime/Cargo.toml --target-dir "$targets/runtime" -p solid-core --lib \
  illustration::posterior_illustration -- --ignored --exact --nocapture --test-threads=1 > "$run/test.log" 2>&1; then
  tail -n 30 "$run/test.log" >&2
  refuse "the engine test failed, log $run/test.log"
fi
grep -q "test result: ok. 1 passed" "$run/test.log" || refuse "the engine test did not run, log $run/test.log"
grep -E '^(agent|wrote) ' "$run/test.log" || true

# the inputs of the record by SHA-256, the checkout's commit and the identity of its frozen sources. A reader's checkout
# (SOLID_READER=1) need not be a git repository, and its record then names no commit.
python3 - "$run/record.json" "$archive" "$profile" "$repo" "$logs/sources.sha256" << 'PY' || refuse "the provenance of the record could not be written"
import hashlib, json, os, subprocess, sys
record_path, archive, profile, repo, sources = sys.argv[1:6]
def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 23), b""):
            digest.update(chunk)
    return digest.hexdigest()
record = json.load(open(record_path))
record["provenance"] = dict(
    archive=dict(path=archive, sha256=sha(archive)), profile=dict(path=profile, sha256=sha(profile)),
    commit=(lambda run: run.stdout.strip() if run.returncode == 0 else None)(subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=os.environ.get("SOLID_READER") != "1")),
    sources_identity=subprocess.run(["python3", "research/benchmarks/datasets/isolation.py"], cwd=repo, capture_output=True, text=True,
                                    check=True).stdout.strip(),
    copied_sources_sha256=sha(sources))
with open(record_path, "w") as stream:
    json.dump(record, stream)
PY
echo "the record is $run/record.json"
