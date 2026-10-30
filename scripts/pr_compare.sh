#!/usr/bin/env bash
# Generate matching populations and compare their CSV exports.
set -euo pipefail

usage_error() {
  echo "$*" >&2
  echo "usage: scripts/pr_compare.sh BASE_BUILD_DIR PR_BUILD_DIR MODULE_FILE SEED POPULATION OUT_DIR" >&2
  exit 1
}

[ "$#" -eq 6 ] || usage_error "Expected six arguments"
BASE_BUILD="$1"
PR_BUILD="$2"
MODULE="$3"
SEED="$4"
POPULATION="$5"
OUT_DIR="$6"

for build in "$BASE_BUILD" "$PR_BUILD"; do
  [ -d "$build" ] && [ -f "$build/run_synthea" ] && [ -x "$build/run_synthea" ] \
    || usage_error "Build must contain an executable run_synthea: $build"
done
[ -f "$MODULE" ] || usage_error "Module file does not exist: $MODULE"
[[ "$SEED" =~ ^[0-9]+$ && "$SEED" =~ [1-9] ]] \
  || usage_error "SEED must be a positive integer"
[[ "$POPULATION" =~ ^[0-9]+$ && "$POPULATION" =~ [1-9] ]] \
  || usage_error "POPULATION must be a positive integer"
[ -n "$OUT_DIR" ] || usage_error "OUT_DIR must not be empty"

shopt -s nullglob dotglob
if [ -e "$OUT_DIR" ] || [ -L "$OUT_DIR" ]; then
  [ -d "$OUT_DIR" ] || usage_error "OUT_DIR must be a directory"
  contents=("$OUT_DIR"/*)
  [ "${#contents[@]}" -eq 0 ] || usage_error "OUT_DIR must be empty"
else
  mkdir -- "$OUT_DIR" \
    || usage_error "Cannot create OUT_DIR: $OUT_DIR (its parent must exist)"
fi
OUT_DIR="$(cd "$OUT_DIR" && pwd -P)"
MODULE="$(cd "$(dirname "$MODULE")" && pwd -P)/$(basename "$MODULE")"

metadata=()
for side in base pr; do
  if [ "$side" = base ]; then
    build="$BASE_BUILD"
  else
    build="$PR_BUILD"
  fi
  if ! (cd "$build" && ./run_synthea -s "$SEED" -cs "$SEED" -p "$POPULATION" \
      --exporter.csv.export=true "--exporter.baseDirectory=$OUT_DIR/$side/"); then
    echo "Synthea failed on $side side" >&2
    exit 2
  fi
  json_files=("$OUT_DIR/$side/metadata/"*.json)
  if [ "${#json_files[@]}" -ne 1 ] || [ ! -f "${json_files[0]}" ]; then
    echo "Expected exactly one metadata JSON file on $side side" >&2
    exit 2
  fi
  metadata+=("${json_files[0]}")
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if command -v synthea-compare >/dev/null 2>&1; then
  COMPARE="synthea-compare"
else
  COMPARE="$REPO/.venv/bin/synthea-compare"
fi
if ! "$COMPARE" "$OUT_DIR/base/csv" "$OUT_DIR/pr/csv" --module "$MODULE" \
    --metadata-baseline "${metadata[0]}" --metadata-candidate "${metadata[1]}" \
    --output-dir "$OUT_DIR"; then
  echo "synthea-compare failed" >&2
  exit 3
fi
