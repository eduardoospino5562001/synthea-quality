#!/usr/bin/env bash
# check.sh - fast local gate for lint, types and affected tests.
#
# Why this script exists: ruff catches style and import problems, mypy with a
# baseline stops new type errors from creeping in, and running only the tests
# of the touched area keeps feedback quick while work is in progress. The full
# suite still runs at the end of a task; this script is the shortcut used
# while editing and by the pre-commit hook.
#
# Usage:
#   scripts/check.sh          changed files versus main (commits plus staged
#                             and unstaged work)
#   scripts/check.sh --staged  only files staged in the index (used by the hook)
#   scripts/check.sh --all     whole repository (lint and types over
#                             everything, full fast test suite)
set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"

RUFF="$REPO/.venv/bin/ruff"
MYPY="$REPO/.venv/bin/mypy"
PYTEST="$REPO/.venv/bin/pytest"
BASELINE="$REPO/scripts/mypy-baseline.txt"

MODE="changed"
for arg in "$@"; do
  case "$arg" in
    --staged) MODE="staged" ;;
    --all) MODE="all" ;;
    -h|--help)
      echo "usage: scripts/check.sh [--staged] [--all]"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      echo "usage: scripts/check.sh [--staged] [--all]" >&2
      exit 2
      ;;
  esac
done

if [ "$MODE" = "staged" ]; then
  CHANGED="$(git diff --cached --name-only --diff-filter=ACMR || true)"
elif [ "$MODE" = "all" ]; then
  CHANGED=""
else
  COMMITS="$(git diff --name-only 'main...HEAD' --diff-filter=ACMR || true)"
  UNSTAGED="$(git diff --name-only --diff-filter=ACMR || true)"
  STAGED="$(git diff --cached --name-only --diff-filter=ACMR || true)"
  CHANGED="$(printf '%s\n%s\n%s\n' "$COMMITS" "$UNSTAGED" "$STAGED" | grep -v '^$' | sort -u || true)"
fi

failures=0

echo "== check.sh ($MODE): ruff =="
if "$RUFF" check .; then
  echo "ruff: clean"
else
  echo "ruff: FAILED"
  failures=$((failures + 1))
fi

echo "== check.sh ($MODE): mypy against baseline =="
RAW="$(mktemp)"
ACTUAL="$(mktemp)"
trap 'rm -f "$RAW" "$ACTUAL"' EXIT
mypy_status=0
"$MYPY" > "$RAW" 2>&1 || mypy_status=$?
if [ "$mypy_status" -gt 1 ]; then
  echo "mypy: FAILED (mypy could not run, exit $mypy_status)"
  cat "$RAW"
  failures=$((failures + 1))
elif [ ! -f "$BASELINE" ]; then
  echo "mypy: FAILED (missing baseline $BASELINE)" >&2
  failures=$((failures + 1))
else
  { grep ": error:" "$RAW" || true; } | sed -E 's/^([^:]+):[0-9]+: error:/\1: error:/' | LC_ALL=C sort > "$ACTUAL"
  NEW_ERRORS="$(mktemp)"
  trap 'rm -f "$RAW" "$ACTUAL" "$NEW_ERRORS"' EXIT
  LC_ALL=C comm -13 "$BASELINE" "$ACTUAL" > "$NEW_ERRORS" || true
  if [ -s "$NEW_ERRORS" ]; then
    echo "mypy: FAILED (new errors not in baseline):"
    cat "$NEW_ERRORS"
    failures=$((failures + 1))
  else
    echo "mypy: clean (no new errors versus baseline)"
  fi
  SHRUNK="$(mktemp)"
  trap 'rm -f "$RAW" "$ACTUAL" "$NEW_ERRORS" "$SHRUNK"' EXIT
  LC_ALL=C comm -23 "$BASELINE" "$ACTUAL" > "$SHRUNK" || true
  if [ -s "$SHRUNK" ]; then
    count="$(wc -l < "$SHRUNK" | tr -d ' ')"
    echo "mypy: note: baseline can shrink ($count line(s) no longer reported)"
  fi
fi

echo "== check.sh ($MODE): tests =="
if [ "$MODE" = "all" ]; then
  echo "running full fast suite"
  if "$PYTEST" -q -m "not integration"; then
    echo "tests: clean (full suite)"
  else
    echo "tests: FAILED (full suite)"
    failures=$((failures + 1))
  fi
else
  PY_CHANGED="$(printf '%s\n' "$CHANGED" | grep -E '\.py$' || true)"
  if printf '%s\n' "$CHANGED" | grep -qx "pyproject.toml"; then
    echo "pyproject.toml changed: running full fast suite"
    if "$PYTEST" -q -m "not integration"; then
      echo "tests: clean (full suite)"
    else
      echo "tests: FAILED (full suite)"
      failures=$((failures + 1))
    fi
  elif [ -z "$PY_CHANGED" ]; then
    echo "tests: skipped (no Python files changed)"
  else
    TESTS=""
    NEED_FULL=0
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        case "$f" in
          tests/*.py)
            TESTS="$(printf '%s\n%s\n' "$TESTS" "$f")"
            ;;
          synthea_quality/*)
            rest="${f#synthea_quality/}"
            case "$rest" in
              */*)
                pkg="${rest%%/*}"
                matches=$(ls tests/test_"$pkg"_*.py 2>/dev/null || true)
                if [ -z "$matches" ]; then
                  NEED_FULL=1
                else
                  TESTS="$(printf '%s\n%s\n%s' "$TESTS" "$matches")"
                fi
                ;;
              *.py)
                mod="${rest%.py}"
                matches=$(ls tests/test_"$mod"*.py 2>/dev/null || true)
                if [ -z "$matches" ]; then
                  NEED_FULL=1
                else
                  TESTS="$(printf '%s\n%s\n%s' "$TESTS" "$matches")"
                fi
                ;;
              *)
                NEED_FULL=1
                ;;
            esac
            ;;
          *)
            # Non-package Python file (for example scripts/): no mapping.
            NEED_FULL=1
            ;;
        esac
      done <<< "$PY_CHANGED"
      TESTS="$(printf '%s\n' "$TESTS" | grep -v '^$' | sort -u || true)"
      if [ "$NEED_FULL" -eq 1 ] || [ -z "$TESTS" ]; then
        echo "no mapped tests for some changed file: running full fast suite"
        if "$PYTEST" -q -m "not integration"; then
          echo "tests: clean (full suite)"
        else
          echo "tests: FAILED (full suite)"
          failures=$((failures + 1))
        fi
      else
        echo "running tests for the touched area:"
        printf '%s\n' "$TESTS"
        # shellcheck disable=SC2086
        if "$PYTEST" -q -m "not integration" $TESTS; then
          echo "tests: clean (touched area)"
        else
          echo "tests: FAILED (touched area)"
          failures=$((failures + 1))
        fi
      fi
  fi
fi

echo "== check.sh ($MODE): summary =="
if [ "$failures" -eq 0 ]; then
  echo "check.sh: all steps passed"
else
  echo "check.sh: $failures step(s) failed" >&2
  exit 1
fi
