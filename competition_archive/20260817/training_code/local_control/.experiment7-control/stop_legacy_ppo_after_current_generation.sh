#!/usr/bin/env bash
set -Eeuo pipefail

parent_pid="${1:?controller PID required}"
generation_root="${2:?current generation root required}"
receipt="${3:?receipt path required}"

if [[ -e "$receipt" ]]; then
  echo "receipt already exists: $receipt" >&2
  exit 2
fi
if ! kill -0 "$parent_pid" 2>/dev/null; then
  echo "controller is not running: $parent_pid" >&2
  exit 3
fi
parent_command=$(tr '\0' ' ' <"/proc/$parent_pid/cmdline")
if [[ "$parent_command" != *run_universal_ppo_role_13c2fe3.sh* ]]; then
  echo "unexpected controller command: $parent_command" >&2
  exit 4
fi

mkdir -p "$(dirname "$receipt")"
kill -STOP "$parent_pid"
collector_pid=""
for candidate in /proc/[0-9]*; do
  [[ -r "$candidate/stat" && -r "$candidate/cmdline" ]] || continue
  ppid=$(awk '{print $4}' "$candidate/stat" 2>/dev/null || true)
  command=$(tr '\0' ' ' <"$candidate/cmdline" 2>/dev/null || true)
  if [[ "$ppid" == "$parent_pid" && "$command" == *collect_universal_ppo_rollouts.py* ]]; then
    collector_pid="${candidate##*/}"
    break
  fi
done
if [[ -z "$collector_pid" ]]; then
  echo "no current collector child for controller $parent_pid" >&2
  kill -CONT "$parent_pid"
  exit 5
fi

while kill -0 "$collector_pid" 2>/dev/null; do
  sleep 1
done
kill -CONT "$parent_pid"

trainer_pid=""
for _ in $(seq 1 1200); do
  for candidate in /proc/[0-9]*; do
    [[ -r "$candidate/stat" && -r "$candidate/cmdline" ]] || continue
    ppid=$(awk '{print $4}' "$candidate/stat" 2>/dev/null || true)
    command=$(tr '\0' ' ' <"$candidate/cmdline" 2>/dev/null || true)
    if [[ "$ppid" == "$parent_pid" && "$command" == *train_universal_ppo.py* ]]; then
      trainer_pid="${candidate##*/}"
      break 2
    fi
  done
  sleep 0.05
done
if [[ -z "$trainer_pid" ]]; then
  echo "controller did not start the expected trainer: $parent_pid" >&2
  kill -STOP "$parent_pid"
  exit 6
fi
kill -STOP "$parent_pid"
while kill -0 "$trainer_pid" 2>/dev/null; do
  sleep 1
done

if [[ ! -s "$generation_root/checkpoint.pt" || ! -s "$generation_root/metrics.json" ]]; then
  echo "generation did not complete cleanly: $generation_root" >&2
  exit 7
fi
{
  echo "stoppedAtUtc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "controllerPid=$parent_pid"
  echo "collectorPid=$collector_pid"
  echo "trainerPid=$trainer_pid"
  echo "generationRoot=$generation_root"
  sha256sum "$generation_root/checkpoint.pt" "$generation_root/metrics.json"
} >"$receipt"
kill -TERM "$parent_pid"
kill -CONT "$parent_pid"
echo "LEGACY_CONTROLLER_STOPPED controller=$parent_pid generation=$generation_root receipt=$receipt"
