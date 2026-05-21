#!/usr/bin/env bash
# Orchestrate the 4 Town13 stability triage attempts.
# Stops at first success. Logs each attempt to /tmp/carla_stability_N.log.
set -uo pipefail

CARLA_DIR="$HOME/Sim/CARLA_0.9.15"
VENV_PY="$HOME/Projects/phantom-braking/.venv/bin/python"
TEST_SCRIPT="$HOME/Projects/phantom-braking/scripts/town13_stability_test.py"

results=()

run_attempt () {
  local tag="$1"; shift
  local flags="$1"; shift
  local cam_w="$1"; shift
  local cam_h="$1"; shift
  local qtag="$1"; shift
  local vtag="$1"

  echo
  echo "=================================================================="
  echo "[$tag] ATTEMPT: $flags  cam=${cam_w}x${cam_h}"
  echo "=================================================================="

  # ensure no leftover CARLA processes
  pkill -9 -f CarlaUE4 2>/dev/null || true
  sleep 3

  # start server
  local server_log="/tmp/carla_stability_${tag}.log"
  (cd "$CARLA_DIR" && ./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2000 $flags > "$server_log" 2>&1) &
  local server_pid=$!
  echo "[$tag] server pid=$server_pid, waiting up to 90s for port 2000..."

  # poll
  local up=0
  for i in $(seq 1 90); do
    if timeout 1 bash -c '</dev/tcp/localhost/2000' 2>/dev/null; then
      up=1
      echo "[$tag] port up at t=${i}s"
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "[$tag] server PID died before port came up at t=${i}s"
      break
    fi
    sleep 1
  done

  if [[ $up -ne 1 ]]; then
    echo "[$tag] SERVER NEVER LISTENED. Tail of $server_log:"
    tail -15 "$server_log" 2>/dev/null
    results+=("$tag: SERVER_DID_NOT_START")
    pkill -9 -f CarlaUE4 2>/dev/null || true
    sleep 2
    return 4
  fi

  # give engine 2s after listen to finish init
  sleep 2

  # run test
  PYTHONPATH= env -u PYTHONPATH "$VENV_PY" "$TEST_SCRIPT" \
    --cam-w "$cam_w" --cam-h "$cam_h" \
    --quality-tag "$qtag" --vulkan-tag "$vtag" --tag "$tag"
  local rc=$?
  echo "[$tag] test exit code: $rc"
  results+=("$tag: rc=$rc ($flags)")

  # if server still alive, kill it for cleanup
  if pgrep -f CarlaUE4 > /dev/null 2>&1; then
    pkill -9 -f CarlaUE4 2>/dev/null || true
    sleep 2
  fi

  return $rc
}

# 1. Low quality, default renderer
run_attempt "a1_low" "-quality-level=Low" 1164 874 "Low" "off"
if [[ $? -eq 0 ]]; then
  echo
  echo "FIRST SUCCESS: a1_low"
  printf '%s\n' "${results[@]}"
  exit 0
fi

# 2. Epic quality, Vulkan
run_attempt "a2_vulkan_epic" "-quality-level=Epic -vulkan" 1164 874 "Epic" "on"
if [[ $? -eq 0 ]]; then
  echo
  echo "FIRST SUCCESS: a2_vulkan_epic"
  printf '%s\n' "${results[@]}"
  exit 0
fi

# 3. Low quality, Vulkan
run_attempt "a3_vulkan_low" "-quality-level=Low -vulkan" 1164 874 "Low" "on"
if [[ $? -eq 0 ]]; then
  echo
  echo "FIRST SUCCESS: a3_vulkan_low"
  printf '%s\n' "${results[@]}"
  exit 0
fi

# 4. Low quality, default renderer, 320x240
run_attempt "a4_low_tiny" "-quality-level=Low" 320 240 "Low" "off"
if [[ $? -eq 0 ]]; then
  echo
  echo "FIRST SUCCESS: a4_low_tiny"
  printf '%s\n' "${results[@]}"
  exit 0
fi

echo
echo "ALL ATTEMPTS FAILED:"
printf '%s\n' "${results[@]}"
exit 1
