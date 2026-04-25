#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator for dataprivacy-env
#
# Checks that your HF Space is live, Docker image builds, and openenv validate passes.
#
# Prerequisites:
#   - Docker:       https://docs.docker.com/get-docker/
#   - openenv-core: pip install openenv-core  (or: uv sync)
#   - curl (usually pre-installed)
#
# Run from this directory:
#   bash validate-submission.sh <ping_url>
#
# Arguments:
#   ping_url   Your HuggingFace Space URL (e.g. https://Maddy140605-dataprivacy-env.hf.space)
#              Defaults to the env var PING_URL if set.
#
# Example:
#   bash validate-submission.sh https://Maddy140605-dataprivacy-env.hf.space
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600

# Color support only when running in a real terminal
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

# ── helpers ────────────────────────────────────────────────────────────────────

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

# ── args ───────────────────────────────────────────────────────────────────────

# Accept ping_url as arg or env var
PING_URL="${1:-${PING_URL:-}}"

if [ -z "$PING_URL" ]; then
  printf "Usage: bash %s <ping_url>\n" "$0"
  printf "\n"
  printf "  ping_url   Your HuggingFace Space URL\n"
  printf "             e.g. https://Maddy140605-dataprivacy-env.hf.space\n"
  printf "\n"
  printf "  Or set the PING_URL environment variable before running.\n"
  exit 1
fi

# Resolve repo dir to the directory this script lives in
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

PING_URL="${PING_URL%/}"
export PING_URL
PASS=0

# ── banner ─────────────────────────────────────────────────────────────────────

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}  Project: dataprivacy-env${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

# ── Step 1: HF Space liveness ──────────────────────────────────────────────────

log "${BOLD}Step 1/3: Pinging HF Space${NC} ($PING_URL/reset) ..."

CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not reachable (connection failed or timed out)"
  hint "Check that your Space is running at $PING_URL"
  hint "Try: curl -s -o /dev/null -w '%%{http_code}' -X POST $PING_URL/reset"
  stop_at "Step 1"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  hint "Make sure your Space is running and the URL is correct."
  hint "Try opening $PING_URL in your browser first."
  stop_at "Step 1"
fi

# ── Step 2: Docker build ───────────────────────────────────────────────────────

log "${BOLD}Step 2/3: Running docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

# Prefer server/Dockerfile (OpenEnv convention), fall back to root Dockerfile
if [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
  DOCKERFILE="$REPO_DIR/server/Dockerfile"
elif [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
  DOCKERFILE="$REPO_DIR/Dockerfile"
else
  fail "No Dockerfile found in $REPO_DIR or $REPO_DIR/server/"
  stop_at "Step 2"
fi

log "  Dockerfile: $DOCKERFILE"
log "  Context:    $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" \
  docker build -f "$DOCKERFILE" "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

# ── Step 3: openenv validate ───────────────────────────────────────────────────

log "${BOLD}Step 3/3: Running openenv validate${NC} ..."

# Try venv python first (project convention), then system openenv
OPENENV_CMD=""
if [ -f "$REPO_DIR/.venv/Scripts/openenv" ]; then
  OPENENV_CMD="$REPO_DIR/.venv/Scripts/openenv"
elif [ -f "$REPO_DIR/.venv/bin/openenv" ]; then
  OPENENV_CMD="$REPO_DIR/.venv/bin/openenv"
elif command -v openenv &>/dev/null; then
  OPENENV_CMD="openenv"
else
  fail "openenv command not found"
  hint "Install it: pip install openenv-core  (or: uv sync then use .venv)"
  stop_at "Step 3"
fi

log "  Using: $OPENENV_CMD"

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && "$OPENENV_CMD" validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

# ── Summary ────────────────────────────────────────────────────────────────────

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0
