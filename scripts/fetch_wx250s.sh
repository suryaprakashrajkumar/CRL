#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/third_party/mujoco_menagerie"
REPO_URL="https://github.com/google-deepmind/mujoco_menagerie.git"

if [[ ! -d "${TARGET_DIR}/.git" ]]; then
  echo "Cloning MuJoCo Menagerie into ${TARGET_DIR}"
  git clone --depth 1 "${REPO_URL}" "${TARGET_DIR}"
else
  echo "Updating existing MuJoCo Menagerie checkout"
  git -C "${TARGET_DIR}" pull --ff-only
fi

MODEL_PATH="${TARGET_DIR}/trossen_wx250s/wx250s.xml"
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Expected model not found at ${MODEL_PATH}"
  exit 1
fi

echo "WidowX model ready: ${MODEL_PATH}"
