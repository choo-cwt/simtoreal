#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec git --git-dir="${ROOT_DIR}/.gitrepo" --work-tree="${ROOT_DIR}" "$@"
