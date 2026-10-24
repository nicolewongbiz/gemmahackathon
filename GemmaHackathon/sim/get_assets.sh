#!/usr/bin/env bash
# Fetch the Kinova Gen3 and Robotiq 2F-85 models from MuJoCo Menagerie into sim/menagerie (pinned commit).
set -euo pipefail
cd "$(dirname "$0")"
REV=f3b6013629c7b94f8ee93fd977b161fa8350f9bc
if [ ! -d menagerie ]; then
  git clone -q --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git menagerie
fi
cd menagerie
git sparse-checkout set kinova_gen3 robotiq_2f85
git -c advice.detachedHead=false checkout -q "$REV"
echo "ok: $(pwd)"
