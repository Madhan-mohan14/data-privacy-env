#!/bin/bash
# Wrapper to run openenv push without Windows charmap encoding errors.
# Usage: bash push_to_hf.sh [--repo-id <repo>]
REPO_ID="${1:-Maddy140605/dataprivacy-env}"

.venv/Scripts/python << PYEOF
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.argv = ['openenv', 'push', '--repo-id', '${REPO_ID}']
from openenv.cli.__main__ import main
main()
PYEOF
