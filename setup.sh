#!/usr/bin/env bash
# Kintrace setup for Mac / Linux: bash setup.sh
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pytest
python -m kintrace incident --outdir incidents_check
echo "Kintrace is set up. Next time, run: source .venv/bin/activate"
