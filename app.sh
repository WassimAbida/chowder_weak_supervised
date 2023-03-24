#! /bin/bash
set -o errexit -o pipefail -o nounset
source ./venv/bin/activate
python -m chowder_weak_supervised.evaluation.metrics --n_extreme=2 --num_workers=0
