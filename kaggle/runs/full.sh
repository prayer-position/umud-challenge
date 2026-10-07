# Train both models with the default configs and write a submission.
# Usage: bash kaggle/runs/full.sh [extra train args, e.g. --epochs 1]
set -euo pipefail
OUT=/kaggle/working
python scripts/train_apo.py --config configs/apo_seg.yaml "$@"
python scripts/train_fasc.py --config configs/fasc_seg.yaml "$@"
python scripts/predict.py
cp outputs/checkpoints/*.pt "$OUT"/
cp "$(ls submissions/submission_*.csv | tail -1)" "$OUT"/submission.csv
