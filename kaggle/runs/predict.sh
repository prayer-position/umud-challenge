# Predict the test set with checkpoints from earlier runs (no training) and
# write the submission, per-image diagnostics, calibration table and
# fascicle overlays. Attach the runs holding the checkpoints as kernel
# sources (--kernel-source <owner>/<slug>).
# Usage: bash kaggle/runs/predict.sh [apo_ckpt] [fasc_ckpt] [fasc_config]
set -euo pipefail
OUT=${OUT:-/kaggle/working}
INPUT=${INPUT:-/kaggle/input}
APO_CKPT=${1:-apo_best.pt}
FASC_CKPT=${2:-fasc_seg_768x1152_best.pt}
FASC_CONFIG=${3:-configs/fasc_seg_768x1152.yaml}
mkdir -p "$OUT/viz"

apo=$(find "$INPUT" -name "$APO_CKPT" | head -1)
fasc=$(find "$INPUT" -name "$FASC_CKPT" | head -1)
echo "apo: $apo"
echo "fasc: $fasc ($FASC_CONFIG)"

python scripts/calibrate_images.py --out "$OUT/calibration.csv"
python scripts/predict.py --apo-checkpoint "$apo" --apo-config configs/apo_seg.yaml \
    --fasc-checkpoint "$fasc" --fasc-config "$FASC_CONFIG"
cp "$(ls submissions/submission_*.csv | tail -1)" "$OUT/submission.csv"
cp "$(ls submissions/diagnostics_*.csv | tail -1)" "$OUT/diagnostics.csv"
python scripts/visualize_fascicles.py --source test --checkpoint "$fasc" --config "$FASC_CONFIG" \
    --n 12 --out "$OUT/viz/fasc_test.png"
