# Retrain both models on the grouped (leak-free) split, in parallel: the
# aponeurosis model at 512x768 on GPU 0 (selected by thickness error) and the
# fascicle model at 768x1152 on GPU 1 (selected by angle error). Then score
# them, predict the test set with both, and render overlays.
#
# Kernel source: <owner>/umud-unet-train, whose apo_best.pt is scored on the
# same grouped validation split for reference (it was trained on a random
# split, so ~22% of these validation images were in its training set).
set -uo pipefail
OUT=${OUT:-/kaggle/working}
INPUT=${INPUT:-/kaggle/input}
mkdir -p "$OUT/logs" "$OUT/eval" "$OUT/history" "$OUT/viz" outputs/checkpoints
APO=apo_seg_512x768
FASC=fasc_seg_768x1152_grouped

prev_apo=$(find "$INPUT" -name apo_best.pt | head -1)
echo "previous apo model: $prev_apo"

CUDA_VISIBLE_DEVICES=0 python scripts/train_apo.py --config "configs/$APO.yaml" --run-name "$APO" \
    > "$OUT/logs/train_$APO.log" 2>&1 &
apo_pid=$!
CUDA_VISIBLE_DEVICES=1 python scripts/train_fasc.py --config "configs/$FASC.yaml" --run-name "$FASC" \
    > "$OUT/logs/train_$FASC.log" 2>&1 &
fasc_pid=$!
wait $apo_pid || echo "training $APO FAILED (see logs/train_$APO.log)"
wait $fasc_pid || echo "training $FASC FAILED (see logs/train_$FASC.log)"
for name in $APO $FASC; do
    echo "== $name"; grep -E "data ready|epoch|early stop|Error" "$OUT/logs/train_$name.log" | tail -70
done
cp outputs/checkpoints/*_history.json "$OUT/history/" 2>/dev/null

# Final scores with flip TTA on the grouped validation split.
python scripts/eval_model.py --pool apo --config "configs/$APO.yaml" \
    --checkpoint "outputs/checkpoints/${APO}_best.pt" --out "$OUT/eval/$APO.json" > "$OUT/logs/eval_$APO.log" 2>&1 \
    || echo "eval $APO FAILED"
python scripts/eval_model.py --pool apo --config configs/apo_seg.yaml --split grouped \
    --checkpoint "$prev_apo" --out "$OUT/eval/apo_run2.json" > "$OUT/logs/eval_apo_run2.log" 2>&1 \
    || echo "eval apo_run2 FAILED"
python scripts/eval_model.py --pool fasc --config "configs/$FASC.yaml" \
    --checkpoint "outputs/checkpoints/${FASC}_best.pt" --out "$OUT/eval/$FASC.json" > "$OUT/logs/eval_$FASC.log" 2>&1 \
    || echo "eval $FASC FAILED"
python - "$OUT/eval" <<'PY'
import json, sys
from pathlib import Path
for p in sorted(Path(sys.argv[1]).glob("*.json")):
    m = json.loads(p.read_text())
    keys = [k for k in ("dice", "mt_rel_err", "mt_rel_median_err", "mt_rel_p90_err", "deep_angle_err",
                        "angle_mae", "angle_median_err", "angle_p90_err", "miss_rate") if k in m]
    print(f"{p.stem:28s} " + " ".join(f"{k}={m[k]:.4f}" for k in keys))
PY

set -e
for name in $APO $FASC; do cp "outputs/checkpoints/${name}_best.pt" "$OUT/"; done
python scripts/predict.py --apo-checkpoint "outputs/checkpoints/${APO}_best.pt" --apo-config "configs/$APO.yaml" \
    --fasc-checkpoint "outputs/checkpoints/${FASC}_best.pt" --fasc-config "configs/$FASC.yaml"
cp "$(ls submissions/submission_*.csv | tail -1)" "$OUT/submission.csv"
# Only submission.csv sits at the top of the output, so it's the one CSV
# offered when submitting from the notebook; tables go to extras/.
mkdir -p "$OUT/extras"
cp "$(ls submissions/diagnostics_*.csv | tail -1)" "$OUT/extras/diagnostics.csv"

python scripts/visualize_masks.py --pool apo --source val --checkpoint "outputs/checkpoints/${APO}_best.pt" \
    --config "configs/$APO.yaml" --n 12 --out "$OUT/viz/apo_val.png"
python scripts/visualize_masks.py --pool apo --source test --checkpoint "outputs/checkpoints/${APO}_best.pt" \
    --config "configs/$APO.yaml" --n 12 --out "$OUT/viz/apo_test.png"
python scripts/visualize_masks.py --pool fasc --source val --checkpoint "outputs/checkpoints/${FASC}_best.pt" \
    --config "configs/$FASC.yaml" --n 12 --out "$OUT/viz/fasc_val.png"
python scripts/visualize_masks.py --pool fasc --source test --checkpoint "outputs/checkpoints/${FASC}_best.pt" \
    --config "configs/$FASC.yaml" --n 12 --out "$OUT/viz/fasc_test.png"
