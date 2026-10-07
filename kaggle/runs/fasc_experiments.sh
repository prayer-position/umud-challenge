# Fascicle experiment: train the 512x768 and 768x1152 fascicle configs in
# parallel (one GPU each), score them and the previous fascicle model on the
# same validation split, then build a submission with the best one.
#
# Needs a previous full run attached as a kernel source
# (--kernel-source <owner>/umud-unet-train): its apo_best.pt is reused and its
# fasc_best.pt (configs/fasc_seg.yaml) is the baseline.
set -uo pipefail
OUT=${OUT:-/kaggle/working}
INPUT=${INPUT:-/kaggle/input}
mkdir -p "$OUT/logs" "$OUT/eval" "$OUT/history" "$OUT/viz"
EXPERIMENTS=(fasc_seg_512x768 fasc_seg_768x1152)

PREV=$(dirname "$(find "$INPUT" -name apo_best.pt | head -1)")
echo "previous run outputs: $PREV"
mkdir -p outputs/checkpoints
cp "$PREV/apo_best.pt" outputs/checkpoints/apo_best.pt
cp "$PREV/fasc_best.pt" outputs/checkpoints/fasc_baseline_best.pt

pids=()
for i in "${!EXPERIMENTS[@]}"; do
    name=${EXPERIMENTS[$i]}
    CUDA_VISIBLE_DEVICES=$i python scripts/train_fasc.py --config "configs/$name.yaml" --run-name "$name" \
        > "$OUT/logs/train_$name.log" 2>&1 &
    pids+=($!)
done
CUDA_VISIBLE_DEVICES=0 python scripts/eval_fasc.py --config configs/fasc_seg.yaml \
    --checkpoint outputs/checkpoints/fasc_baseline_best.pt --out "$OUT/eval/fasc_baseline.json" \
    > "$OUT/logs/eval_fasc_baseline.log" 2>&1
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || echo "training ${EXPERIMENTS[$i]} FAILED (see logs/train_${EXPERIMENTS[$i]}.log)"
done
for name in "${EXPERIMENTS[@]}"; do
    echo "== $name"; grep -E "data ready|epoch|early stop|Error" "$OUT/logs/train_$name.log" | tail -60
done
cp outputs/checkpoints/*_history.json "$OUT/history/" 2>/dev/null

# Final scores with flip TTA, on the same validation split as the baseline.
for name in "${EXPERIMENTS[@]}"; do
    [ -f "outputs/checkpoints/${name}_best.pt" ] || continue
    python scripts/eval_fasc.py --config "configs/$name.yaml" --checkpoint "outputs/checkpoints/${name}_best.pt" \
        --out "$OUT/eval/$name.json" > "$OUT/logs/eval_$name.log" 2>&1 || echo "eval $name FAILED"
done
best=$(python - "$OUT/eval" <<'PY'
import json, sys
from pathlib import Path
scores = {p.stem: json.loads(p.read_text()) for p in Path(sys.argv[1]).glob("fasc_*.json")}
for name, m in sorted(scores.items(), key=lambda kv: kv[1]["angle_mae"]):
    print(f"{name:22s} angle_mae={m['angle_mae']:.2f} median={m['angle_median_err']:.2f} "
          f"p90={m['angle_p90_err']:.2f} miss={m['miss_rate']:.3f} dice={m['dice']:.3f}", file=sys.stderr)
print(min(scores, key=lambda k: scores[k]["angle_mae"]))
PY
)
echo "best fascicle model: $best"
if [ "$best" = fasc_baseline ]; then best_config=configs/fasc_seg.yaml; else best_config=configs/$best.yaml; fi
best_ckpt=outputs/checkpoints/${best}_best.pt

set -e
python scripts/predict.py --apo-checkpoint outputs/checkpoints/apo_best.pt --apo-config configs/apo_seg.yaml \
    --fasc-checkpoint "$best_ckpt" --fasc-config "$best_config"
cp "$(ls submissions/submission_*.csv | tail -1)" "$OUT/submission.csv"
cp "$best_ckpt" "$OUT/fasc_best.pt"
cp outputs/checkpoints/apo_best.pt "$OUT/apo_best.pt"
for name in "${EXPERIMENTS[@]}"; do
    [ -f "outputs/checkpoints/${name}_best.pt" ] && cp "outputs/checkpoints/${name}_best.pt" "$OUT/"
done

python scripts/visualize_fascicles.py --source val --checkpoint "$best_ckpt" --config "$best_config" \
    --n 12 --out "$OUT/viz/fasc_val_${best}.png"
python scripts/visualize_fascicles.py --source val --checkpoint outputs/checkpoints/fasc_baseline_best.pt \
    --config configs/fasc_seg.yaml --n 12 --out "$OUT/viz/fasc_val_baseline.png"
python scripts/visualize_fascicles.py --source test --checkpoint "$best_ckpt" --config "$best_config" \
    --n 12 --out "$OUT/viz/fasc_test_${best}.png"
