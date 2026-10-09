# Aponeurosis retrain, second attempt (run 5's apo model underperformed).
# GPU 0: the run-2 recipe on the grouped split (honest baseline); GPU 1:
# 512x768 with a completed cosine schedule. All apo models are then scored
# on the grouped validation split, and the test set is predicted with the
# best apo model (lowest thickness error among those trained on the grouped
# split) plus run 5's grouped fascicle model.
#
# Kernel sources: <owner>/umud-unet-train (run-2 apo, reference only: ~22% of
# the grouped validation images were in its training set) and
# <owner>/umud-grouped-v1 (run-5 apo and fascicle models).
set -uo pipefail
OUT=${OUT:-/kaggle/working}
INPUT=${INPUT:-/kaggle/input}
mkdir -p "$OUT/logs" "$OUT/eval" "$OUT/history" "$OUT/viz" outputs/checkpoints
NEW=(apo_seg_grouped apo_seg_512x768_v2)
FASC=fasc_seg_768x1152_grouped

cp "$(find "$INPUT" -path "*umud-unet-train*" -name apo_best.pt | head -1)" outputs/checkpoints/apo_run2_best.pt
cp "$(find "$INPUT" -name apo_seg_512x768_best.pt | head -1)" outputs/checkpoints/apo_seg_512x768_best.pt
cp "$(find "$INPUT" -name "${FASC}_best.pt" | head -1)" "outputs/checkpoints/${FASC}_best.pt"

pids=()
for i in "${!NEW[@]}"; do
    CUDA_VISIBLE_DEVICES=$i python scripts/train_apo.py --config "configs/${NEW[$i]}.yaml" --run-name "${NEW[$i]}" \
        > "$OUT/logs/train_${NEW[$i]}.log" 2>&1 &
    pids+=($!)
done
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || echo "training ${NEW[$i]} FAILED (see logs/train_${NEW[$i]}.log)"
done
for name in "${NEW[@]}"; do
    echo "== $name"; grep -E "data ready|epoch|early stop|Error" "$OUT/logs/train_$name.log" | tail -45
done
cp outputs/checkpoints/*_history.json "$OUT/history/" 2>/dev/null

# name:config pairs; every model is scored on the grouped validation split.
models=(apo_run2:configs/apo_seg.yaml apo_seg_512x768:configs/apo_seg_512x768.yaml
        apo_seg_grouped:configs/apo_seg_grouped.yaml apo_seg_512x768_v2:configs/apo_seg_512x768_v2.yaml)
for entry in "${models[@]}"; do
    name=${entry%%:*}; config=${entry#*:}
    [ -f "outputs/checkpoints/${name}_best.pt" ] || continue
    python scripts/eval_model.py --pool apo --config "$config" --split grouped \
        --checkpoint "outputs/checkpoints/${name}_best.pt" --out "$OUT/eval/$name.json" \
        > "$OUT/logs/eval_$name.log" 2>&1 || echo "eval $name FAILED"
done
best=$(python - "$OUT/eval" <<'PY'
import json, sys
from pathlib import Path
scores = {p.stem: json.loads(p.read_text()) for p in Path(sys.argv[1]).glob("apo_*.json")}
for name, m in sorted(scores.items(), key=lambda kv: kv[1]["mt_rel_err"]):
    print(f"{name:22s} dice={m['dice']:.4f} mt_rel_err={m['mt_rel_err']:.4f} median={m['mt_rel_median_err']:.4f} "
          f"p90={m['mt_rel_p90_err']:.4f} deep_angle_err={m['deep_angle_err']:.3f} miss={m['miss_rate']:.3f}",
          file=sys.stderr)
# run 2 trained on part of this validation set, so it is not eligible.
eligible = {k: v for k, v in scores.items() if k != "apo_run2"}
print(min(eligible, key=lambda k: eligible[k]["mt_rel_err"]))
PY
)
echo "best apo model (grouped-split trained): $best"
case $best in
    apo_seg_512x768) best_config=configs/apo_seg_512x768.yaml ;;
    *) best_config=configs/$best.yaml ;;
esac

set -e
for name in "${NEW[@]}"; do cp "outputs/checkpoints/${name}_best.pt" "$OUT/"; done
python scripts/predict.py --apo "outputs/checkpoints/${best}_best.pt" "$best_config" \
    --fasc "outputs/checkpoints/${FASC}_best.pt" "configs/$FASC.yaml"
cp "$(ls submissions/submission_*.csv | tail -1)" "$OUT/submission.csv"
# Only submission.csv sits at the top of the output, so it's the one CSV
# offered when submitting from the notebook; tables go to extras/.
mkdir -p "$OUT/extras"
cp "$(ls submissions/diagnostics_*.csv | tail -1)" "$OUT/extras/diagnostics.csv"
python scripts/visualize_masks.py --pool apo --source val --checkpoint "outputs/checkpoints/${best}_best.pt" \
    --config "$best_config" --n 12 --out "$OUT/viz/apo_val_${best}.png"
python scripts/visualize_masks.py --pool apo --source test --checkpoint "outputs/checkpoints/${best}_best.pt" \
    --config "$best_config" --n 12 --out "$OUT/viz/apo_test_${best}.png"
