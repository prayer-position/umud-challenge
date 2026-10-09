# Thickness outliers and ensembles.
#
# GPU 1 trains a second fascicle model on the grouped split (512x768), so a
# fascicle ensemble can be scored honestly. Meanwhile GPU 0 scores the
# grouped-trained apo models and their ensembles with eval_ensemble.py
# (both pairing rules, both edge conventions, worst images listed). Then the
# fascicle singles and ensemble are scored, and the test set is predicted
# with ensembles chosen by those scores, plus variant submissions for
# leaderboard checks (probes/).
#
# Kernel sources: <owner>/umud-unet-train, <owner>/umud-fasc-exp1,
# <owner>/umud-grouped-v1, <owner>/umud-apo-v2.
set -uo pipefail
OUT=${OUT:-/kaggle/working}
INPUT=${INPUT:-/kaggle/input}
mkdir -p "$OUT/logs" "$OUT/eval" "$OUT/extras" outputs/checkpoints

ck() { find "$INPUT" -path "*$1*" -name "$2" | head -1; }
RUN2_APO=$(ck umud-unet-train apo_best.pt)
APO_G=$(ck umud-apo-v2 apo_seg_grouped_best.pt)
APO_V1=$(ck umud-grouped-v1 apo_seg_512x768_best.pt)
APO_V2=$(ck umud-apo-v2 apo_seg_512x768_v2_best.pt)
FASC_768G=$(ck umud-grouped-v1 fasc_seg_768x1152_grouped_best.pt)
FASC_768=$(ck umud-fasc-exp1 fasc_seg_768x1152_best.pt)
FASC_512=$(ck umud-fasc-exp1 fasc_seg_512x768_best.pt)
FASC_512G=outputs/checkpoints/fasc_seg_512x768_grouped_best.pt
for v in RUN2_APO APO_G APO_V1 APO_V2 FASC_768G FASC_768 FASC_512; do echo "$v=${!v}"; done

CUDA_VISIBLE_DEVICES=1 python scripts/train_fasc.py --config configs/fasc_seg_512x768_grouped.yaml \
    --run-name fasc_seg_512x768_grouped > "$OUT/logs/train_fasc_seg_512x768_grouped.log" 2>&1 &
train_pid=$!

ev() {  # ev <pool> <split config> <name> <eval_ensemble args...>
    local pool=$1 split=$2 name=$3; shift 3
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_ensemble.py --pool "$pool" --split-config "$split" --name "$name" \
        --out "$OUT/eval/$name.json" "$@" > "$OUT/logs/eval_$name.log" 2>&1 || echo "eval $name FAILED"
}
APO_SPLIT=configs/apo_seg_grouped.yaml
ev apo "$APO_SPLIT" apo_grouped --member "$APO_G" configs/apo_seg_grouped.yaml --details 20
ev apo "$APO_SPLIT" apo_v1 --member "$APO_V1" configs/apo_seg_512x768.yaml
ev apo "$APO_SPLIT" apo_v2 --member "$APO_V2" configs/apo_seg_512x768_v2.yaml
ev apo "$APO_SPLIT" apo_ens_g_v1 --member "$APO_G" configs/apo_seg_grouped.yaml \
    --member "$APO_V1" configs/apo_seg_512x768.yaml --details 20
ev apo "$APO_SPLIT" apo_ens_g_v1_v2 --member "$APO_G" configs/apo_seg_grouped.yaml \
    --member "$APO_V1" configs/apo_seg_512x768.yaml --member "$APO_V2" configs/apo_seg_512x768_v2.yaml
ev apo "$APO_SPLIT" apo_run2_leaky --member "$RUN2_APO" configs/apo_seg.yaml
for name in apo_grouped apo_ens_g_v1; do echo "== $name"; sed -n '/^worst/,$p' "$OUT/logs/eval_$name.log"; done

wait $train_pid || echo "training fasc_seg_512x768_grouped FAILED"
grep -E "data ready|epoch|early stop|Error" "$OUT/logs/train_fasc_seg_512x768_grouped.log" | tail -55

FASC_SPLIT=configs/fasc_seg_768x1152_grouped.yaml
ev fasc "$FASC_SPLIT" fasc_768g --member "$FASC_768G" configs/fasc_seg_768x1152_grouped.yaml --details 10
ev fasc "$FASC_SPLIT" fasc_512g --member "$FASC_512G" configs/fasc_seg_512x768_grouped.yaml
ev fasc "$FASC_SPLIT" fasc_ens_g --member "$FASC_768G" configs/fasc_seg_768x1152_grouped.yaml \
    --member "$FASC_512G" configs/fasc_seg_512x768_grouped.yaml --details 10
echo "== fasc_ens_g"; sed -n '/^worst/,$p' "$OUT/logs/eval_fasc_ens_g.log"

# Summary, and ensemble choices: an ensemble is used when it beats its best
# single member on the honest (grouped) split. The final ensembles also add
# the same recipes trained on the random split (run 2 apo, run 3 fascicles):
# they can't be scored on this split, but leakage is irrelevant on the test
# set and they add diversity.
choice=$(python - "$OUT/eval" <<'PY'
import json, sys
from pathlib import Path
m = {p.stem: json.loads(p.read_text()) for p in Path(sys.argv[1]).glob("*.json")}
for name, r in sorted(m.items()):
    if name.startswith("apo"):
        parts = [f"{k[11:]}={v['mean']:.4f}/{v['median']:.4f}/{v['p90']:.4f}" for k, v in r.items() if k.startswith("mt_rel_err_")]
        parts.append(f"deep_angle={r['deep_angle_err']['mean']:.3f}")
    else:
        a = r["angle_err"]
        parts = [f"angle={a['mean']:.3f}/{a['median']:.3f}/{a['p90']:.3f}"]
    print(f"{name:18s} " + "  ".join(parts), file=sys.stderr)
mt = lambda n: m[n]["mt_rel_err_topmost_center"]["mean"] if n in m else float("inf")
ang = lambda n: m[n]["angle_err"]["mean"] if n in m else float("inf")
apo = "ens" if mt("apo_ens_g_v1") < min(mt("apo_grouped"), mt("apo_v1")) else "single"
fasc = "ens" if ang("fasc_ens_g") < min(ang("fasc_768g"), ang("fasc_512g")) else "single"
print(apo, fasc)
PY
)
read apo_choice fasc_choice <<< "$choice"
echo "apo ensemble: $apo_choice, fasc ensemble: $fasc_choice"

APO_ARGS=(--apo "$RUN2_APO" configs/apo_seg.yaml --apo "$APO_G" configs/apo_seg_grouped.yaml)
[ "$apo_choice" = ens ] && APO_ARGS+=(--apo "$APO_V1" configs/apo_seg_512x768.yaml)
FASC_ARGS=(--fasc "$FASC_768G" configs/fasc_seg_768x1152_grouped.yaml --fasc "$FASC_768" configs/fasc_seg_768x1152.yaml)
[ "$fasc_choice" = ens ] && FASC_ARGS+=(--fasc "$FASC_512G" configs/fasc_seg_512x768_grouped.yaml \
    --fasc "$FASC_512" configs/fasc_seg_512x768.yaml)

set -e
cp "$FASC_512G" "$OUT/"
python scripts/predict.py "${APO_ARGS[@]}" "${FASC_ARGS[@]}"
# Only submission.csv sits at the top of the output, so it's the one CSV
# offered when submitting from the notebook; the rest goes to extras/ and
# probes/.
cp "$(ls submissions/submission_*.csv | tail -1)" "$OUT/submission.csv"
cp "$(ls submissions/diagnostics_*.csv | tail -1)" "$OUT/extras/diagnostics.csv"
python scripts/make_variants.py --diagnostics "$OUT/extras/diagnostics.csv" --out-dir "$OUT/probes"
