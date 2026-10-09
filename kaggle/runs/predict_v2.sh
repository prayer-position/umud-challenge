# Predict the test set (no training) with the members chosen from run 7's
# grouped-split scores, and print diagnostics plus every predicted mask to
# the log (predict.py --dump-log; decode with scripts/decode_log_dump.py).
#
# apo: grouped-recipe + 512x768 v1 + v2 ensemble (best grouped-split
# thickness error: 2.83% mean / 2.06% p90) plus the run-2 model (same recipe
# as the grouped one, random split). fasc: 512x768 grouped (best: 0.88 deg)
# plus its run-3 random-split twin.
#
# Kernel sources: <owner>/umud-unet-train, <owner>/umud-fasc-exp1,
# <owner>/umud-grouped-v1, <owner>/umud-apo-v2, <owner>/umud-ensemble-v1.
set -euo pipefail
OUT=${OUT:-/kaggle/working}
INPUT=${INPUT:-/kaggle/input}
mkdir -p "$OUT/extras"

ck() { find "$INPUT" -path "*$1*" -name "$2" | head -1; }
python scripts/predict.py --dump-log \
    --apo "$(ck umud-unet-train apo_best.pt)" configs/apo_seg.yaml \
    --apo "$(ck umud-apo-v2 apo_seg_grouped_best.pt)" configs/apo_seg_grouped.yaml \
    --apo "$(ck umud-grouped-v1 apo_seg_512x768_best.pt)" configs/apo_seg_512x768.yaml \
    --apo "$(ck umud-apo-v2 apo_seg_512x768_v2_best.pt)" configs/apo_seg_512x768_v2.yaml \
    --fasc "$(ck umud-ensemble-v1 fasc_seg_512x768_grouped_best.pt)" configs/fasc_seg_512x768_grouped.yaml \
    --fasc "$(ck umud-fasc-exp1 fasc_seg_512x768_best.pt)" configs/fasc_seg_512x768.yaml
# Only submission.csv sits at the top of the output, so it's the one CSV
# offered when submitting from the notebook.
cp "$(ls submissions/submission_*.csv | tail -1)" "$OUT/submission.csv"
cp "$(ls submissions/diagnostics_*.csv | tail -1)" "$OUT/extras/diagnostics.csv"
python scripts/make_variants.py --diagnostics "$OUT/extras/diagnostics.csv" --out-dir "$OUT/probes"
