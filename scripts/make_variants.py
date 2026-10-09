"""Write alternative submissions from a predict.py diagnostics CSV, for
leaderboard checks of one measurement choice at a time.

    python scripts/make_variants.py --diagnostics extras/diagnostics.csv --out-dir probes

Variants: "inner" (MT/FL/PA between the aponeuroses' inner edges instead of
band centres), "largest" (the original two-largest-components pairing),
"fl_x0.85" and "fl_x1.15" (fascicle length scaled). Each gets the same
median fill, range clipping and Kaggle format check as submission.csv.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import TEST_RANGES, check_submission_file  # noqa: E402

from umud.data import schema  # noqa: E402

VARIANTS = {
    "inner": {"pa_deg": "pa_inner", "fl_mm": "fl_inner", "mt_mm": "mt_inner"},
    "largest": {"pa_deg": "pa_largest", "fl_mm": "fl_largest", "mt_mm": "mt_largest"},
    "fl_x0.85": {"fl_mm": ("fl_mm", 0.85)},
    "fl_x1.15": {"fl_mm": ("fl_mm", 1.15)},
}


def build(diag: pd.DataFrame, spec: dict) -> pd.DataFrame:
    sub = diag[schema.SUBMISSION_COLUMNS].copy()
    for column, source in spec.items():
        sub[column] = diag[source[0]] * source[1] if isinstance(source, tuple) else diag[source]
    for column in schema.SUBMISSION_COLUMNS[1:]:
        sub[column] = sub[column].fillna(sub[column].median()).clip(*TEST_RANGES[column])
    return sub


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    diag = pd.read_csv(args.diagnostics)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in VARIANTS.items():
        sub = build(diag, spec)
        path = out_dir / f"submission_{name}.csv"
        sub.to_csv(path, sep=schema.SUBMISSION_SEP, index=False, encoding=schema.SUBMISSION_ENCODING)
        check_submission_file(path, diag["image_id"].tolist())
        print(f"{path}: medians " + ", ".join(f"{c}={sub[c].median():.2f}" for c in schema.SUBMISSION_COLUMNS[1:]))


if __name__ == "__main__":
    main()
