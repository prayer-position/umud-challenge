from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

APO_IMAGE_DIR = PROJECT_ROOT / "apo_imgs_v1" / "apo_images_new_model_v1"
APO_MASK_DIR = PROJECT_ROOT / "apo_masks_v1" / "apo_masks_new_model_v1"
FASC_IMAGE_DIR = PROJECT_ROOT / "fasc_imgs_v1" / "fasc_images_new_model_v1"
FASC_MASK_DIR = PROJECT_ROOT / "fasc_masks_v1" / "fasc_masks_new_model_v1"
TEST_IMAGE_DIR = PROJECT_ROOT / "test_images_v2" / "test_set_v2"

SAMPLE_SUBMISSION_PATH = PROJECT_ROOT / "sample_submission.csv"
# The organizers' sample_submission.csv is semicolon-delimited with a UTF-8
# BOM, but Kaggle parses submissions as plain comma-separated CSV: a file in
# the sample's format is rejected with "ID column image_id not found in
# submission" (the whole header reads as one column). Read the sample in its
# own format; write submissions as plain CSV.
SAMPLE_SUBMISSION_SEP = ";"
SAMPLE_SUBMISSION_ENCODING = "utf-8-sig"
SUBMISSION_SEP = ","
SUBMISSION_ENCODING = "utf-8"
SUBMISSION_COLUMNS = ["image_id", "pa_deg", "fl_mm", "mt_mm"]

IGNORED_FILENAMES = {"Thumbs.db"}
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Confirmed by notebooks/01_eda.ipynb: number of foreground classes encoded in
# apo masks (e.g. 1 if superficial+deep aponeurosis share one label, 2 if they
# are distinct label values) and in fasc masks. Update after running the EDA.
APO_NUM_CLASSES = 1
FASC_NUM_CLASSES = 1

# Target common resolution for resizing/padding all images before feeding a
# model. Confirm this is a sane choice against the size distribution found in
# notebooks/01_eda.ipynb.
IMAGE_SIZE = (512, 512)


def list_image_files(directory: Path) -> list[Path]:
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.name not in IGNORED_FILENAMES
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )
