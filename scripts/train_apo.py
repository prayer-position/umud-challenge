import argparse

from umud.train_pipeline import run_training
from umud.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the aponeurosis segmentation model.")
    parser.add_argument("--config", default="configs/apo_seg.yaml")
    parser.add_argument("--subset", type=int, default=None, help="Use only the first N samples (smoke test).")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = run_training("apo", config, subset=args.subset, epochs=args.epochs)
    print(f"Best checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
