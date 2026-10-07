"""One entry point for the three original per-recording BrainAE models."""
import argparse
import importlib.util
import sys
from pathlib import Path
from brainae.runner import run


def main():
    parser = argparse.ArgumentParser(
        description="BrainAE: train, evaluate, or check prepared inputs.",
        usage="python run.py {eeg,meg,mua} {train,evaluate,check} [options]",
        epilog="Example: python run.py eeg check --data-dir data/prepared/eeg/sub-01\n"
               "For command options: python run.py eeg train --help",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("modality", choices=["eeg", "meg", "mua"])
    parser.add_argument("command", choices=["train", "evaluate", "check"])
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
        parser.parse_args()
        return
    args = parser.parse_args(sys.argv[1:3])
    path = Path(__file__).parent / f"BAE_v0.2_{args.modality}.py"
    spec = importlib.util.spec_from_file_location("brainae_model_" + args.modality, path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    try:
        run(args.modality, model.Enc_eeg, model.Dec_eeg, model.weights_init_normal,
            argv=sys.argv[3:], mode=args.command)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"Input error: {exc}\n")


if __name__ == "__main__":
    main()
