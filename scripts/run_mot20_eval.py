#!/usr/bin/env python3
"""Run TrackEval on MOT20 tracker results produced by veilsight_eval_mot20."""

import argparse
import csv
import sys
from pathlib import Path

# Add bundled TrackEval to path
REPO_ROOT = Path(__file__).resolve().parent.parent
TRACK_EVAL_ROOT = REPO_ROOT / "thirdparty" / "TrackEval"
sys.path.insert(0, str(TRACK_EVAL_ROOT))

import trackeval  # noqa: E402


def _expand_sequence_args(items):
    """Accept both space-separated and comma-separated sequence lists."""
    if not items:
        return []

    out = []
    for item in items:
        for part in item.split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _read_seqmap(seqmap_file):
    """Read a MOT-style seqmap file.

    Expected format:
        name
        MOT20-01
        MOT20-02
        MOT20-03

    Also tolerates CSV rows where the first column is the sequence name.
    """
    seqmap_path = Path(seqmap_file)
    if not seqmap_path.exists():
        raise FileNotFoundError(f"Seqmap file not found: {seqmap_path}")

    sequences = []
    with seqmap_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            first = row[0].strip()
            if not first or first.lower() == "name":
                continue

            sequences.append(first)

    if not sequences:
        raise ValueError(f"Seqmap file contains no sequences: {seqmap_path}")

    return sequences


def _discover_tracker_sequences(tracker_data_dir):
    """Discover sequence names from tracker result txt files."""
    return sorted(p.stem for p in tracker_data_dir.glob("*.txt"))


def _validate_sequences(sequences, gt_folder, tracker_data_dir, split):
    """Validate selected sequences against GT and tracker output files."""
    missing_tracker = []
    missing_gt = []

    for seq in sequences:
        tracker_file = tracker_data_dir / f"{seq}.txt"
        gt_dir = Path(gt_folder) / f"MOT20-{split}" / seq

        # Some MOT20 layouts are assets/MOT20/train/MOT20-01 or assets/MOT20/MOT20-01.
        gt_dir_alt_1 = Path(gt_folder) / split / seq
        gt_dir_alt_2 = Path(gt_folder) / seq

        if not tracker_file.exists():
            missing_tracker.append(str(tracker_file))

        if not (gt_dir.exists() or gt_dir_alt_1.exists() or gt_dir_alt_2.exists()):
            missing_gt.append(seq)

    if missing_tracker:
        joined = "\n  ".join(missing_tracker)
        raise FileNotFoundError(f"Missing tracker result files:\n  {joined}")

    if missing_gt:
        joined = ", ".join(missing_gt)
        raise FileNotFoundError(f"Missing GT sequence folders for: {joined}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MOT20 tracking results")
    parser.add_argument(
        "--tracker_name",
        default="veilsight_tracker",
        help="Name of tracker folder under results/MOT20-{split}/",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Dataset split",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Directory containing tracker outputs",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["HOTA", "CLEAR", "Identity"],
        help="Metrics to compute",
    )
    parser.add_argument(
        "--gt_folder",
        default=str(REPO_ROOT / "assets" / "MOT20"),
        help="Path to MOT20 ground-truth root",
    )

    # New subset controls
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help=(
            "Optional sequence subset. Accepts space-separated or comma-separated values, "
            "e.g. --sequences MOT20-01 MOT20-02 MOT20-03 or "
            "--sequences MOT20-01,MOT20-02,MOT20-03"
        ),
    )
    parser.add_argument(
        "--exclude_sequences",
        nargs="*",
        default=None,
        help=(
            "Optional sequences to exclude from evaluation, e.g. "
            "--exclude_sequences MOT20-05"
        ),
    )
    parser.add_argument(
        "--seqmap_file",
        default=None,
        help=(
            "Optional MOT-style seqmap file. If provided, it defines the sequence subset. "
            "Cannot be used together with --sequences."
        ),
    )
    parser.add_argument(
        "--print_only_combined",
        action="store_true",
        help="Only print combined results.",
    )

    args = parser.parse_args()

    if args.split == "test":
        print("WARNING: MOT20 test split has no public ground truth. Evaluation will fail or produce dummy scores.")
        print("         Use --split train for actual metric computation.\n")

    gt_folder = args.gt_folder
    trackers_folder = args.output_dir
    tracker_name = args.tracker_name

    tracker_data_dir = Path(trackers_folder) / f"MOT20-{args.split}" / tracker_name / "data"
    if not tracker_data_dir.exists():
        print(f"ERROR: Tracker data not found: {tracker_data_dir}")
        sys.exit(1)

    try:
        requested_sequences = _expand_sequence_args(args.sequences)
        excluded_sequences = set(_expand_sequence_args(args.exclude_sequences))

        if args.seqmap_file and requested_sequences:
            print("ERROR: Use either --seqmap_file or --sequences, not both.")
            sys.exit(1)

        if args.seqmap_file:
            selected_sequences = _read_seqmap(args.seqmap_file)
        elif requested_sequences:
            selected_sequences = requested_sequences
        else:
            selected_sequences = _discover_tracker_sequences(tracker_data_dir)

        if excluded_sequences:
            selected_sequences = [s for s in selected_sequences if s not in excluded_sequences]

        # Preserve order but remove duplicates.
        selected_sequences = list(dict.fromkeys(selected_sequences))

        if not selected_sequences:
            print("ERROR: No sequences selected for evaluation.")
            sys.exit(1)

        _validate_sequences(selected_sequences, gt_folder, tracker_data_dir, args.split)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Build configs matching TrackEval defaults
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["USE_PARALLEL"] = False
    eval_config["PRINT_RESULTS"] = True
    eval_config["PRINT_ONLY_COMBINED"] = args.print_only_combined
    eval_config["OUTPUT_SUMMARY"] = True
    eval_config["OUTPUT_DETAILED"] = True
    eval_config["PLOT_CURVES"] = False
    eval_config["DISPLAY_LESS_PROGRESS"] = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = gt_folder
    dataset_config["TRACKERS_FOLDER"] = trackers_folder
    dataset_config["OUTPUT_FOLDER"] = trackers_folder
    dataset_config["TRACKERS_TO_EVAL"] = [tracker_name]
    dataset_config["CLASSES_TO_EVAL"] = ["pedestrian"]
    dataset_config["BENCHMARK"] = "MOT20"
    dataset_config["SPLIT_TO_EVAL"] = args.split
    dataset_config["INPUT_AS_ZIP"] = False
    dataset_config["PRINT_CONFIG"] = True
    dataset_config["DO_PREPROC"] = True
    dataset_config["TRACKER_SUB_FOLDER"] = "data"
    dataset_config["OUTPUT_SUB_FOLDER"] = ""

    # This is the key subset hook.
    # TrackEval accepts SEQ_INFO as {sequence_name: sequence_length}.
    # None lets TrackEval infer sequence length from seqinfo/GT where possible.
    dataset_config["SEQ_INFO"] = {seq: None for seq in selected_sequences}

    metrics_config = {"METRICS": args.metrics, "THRESHOLD": 0.5}

    print("=" * 60)
    print("TrackEval MOT20 Evaluation")
    print("=" * 60)
    print(f"GT folder:      {gt_folder}")
    print(f"Tracker folder: {trackers_folder}")
    print(f"Tracker:        {tracker_name}")
    print(f"Split:          {args.split}")
    print(f"Metrics:        {args.metrics}")
    print(f"Sequences:      {selected_sequences}")
    if excluded_sequences:
        print(f"Excluded:       {sorted(excluded_sequences)}")
    if args.seqmap_file:
        print(f"Seqmap file:    {args.seqmap_file}")
    print("=" * 60 + "\n")

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

    metrics_list = []
    for metric in [
        trackeval.metrics.HOTA,
        trackeval.metrics.CLEAR,
        trackeval.metrics.Identity,
        trackeval.metrics.VACE,
        trackeval.metrics.Count,
    ]:
        if metric.get_name() in metrics_config["METRICS"]:
            metrics_list.append(metric(metrics_config))

    if not metrics_list:
        print("ERROR: No valid metrics selected.")
        sys.exit(1)

    evaluator.evaluate(dataset_list, metrics_list)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()