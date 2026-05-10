#!/usr/bin/env python3
"""Run TrackEval on MOT20 tracker results produced by veilsight_eval_mot20."""

import argparse
import os
import sys
from pathlib import Path

# Add bundled TrackEval to path
REPO_ROOT = Path(__file__).resolve().parent.parent
TRACK_EVAL_ROOT = REPO_ROOT / "thirdparty" / "TrackEval"
sys.path.insert(0, str(TRACK_EVAL_ROOT))

import trackeval  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Evaluate MOT20 tracking results")
    parser.add_argument("--tracker_name", default="veilsight_tracker", help="Name of tracker folder under results/")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--output_dir", default="results", help="Directory containing tracker outputs")
    parser.add_argument("--metrics", nargs="+", default=["HOTA", "CLEAR", "Identity"],
                        help="Metrics to compute")
    parser.add_argument("--gt_folder", default=str(REPO_ROOT / "assets" / "MOT20"),
                        help="Path to MOT20 ground-truth root")
    args = parser.parse_args()

    if args.split == "test":
        print("WARNING: MOT20 test split has no public ground truth. Evaluation will fail or produce dummy scores.")
        print("         Use --split train for actual metric computation.\n")

    gt_folder = args.gt_folder
    trackers_folder = args.output_dir
    tracker_name = args.tracker_name

    # Verify tracker results exist
    tracker_data_dir = Path(trackers_folder) / f"MOT20-{args.split}" / tracker_name / "data"
    if not tracker_data_dir.exists():
        print(f"ERROR: Tracker data not found: {tracker_data_dir}")
        sys.exit(1)

    # Build configs matching TrackEval defaults
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["USE_PARALLEL"] = False
    eval_config["PRINT_RESULTS"] = True
    eval_config["PRINT_ONLY_COMBINED"] = False
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

    metrics_config = {"METRICS": args.metrics, "THRESHOLD": 0.5}

    print("=" * 60)
    print("TrackEval MOT20 Evaluation")
    print("=" * 60)
    print(f"GT folder:      {gt_folder}")
    print(f"Tracker folder: {trackers_folder}")
    print(f"Tracker:        {tracker_name}")
    print(f"Split:          {args.split}")
    print(f"Metrics:        {args.metrics}")
    print("=" * 60 + "\n")

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR,
                   trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config["METRICS"]:
            metrics_list.append(metric(metrics_config))

    if not metrics_list:
        print("ERROR: No valid metrics selected.")
        sys.exit(1)

    evaluator.evaluate(dataset_list, metrics_list)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
