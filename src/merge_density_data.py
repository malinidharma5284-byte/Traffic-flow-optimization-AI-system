"""
merge_density_data.py

Prep step before Stage 6 (Train the Prediction Model).

Combines multiple density_summary.csv files (one per video, produced by
detect_vehicles.py) into a single merged CSV with:
    - a source_video column (which video the row came from)
    - a time_step column (ordered position within that video's sequence)
    - the original frame, vehicle_count, and density columns

This merged file is what the LSTM training script will read.

Usage (from terminal, inside your activated venv):
    python src/merge_density_data.py --inputs data/detections/traffic1/density_summary.csv data/detections/traffic2/density_summary.csv data/detections/traffic3/density_summary.csv --output data/merged_density_data.csv

Or, simpler: point it at the parent "detections" folder and it will
auto-discover every density_summary.csv inside subfolders:
    python src/merge_density_data.py --detections_dir data/detections --output data/merged_density_data.csv
"""

import os
import csv
import argparse


def find_csv_files(detections_dir: str):
    """Auto-discover every density_summary.csv under a detections directory."""
    found = []
    for root, _, files in os.walk(detections_dir):
        for f in files:
            if f == "density_summary.csv":
                found.append(os.path.join(root, f))
    return sorted(found)


def merge_csvs(csv_paths, output_path: str):
    if not csv_paths:
        raise ValueError("No CSV files provided or found to merge.")

    all_rows = []

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"Warning: skipping missing file {csv_path}")
            continue

        # Use the parent folder name as the video identifier, e.g. "traffic1"
        video_name = os.path.basename(os.path.dirname(csv_path))

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for time_step, row in enumerate(reader):
                all_rows.append({
                    "source_video": video_name,
                    "time_step": time_step,
                    "frame": row["frame"],
                    "vehicle_count": row["vehicle_count"],
                    "density": row["density"],
                })

        print(f"Loaded {csv_path} -> {video_name} ({time_step + 1} rows)")

    if not all_rows:
        raise ValueError("No rows found across the given CSV files.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["source_video", "time_step", "frame", "vehicle_count", "density"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. Merged {len(csv_paths)} file(s), {len(all_rows)} total rows.")
    print(f"Saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge multiple density_summary.csv files into one dataset.")
    parser.add_argument("--inputs", nargs="+", help="Explicit list of density_summary.csv paths to merge")
    parser.add_argument("--detections_dir", help="Parent folder to auto-discover density_summary.csv files (alternative to --inputs)")
    parser.add_argument("--output", required=True, help="Path to save the merged CSV, e.g. data/merged_density_data.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.inputs:
        csv_paths = args.inputs
    elif args.detections_dir:
        csv_paths = find_csv_files(args.detections_dir)
        print(f"Auto-discovered {len(csv_paths)} file(s): {csv_paths}")
    else:
        raise ValueError("Provide either --inputs (list of files) or --detections_dir (auto-discover).")

    merge_csvs(csv_paths, args.output)