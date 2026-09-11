"""
signal_timing.py

Stage 7 of the AI Traffic Flow Optimization project: Signal Timing Logic.

Takes vehicle counts for multiple approaches/directions at an intersection
(e.g. North, South, East, West) and decides how many seconds of green light
each direction should get, based on how congested it is relative to the
others. This is deliberately simple, transparent rule-based logic -- it
doesn't need training data, and it's easy to explain in a project report.

Two ways to use it:
1. Directly with numbers you provide (e.g. from your density_summary.csv,
   or predicted values from the Stage 6 model).
2. As a reusable function (allocate_green_time) you can import into a
   simulation script later (Stage 8).

Usage (from terminal, inside your activated venv):
    python src/signal_timing.py --counts 12 4 9 2 --labels North South East West --total_cycle 90
"""

import argparse


# Safety bounds so no direction ever gets 0 seconds (starves traffic)
# or an unreasonably long green light.
MIN_GREEN_SECONDS = 10
MAX_GREEN_SECONDS = 60


def allocate_green_time(vehicle_counts: list, total_cycle_seconds: int = 90) -> list:
    """
    Given vehicle counts per direction, split total_cycle_seconds of green
    time proportionally to how congested each direction is.

    Returns a list of green-light durations (seconds), same order as input.
    """
    if not vehicle_counts:
        raise ValueError("vehicle_counts cannot be empty")

    total_vehicles = sum(vehicle_counts)

    if total_vehicles == 0:
        # No traffic anywhere -- split time evenly
        equal_share = total_cycle_seconds / len(vehicle_counts)
        return [round(equal_share, 1) for _ in vehicle_counts]

    raw_allocations = []
    for count in vehicle_counts:
        share = (count / total_vehicles) * total_cycle_seconds
        # Clamp to safety bounds
        share = max(MIN_GREEN_SECONDS, min(MAX_GREEN_SECONDS, share))
        raw_allocations.append(share)

    return [round(a, 1) for a in raw_allocations]


def classify_density(vehicle_count: int) -> str:
    """Same thresholds as detect_vehicles.py, kept here so this script
    can run standalone without importing that module."""
    if vehicle_count <= 5:
        return "low"
    elif vehicle_count <= 15:
        return "medium"
    else:
        return "high"


def print_signal_plan(labels: list, vehicle_counts: list, green_times: list):
    print("\nSignal Timing Plan")
    print("-" * 50)
    for label, count, green_time in zip(labels, vehicle_counts, green_times):
        density = classify_density(count)
        print(f"{label:10s} | vehicles: {count:3d} | density: {density:6s} | green light: {green_time:5.1f}s")
    print("-" * 50)
    print(f"Total cycle time: {sum(green_times):.1f}s\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Allocate traffic signal green time based on vehicle counts.")
    parser.add_argument("--counts", nargs="+", type=int, required=True, help="Vehicle counts per direction, e.g. --counts 12 4 9 2")
    parser.add_argument("--labels", nargs="+", default=None, help="Direction labels, e.g. --labels North South East West")
    parser.add_argument("--total_cycle", type=int, default=90, help="Total cycle time in seconds to split across directions (default: 90)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    labels = args.labels if args.labels else [f"Direction {i+1}" for i in range(len(args.counts))]

    if len(labels) != len(args.counts):
        raise ValueError("Number of --labels must match number of --counts")

    green_times = allocate_green_time(args.counts, args.total_cycle)
    print_signal_plan(labels, args.counts, green_times)