"""
run_ai_simulation.py

Final piece of Stage 8: Simulate & Evaluate.

Runs the SUMO simulation step by step using TraCI (Traffic Control
Interface), reading real-time vehicle counts approaching the intersection
and using signal_timing.py's allocate_green_time() logic to decide how
long each direction's green light should last -- instead of SUMO's
default fixed timing.

At the end, prints summary stats (average waiting time, average duration)
so you can directly compare against your baseline run.

Usage (from terminal, inside your activated venv):
    python src/run_ai_simulation.py --net intersection.net.xml --route routes.rou.xml

Requirements:
    pip install traci
    (traci is included with the SUMO installation / eclipse-sumo package)
"""

import os
import sys
import argparse
import traci

# Import our Stage 7 logic
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from signal_timing import allocate_green_time

TLS_ID = "center"  # the traffic light junction ID from intersection.net.xml
# Incoming lanes at the intersection -- adjust these if your network uses
# different lane IDs (check intersection.net.xml's <edge> entries if unsure)
APPROACH_LANES = ["N_in_0", "S_in_0", "E_in_0", "W_in_0"]

MIN_PHASE_DURATION = 10  # seconds, matches signal_timing.py's MIN_GREEN_SECONDS
DECISION_INTERVAL = 30   # re-evaluate signal timing every N simulated seconds


def get_vehicle_counts(lanes):
    """Ask SUMO (via TraCI) how many vehicles are currently waiting on each lane."""
    return [traci.lane.getLastStepVehicleNumber(lane) for lane in lanes]


def run_simulation(net_file: str, route_file: str, total_steps: int = 300):
    sumo_cmd = [
        "sumo",  # use "sumo-gui" instead if you want to watch it visually
        "-n", net_file,
        "-r", route_file,
        "--duration-log.statistics",
        "--tripinfo-output", "tripinfo_ai.xml",
        "--no-warnings",
    ]

    traci.start(sumo_cmd)
    print("Simulation started with AI-controlled signal timing...\n")

    step = 0
    try:
        while step < total_steps and traci.simulation.getMinExpectedNumber() > 0:
            if step % DECISION_INTERVAL == 0:
                counts = get_vehicle_counts(APPROACH_LANES)
                green_times = allocate_green_time(counts, total_cycle_seconds=90)

                # Map our 4-direction allocation onto SUMO's phase durations.
                # Our network's tlLogic has 4 phases: NS-green, NS-yellow, EW-green, EW-yellow.
                # We adjust the two green phases (index 0 and 2) based on demand.
                ns_demand = counts[0] + counts[1]  # North + South
                ew_demand = counts[2] + counts[3]  # East + West
                ns_green, ew_green = allocate_green_time([ns_demand, ew_demand], total_cycle_seconds=60)

                program = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]
                phases = program.phases
                phases[0].duration = max(MIN_PHASE_DURATION, ns_green)  # NS green phase
                phases[2].duration = max(MIN_PHASE_DURATION, ew_green)  # EW green phase
                traci.trafficlight.setProgramLogic(TLS_ID, program)

                print(f"[t={step}s] counts={counts} -> NS green={ns_green}s, EW green={ew_green}s")

            traci.simulationStep()
            step += 1
    finally:
        traci.close()

    print("\nSimulation finished. Results saved to tripinfo_ai.xml")
    print("Compare this against your baseline tripinfo file using SUMO's built-in stats,")
    print("or reuse the same --duration-log.statistics summary that printed above the run.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SUMO simulation with AI-controlled signal timing via TraCI.")
    parser.add_argument("--net", required=True, help="Path to the .net.xml network file")
    parser.add_argument("--route", required=True, help="Path to the .rou.xml route file")
    parser.add_argument("--steps", type=int, default=300, help="Max simulation steps/seconds to run (default: 300)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulation(args.net, args.route, args.steps)