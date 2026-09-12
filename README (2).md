# AI-Powered Traffic Flow Optimization System

An end-to-end pipeline that uses computer vision and time-series prediction to optimize urban traffic signal timing — detecting vehicles, classifying congestion, forecasting near-future demand, and dynamically allocating green-light duration, validated in simulation against a fixed-timing baseline.

## Overview

Traditional traffic signals run on fixed timers regardless of actual demand, wasting green time on empty approaches while congested ones queue up. This project explores whether AI-driven signal control can do better.

The system combines:
- **Computer vision** (YOLOv8 + OpenCV) to detect and count vehicles from traffic video
- **Time-series prediction** (LSTM / TensorFlow) to forecast near-future vehicle counts
- **Rule-based signal logic** to allocate green time proportionally across intersection approaches
- **SUMO simulation** to validate the approach against a fixed-timing baseline

At prototype scale, the AI-controlled signal timing reduced average waiting time by **~5.2%** and trip duration by **~1.9%** compared to fixed timing, while also improving average speed and time loss — see [Results](#results).

## Objectives

- Reduce average vehicle waiting time at a signalized intersection
- Detect and count vehicles in real time from traffic video using deep learning
- Classify traffic density (low / medium / high) per observation
- Predict near-future traffic density from historical patterns
- Dynamically allocate signal green-time based on real and predicted demand
- Quantitatively evaluate the proposed system against a fixed-timing baseline

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Primary development language |
| OpenCV | Video frame extraction and image preprocessing |
| Ultralytics YOLOv8 | Pretrained deep learning model for vehicle detection |
| TensorFlow / Keras | LSTM model for traffic density prediction |
| SUMO + TraCI | Traffic signal simulation and control |
| Kaggle open datasets | Source traffic video/image data |

## System Pipeline

The system is built as an eight-stage pipeline, moving from raw video to an evaluated, working signal-control system:

| Stage | Name | Purpose |
|---|---|---|
| 1 | Define Objectives | Establish congestion-reduction and signal-timing goals |
| 2 | Collect Video Data | Gather traffic footage from Kaggle datasets and stock video sources |
| 3 | Extract Frames | Sample video into individual frames using OpenCV |
| 4 | Preprocess Images | Resize and normalize frames for model input |
| 5 | Detect Vehicles & Classify Density | Run YOLOv8 to count vehicles and label density (low/medium/high) per frame |
| 6 | Train Prediction Model | Train an LSTM on merged multi-video time-series data to forecast near-future vehicle counts |
| 7 | Signal Timing Logic | Rule-based allocation of green-light duration proportional to real-time demand |
| 8 | Simulate & Evaluate | Compare AI-controlled vs. fixed-timing signals in SUMO using standard traffic metrics |

## Methodology

**Data Collection & Frame Extraction** — Traffic videos sourced from Kaggle traffic-monitoring datasets and supplementary stock footage, processed with OpenCV at 1 frame/second to produce a sequential image dataset.

**Vehicle Detection & Density Classification** — A pretrained YOLOv8 (yolov8n) model detects and counts vehicles (cars, trucks, buses, motorcycles) per frame. Frames are classified as low (0–5), medium (6–15), or high (16+) density. Output: annotated frames + a per-frame CSV (frame, vehicle count, density).

**Multi-Video Data Merging** — Density summaries from multiple videos are merged into a single time-ordered dataset, tagged by source video and time step, without mixing sequences across videos.

**Prediction Model Training** — An LSTM network predicts the next vehicle count from a sliding window of past readings (window size = 3), using two dense layers after the LSTM, the Adam optimizer, and MSE loss. Given the limited dataset (three source videos), this stage mainly validates the end-to-end pipeline; it reached a test MAE of ~1.64 vehicles.

**Signal Timing Logic** — A rule-based function allocates a fixed total green-light cycle (e.g. 90s) proportionally across approaches by relative vehicle count, bounded between 10s–60s per phase to avoid starving or over-prioritizing any direction. Kept transparent and interpretable rather than model-driven.

**Simulation & Evaluation** — A simplified four-way intersection was modeled in SUMO (netconvert-generated nodes/edges, traffic-light-controlled junction). Two runs over identical traffic demand were compared:
- **Baseline**: SUMO's default fixed-timing traffic light program
- **AI-controlled**: signal phases updated every 30 simulated seconds via TraCI, based on live per-approach vehicle counts

Both evaluated via SUMO's duration-log statistics: average waiting time, trip duration, speed, and time loss.

## Results

| Metric | Baseline (Fixed Timing) | AI-Controlled (Proposed) |
|---|---|---|
| Average Waiting Time (s) | 11.65 | **11.05** |
| Average Trip Duration (s) | 47.67 | **46.77** |
| Average Speed (m/s) | 9.24 | **9.52** |
| Average Time Loss (s) | 18.15 | **17.32** |

The AI-controlled system improved on **all four metrics** consistently, indicating the proportional allocation logic effectively redistributed green time toward the more congested approach without degrading performance elsewhere.

## Limitations

- The prediction model was trained on a small dataset (three short videos, ~60 total frames), limiting forecasting reliability — it's a proof of concept, not a deployment-ready predictor.
- The SUMO network is a simplified single four-way intersection, not a real road network, and doesn't capture multi-intersection coordination effects.
- A small number of simulated vehicle teleports (SUMO's handling of gridlocked vehicles) remained even after tuning demand density — a known artifact of simplified auto-generated networks.
- Vehicle detection accuracy was not separately benchmarked against ground-truth annotations in this iteration.

## Future Work

- Collect longer and more numerous traffic videos to improve LSTM prediction accuracy
- Extend the SUMO network to multiple coordinated intersections
- Replace the rule-based signal logic with a reinforcement-learning controller informed by the LSTM predictions
- Benchmark YOLO detection accuracy against labeled datasets (precision/recall on a held-out annotated set)
- Test the pipeline on real-time video streams rather than pre-recorded footage

## Conclusion

This project implements and evaluates an end-to-end AI-powered traffic flow optimization pipeline — from raw traffic video through vehicle detection, density classification, predictive modeling, and simulated signal control. The AI-controlled signal timing showed a measurable, consistent improvement over fixed-timing signals across all evaluated metrics in simulation, supporting the feasibility of AI-driven approaches to urban traffic management at prototype scale.
