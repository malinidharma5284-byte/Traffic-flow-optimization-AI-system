"""
detect_vehicles.py

Stage 5 of the AI Traffic Flow Optimization project: Detect Vehicles & Classify Density.

Reads all frames extracted in Stage 3, runs a pretrained YOLOv8 model
on each one to detect vehicles (car, truck, bus, motorcycle), draws
bounding boxes, and classifies each frame's traffic density (low/medium/high)
based on vehicle count. Saves annotated frames and a CSV summary.

Usage (from terminal, inside your activated venv):
    python src/detect_vehicles.py --frames data/frames/traffic1 --output data/detections/traffic1

Requirements:
    pip install ultralytics opencv-python
"""

import os
import csv
import argparse
import cv2
from ultralytics import YOLO

# COCO class IDs for vehicle types (from the standard YOLO pretrained model)
VEHICLE_CLASS_IDS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Tune these thresholds based on your camera's field of view and typical traffic
DENSITY_THRESHOLDS = {
    "low": (0, 5),      # 0-5 vehicles
    "medium": (6, 15),  # 6-15 vehicles
    "high": (16, float("inf")),  # 16+ vehicles
}


def classify_density(vehicle_count: int) -> str:
    for label, (low, high) in DENSITY_THRESHOLDS.items():
        if low <= vehicle_count <= high:
            return label
    return "unknown"


def detect_vehicles(frames_dir: str, output_dir: str, model_name: str = "yolov8n.pt", conf_threshold: float = 0.4):
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Frames folder not found: {frames_dir}")

    os.makedirs(output_dir, exist_ok=True)
    annotated_dir = os.path.join(output_dir, "annotated_frames")
    os.makedirs(annotated_dir, exist_ok=True)

    print(f"Loading YOLO model: {model_name} (downloads automatically on first run)")
    model = YOLO(model_name)

    frame_files = sorted(
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not frame_files:
        raise ValueError(f"No image frames found in {frames_dir}")

    print(f"Found {len(frame_files)} frames to process.")

    results_summary = []

    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)

        results = model(frame, conf=conf_threshold, verbose=False)[0]

        vehicle_count = 0
        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id in VEHICLE_CLASS_IDS:
                vehicle_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = VEHICLE_CLASS_IDS[class_id]
                confidence = float(box.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"{label} {confidence:.2f}", (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

        density = classify_density(vehicle_count)

        # Overlay the summary on the frame itself
        cv2.putText(
            frame, f"Vehicles: {vehicle_count} | Density: {density}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )

        out_path = os.path.join(annotated_dir, frame_file)
        cv2.imwrite(out_path, frame)

        results_summary.append({
            "frame": frame_file,
            "vehicle_count": vehicle_count,
            "density": density,
        })

        print(f"{frame_file}: {vehicle_count} vehicles -> {density}")

    # Save summary as CSV for the next stage (prediction model)
    csv_path = os.path.join(output_dir, "density_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "vehicle_count", "density"])
        writer.writeheader()
        writer.writerows(results_summary)

    print(f"\nDone. Annotated frames saved to: {annotated_dir}")
    print(f"Density summary saved to: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Detect vehicles and classify traffic density per frame.")
    parser.add_argument("--frames", required=True, help="Folder containing extracted frames (from Stage 3)")
    parser.add_argument("--output", required=True, help="Folder to save annotated frames and CSV summary")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model to use (default: yolov8n.pt, the smallest/fastest)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold for detections (default: 0.4)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detect_vehicles(args.frames, args.output, args.model, args.conf)