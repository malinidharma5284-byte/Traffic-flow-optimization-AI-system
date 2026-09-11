"""
extract_frames.py

Stage 3 of the AI Traffic Flow Optimization project: Extract Frames.

Reads a traffic video file and saves frames at a fixed interval
(e.g. 1 frame per second) as JPG images, ready for preprocessing
and vehicle detection in the next stage.

Usage (from terminal, inside your activated venv):
    python extract_frames.py --video data/raw_videos/traffic1.mp4 --output data/frames/traffic1 --fps 1

Requirements:
    pip install opencv-python
"""

import cv2
import os
import argparse


def extract_frames(video_path: str, output_dir: str, frames_per_second: float = 1.0):
    """
    Extract frames from a video at a given rate (frames_per_second)
    and save them as JPG images in output_dir.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # fallback if metadata missing
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # How many original video frames to skip between each saved frame
    frame_interval = max(1, round(video_fps / frames_per_second))

    print(f"Video FPS: {video_fps:.2f}")
    print(f"Total frames in video: {total_frames}")
    print(f"Saving 1 frame every {frame_interval} video frames "
          f"(~{frames_per_second} frame(s) per second)")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        if frame_count % frame_interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done. Saved {saved_count} frames to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from a traffic video.")
    parser.add_argument("--video", required=True, help="Path to input video file (e.g. data/raw_videos/traffic1.mp4)")
    parser.add_argument("--output", required=True, help="Folder to save extracted frames (e.g. data/frames/traffic1)")
    parser.add_argument("--fps", type=float, default=1.0, help="Number of frames to extract per second of video (default: 1.0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_frames(args.video, args.output, args.fps)