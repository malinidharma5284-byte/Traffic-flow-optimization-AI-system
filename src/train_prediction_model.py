"""
train_prediction_model.py

Stage 6 of the AI Traffic Flow Optimization project: Train the Prediction Model.

Reads the merged time-series dataset (from merge_density_data.py) and trains
a simple LSTM model to predict the NEXT vehicle_count/density based on a
short window of past readings, per video. This lets the system anticipate
congestion a step ahead instead of only reacting to it.

Usage (from terminal, inside your activated venv):
    python src/train_prediction_model.py --data data/merged_density_data.csv --output models/density_predictor.h5 --window 3

Requirements:
    pip install tensorflow pandas numpy scikit-learn
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def build_sequences(df: pd.DataFrame, window: int):
    """
    Build (X, y) training pairs from vehicle_count sequences, grouped per
    source_video so we never mix frames from different videos into one
    sequence. X = window of past counts, y = the next count.
    """
    X, y = [], []

    for video_name, group in df.groupby("source_video"):
        group = group.sort_values("time_step")
        counts = group["vehicle_count"].astype(float).values

        if len(counts) <= window:
            print(f"Skipping '{video_name}': only {len(counts)} rows, needs > {window}")
            continue

        for i in range(len(counts) - window):
            X.append(counts[i:i + window])
            y.append(counts[i + window])

    return np.array(X), np.array(y)


def train_model(data_path: str, output_path: str, window: int = 3, epochs: int = 50):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Merged dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")
    print(f"Videos found: {df['source_video'].unique().tolist()}")

    X, y = build_sequences(df, window)

    if len(X) < 5:
        raise ValueError(
            f"Only {len(X)} training samples available — too few to train on. "
            f"Collect more video data, or reduce --window (currently {window})."
        )

    print(f"Built {len(X)} training sequences (window size = {window})")

    # Reshape for LSTM: (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Small dataset -> keep a reasonable test split, but don't starve training data
    test_size = 0.2 if len(X) >= 15 else 0.0
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None
        print("Dataset too small for a held-out test split — training on all data.")

    model = Sequential([
        LSTM(32, activation="relu", input_shape=(window, 1)),
        Dense(16, activation="relu"),
        Dense(1),  # predicts next vehicle_count (a number)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    print("\nTraining model...")
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=4,
        validation_data=(X_test, y_test) if X_test is not None else None,
        verbose=1,
    )

    if X_test is not None:
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest MAE (avg vehicles off by): {mae:.2f}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.save(output_path)
    print(f"\nModel saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an LSTM to predict future traffic density.")
    parser.add_argument("--data", required=True, help="Path to merged_density_data.csv")
    parser.add_argument("--output", required=True, help="Path to save the trained model, e.g. models/density_predictor.h5")
    parser.add_argument("--window", type=int, default=3, help="How many past readings to use per prediction (default: 3)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args.data, args.output, args.window, args.epochs)