# GSoC 2025 - Heuristic freezing behavior detection module for NIU movement project

This repository contains a heuristic-based freezing behavior detection module, contributed as part of Google Summer of Code 2025 for the [NIU movement project].

## 📌 Overview

This module detects freezing behavior in pose estimation data by analyzing velocity between frame-to-frame movements. Freezing is defined as a period of low movement sustained over time. This tool can assist researchers in behavioral annotation pipelines by identifying periods of inactivity in animals.

## 🧠 Features

- Detects freezing behavior using velocity and duration thresholds
- Easy integration into pose analysis workflows
- Unit-tested with synthetic data
- Simple, flexible API

## 📁 Files

- `movement/behavior_detection.py` – Freezing detection logic
- `tests/test_behavior_detection.py` – Unit test with motion simulation

## 💡 Example

python
import numpy as np
from movement.behavior_detection import detect_freezing

# 20 frames, 3 keypoints, mostly stationary
data = np.zeros((20, 3, 2))

# Simulate movement in frames 10–14
for i in range(10, 15):
    data[i] += (i - 9) * 1.0

events = detect_freezing(data, velocity_threshold=0.5, duration_threshold=3)
print(events)  # Output: [(0, 9), (15, 19)]


🧪 Running Tests
Run tests using pytest:
python -m pytest tests/
