import numpy as np
from movement.behavior_detection import detect_freezing

def test_detect_freezing_basic():
    # Simulate 20 frames, 3 keypoints, all stationary
    data = np.zeros((20, 3, 2))

    # Add motion in the middle frames
    data[10:15] += 5.0

    # Expect freezing from 0–10 and 15–19 (ignores motion frames)
    events = detect_freezing(data, velocity_threshold=0.5, duration_threshold=3)
    assert events == [(0, 10), (15, 19)]
