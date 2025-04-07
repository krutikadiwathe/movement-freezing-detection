import numpy as np
from movement.behavior_detection import detect_freezing

def test_detect_freezing_basic():
    # Create 20 frames, 3 keypoints, stationary by default
    data = np.zeros((20, 3, 2))

    # Simulate real movement between frames 10–14
    for i in range(10, 15):
        data[i] += (i - 9) * 1.0  # Gradually increase positions

    # Expect freezing before 10 and after 14, if duration >= 3
    events = detect_freezing(data, velocity_threshold=0.5, duration_threshold=3)
    assert events == [(0, 9), (15, 19)]
