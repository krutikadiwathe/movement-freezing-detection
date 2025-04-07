import numpy as np

def detect_freezing(trajectory_data, velocity_threshold=0.01, duration_threshold=5):
    """
    Detects freezing behavior based on low velocity for a sustained duration.

    Parameters:
    - trajectory_data (np.ndarray): shape (n_frames, n_keypoints, 2)
    - velocity_threshold (float): max velocity to be considered freezing
    - duration_threshold (int): min number of consecutive frames considered as freezing

    Returns:
    - list of tuples: [(start_frame, end_frame), ...] for each freezing event
    """
    if trajectory_data.ndim != 3 or trajectory_data.shape[2] != 2:
        raise ValueError("trajectory_data must have shape (n_frames, n_keypoints, 2)")

    centroids = trajectory_data.mean(axis=1)
    velocities = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    low_motion = velocities < velocity_threshold

    # Final output: list of (start, end) where the condition is met for duration_threshold
    freezing_events = []
    start = None
    for i in range(len(low_motion)):
        if low_motion[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= duration_threshold:
                    freezing_events.append((start, i))
                start = None
    # Edge case: if still freezing at the end
    if start is not None and len(low_motion) - start >= duration_threshold:
        freezing_events.append((start, len(low_motion)))

    return freezing_events
