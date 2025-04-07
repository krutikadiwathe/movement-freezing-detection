import numpy as np

def detect_freezing(trajectory_data, velocity_threshold=0.01, duration_threshold=5):
    """
    Detects freezing behavior based on low velocity for a sustained duration.

    Parameters:
    - trajectory_data (np.ndarray): shape (n_frames, n_keypoints, 2)
      e.g., (1000, 5, 2) for 1000 frames, 5 keypoints with x/y coordinates.
    - velocity_threshold (float): max average velocity to be considered freezing.
    - duration_threshold (int): minimum number of consecutive frames below threshold.

    Returns:
    - list of tuples: [(start_frame, end_frame), ...] for each freezing event
    """
    if trajectory_data.ndim != 3 or trajectory_data.shape[2] != 2:
        raise ValueError("trajectory_data must have shape (n_frames, n_keypoints, 2)")

    # Compute average x,y position per frame
    centroids = np.mean(trajectory_data, axis=1)  # shape: (n_frames, 2)

    # Compute velocity between frames
    deltas = np.diff(centroids, axis=0)  # shape: (n_frames - 1, 2)
    velocities = np.linalg.norm(deltas, axis=1)  # shape: (n_frames - 1,)

    # Identify low-motion frames
    low_motion = velocities < velocity_threshold

    # Find continuous freezing segments
    freezing_events = []
    start = None
    for i, is_freezing in enumerate(low_motion):
        if is_freezing:
            if start is None:
                start = i
        else:
            if start is not None and (i - start) >= duration_threshold:
                freezing_events.append((start, i))
            start = None
    if start is not None and (len(low_motion) - start) >= duration_threshold:
        freezing_events.append((start, len(low_motion)))

    return freezing_events
