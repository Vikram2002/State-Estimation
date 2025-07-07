import numpy as np
from quaternion import Quaternion

def transform_vector_euler(vector: np.ndarray, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Transform a vector using Euler angles in Z (yaw), Y (pitch), X (roll) intrinsic order.
    """
    q_yaw = Quaternion.from_axis_angle(np.array([0, 0, 1]), yaw)
    q_pitch = Quaternion.from_axis_angle(np.array([0, 1, 0]), pitch)
    q_roll = Quaternion.from_axis_angle(np.array([1, 0, 0]), roll)

    q_total = q_yaw * q_pitch * q_roll

    return q_total.rotate_vector(vector)


if __name__ == "__main__":
    # Testing
    vector = np.array([1, 0, 0])
    roll = np.radians(0)
    pitch = np.radians(-45)
    yaw = np.radians(45)

    transformed_vector = transform_vector_euler(vector, roll, pitch, yaw)
    print("Transformed Vector:", transformed_vector)

    transformed_vector = transform_vector_euler(np.array([0, 1, 0]), np.radians(-1.0), 0, np.radians(-4.0))
    print("Transformed Magnetic North Vector:", transformed_vector)
