import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from quaternion import Quaternion
from attitude_simulator import AttitudeSimulator, AngularAccelerationCommand
from attitude_estimator_ekf import AttitudeEstimatorQ

def quaternion_to_euler_zyx(q_array):
    """
    Converts quaternion [w, x, y, z] to intrinsic ZYX Euler angles.
    """
    w, x, y, z = q_array
    r = R.from_quat([x, y, z, w])  # scipy expects [x, y, z, w]
    return r.as_euler('ZYX', degrees=True)  # [yaw, pitch, roll]


def test_estimator():
    # --- Simulator setup ---
    mag_dec_angle_deg = 4.0
    mag_inc_angle_deg = 60.0
    accel_error = 0.1
    mag_error = 1e-6
    gyro_error = 0.1
    dt = 0.01

    simulator = AttitudeSimulator(
        mag_dec_angle_deg,
        mag_inc_angle_deg,
        accel_error,
        mag_error,
        gyro_error
    )

    initial_attitude = Quaternion.from_axis_angle(np.array([1, 5, 9]), np.pi / 2)
    initial_ang_vel = np.array([-5.0, 0.0, 3.0])
    commands = [AngularAccelerationCommand(np.array([np.pi, 0, 0]), 1),
                AngularAccelerationCommand(np.array([0, 0, 0]), 5),
                AngularAccelerationCommand(np.array([-np.pi, 0, 0]), 1),
                AngularAccelerationCommand(np.array([0, -np.pi, 0]), 1),
                AngularAccelerationCommand(np.array([0, 0, -np.pi]), 2),
                AngularAccelerationCommand(np.array([0, 0, 0]), 5),
                AngularAccelerationCommand(np.array([1, -1, 5]), 3)
    ]

    results = simulator.simulate(initial_attitude, initial_ang_vel, commands, dt)

    # --- Estimator setup ---
    estimator = AttitudeEstimatorQ(
        mag_dec_angle_deg,
        accel_error,
        mag_error,
        gyro_error,
        dt
    )

    true_eulers = []
    estimated_eulers = []

    for t in results['attitude'].keys():
        estimator.update_accelerometer(results['accelerometer'][t])
        estimator.update_gyroscope(results['gyroscope'][t])
        estimator.update_magnetometer(results['magnetometer'][t])
        estimator.update_attitude(initial_attitude)

        est_q = estimator._attitude.to_array()
        true_q = results['attitude'][t]

        est_euler = quaternion_to_euler_zyx(est_q)
        true_euler = quaternion_to_euler_zyx(true_q)

        estimated_eulers.append(est_euler)
        true_eulers.append(true_euler)

    estimated_eulers = np.array(estimated_eulers)
    true_eulers = np.array(true_eulers)

    # Compute error and standard deviation
    error = true_eulers - estimated_eulers
    error = (error + 180) % 360 - 180
    yaw_err_std, pitch_err_std, roll_err_std = np.std(error, axis=0)

    print(f"Standard Deviation of Yaw Error:   {yaw_err_std:.3f} deg")
    print(f"Standard Deviation of Pitch Error: {pitch_err_std:.3f} deg")
    print(f"Standard Deviation of Roll Error:  {roll_err_std:.3f} deg")


if __name__ == "__main__":
    test_estimator()
