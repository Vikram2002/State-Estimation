import numpy as np
from scipy.spatial.transform import Rotation as R
from attitude_simulator import *
from quaternion import Quaternion
from utils import transform_vector_euler

class AttitudeEstimatorQ():

    def __init__(self, mag_dec_angle_deg, mag_inc_angle_deg, accel_error, mag_error, gyro_error, dt):

        self._attitude = Quaternion(1, 0, 0, 0)
        self._attitude_cov = np.zeros((4, 4))

        self._mag = np.zeros(3)
        self._gyro = np.zeros(3)
        self._accel = np.zeros(3)

        self._gyro_error = gyro_error
        self._acc_error = accel_error
        self._mag_error = mag_error

        self._magnetic_declination_angle = mag_dec_angle_deg
        self._magnetic_inclination_angle = mag_inc_angle_deg

        self._dt = dt

        # Create world magnetic north vector with declination and inclination
        self._world_mag_north_vec = transform_vector_euler(
            np.array([0, 5e-5, 0]), 
            np.radians(-mag_inc_angle_deg), 
            0, 
            np.radians(-mag_dec_angle_deg)
        )
        self._world_gravity_vec = np.array([0, 0, -9.81])

        # Initialize world basis vectors in terms of the body frame
        self._body_i_world = np.zeros(3)
        self._body_j_world = np.zeros(3)
        self._body_k_world = np.zeros(3)

    def set_world_basis_vector(self):
        """
        Set the world basis vectors (i, j, k) in terms of the body frame vectors using an ENU frame.
        """
        # Obtain world k-hat vector by normalizing the gravity vector
        if np.linalg.norm(self._accel) == 0:
            raise ValueError("Cannot detect gravity vector. Accelerometer data is zero.")
        self._body_k_world = -(self._accel) / np.linalg.norm(self._accel)
    
        # Obtain world j-hat vector by using the magnetometer vector
        mag = self._mag - np.dot(self._mag, self._body_k_world) * self._body_k_world # Removes inclination
        if np.linalg.norm(mag) == 0:
            raise ValueError("Magnetometer is either zero or incorrectly aligned with gravity vector.")
        dec_q = Quaternion.from_axis_angle(self._body_k_world, np.radians(self._magnetic_declination_angle))
        mag = dec_q.rotate_vector(mag)  # Removes declination
        self._body_j_world = mag / np.linalg.norm(mag)

        # Obtain world i-hat vector by using the cross product
        self._body_i_world = np.cross(self._body_j_world, self._body_k_world)

    def initialize_attitude(self):
        """ 
        Initialize the attitude based on the accelerometer and magnetometer readings.
        Computes world basis vectors in term of the body frame and obtains the attitude quaternion.
        """

        self.set_world_basis_vector()

        # Create direction cosine matrix (DCM) that maps body frame to world frame
        dcm = np.array([self._body_i_world, self._body_j_world, self._body_k_world])

        # Compute the attitude quaternion from the DCM
        x, y, z, w = R.from_matrix(dcm).as_quat()
        self._attitude = Quaternion(w, x, y, z)

    def update_magnetometer(self, data):
        self._mag = data
        
    def update_gyroscope(self, data):
        self._gyro = data

    def update_accelerometer(self, data):
        self._accel = data

    def update_attitude(self):

        delta_q = Quaternion.from_axis_angle(self._gyro, self._dt * np.linalg.norm(self._gyro))

        # Compute the Jacobian of the attitude update
        dw, dx, dy, dz = delta_q.to_array()
        A = np.array([[dw, -dx, -dy, -dz], 
                      [dx,  dw,  dz, -dy], 
                      [dy, -dz,  dw,  dx], 
                      [dz,  dy, -dx,  dw]])
        
        # Process noise covariance matrix
        Q = np.diag([0, 1, 1, 1]) * (self._gyro_error)

        # Compute the Jacobian of the measurement model
        w, x, y, z = self._attitude.to_array()
        H = np.array([
            [0,     0,     -4*y,   -4*z],
            [2*z,   2*y,    2*x,    2*w],
            [-2*y,  2*z,   -2*w,    2*x],
            [-2*z,  2*y,    2*x,   -2*w],
            [0,    -4*x,    0,    -4*z],
            [2*x,   2*w,    2*z,    2*y],
            [2*y,   2*z,    2*w,    2*x],
            [-2*x, -2*w,    2*z,    2*y],
            [0,    -4*x,   -4*y,    0]
        ])

        # Measurement noise covariance matrix
        R = np.eye(9) * (self._acc_error + self._mag_error)
        
        # EKF prediction step
        X = (self._attitude * delta_q).to_array()
        P = A @ self._attitude_cov @ A.T + Q

        # EKF measurement update step
        self.set_world_basis_vector()
        measurement = np.concatenate((self._body_i_world, self._body_j_world, self._body_k_world))
        pred = self._attitude.to_dcm().flatten(order='F')
        y = measurement - pred

        S = H @ P @ H.T + R
        if np.linalg.det(S) == 0:
            K = np.zeros((P.shape[0], H.shape[0]))
        else:
            K = P @ H.T @ np.linalg.inv(S)

        X = X + K @ y

        self._attitude = Quaternion.from_array(X / np.linalg.norm(X))
        self._attitude_cov = (np.eye(4) - K @ H) @ P

        return


def main():

    mag_dec_angle_deg = 4.0
    mag_inc_angle_deg = 60.0
    accel_error = 0.0
    mag_error = 0.0
    gyro_error = 0.0

    simulator = AttitudeSimulator(
        mag_dec_angle_deg,
        mag_inc_angle_deg,
        accel_error,
        mag_error,
        gyro_error
    )

    initial_attitude = Quaternion.from_axis_angle(np.array([1, 0, 0]), np.pi / 2)
    initial_ang_vel = np.array([0.0, 0.0, 0.0])
    commands = [AngularAccelerationCommand(np.array([np.pi, 0, 0]), 1),
                AngularAccelerationCommand(np.array([0, 0, 0]), 5),
                AngularAccelerationCommand(np.array([-np.pi, 0, 0]), 1)
    ]
    dt = 0.01
                
    simulator_results = simulator.simulate(initial_attitude,
                                           initial_ang_vel,
                                           commands,
                                           dt
    )

    estimator = AttitudeEstimatorQ(
        mag_dec_angle_deg,
        mag_inc_angle_deg,
        accel_error,
        mag_error,
        gyro_error,
        dt
    )

    estimated_attitude = {} 

    for time in simulator_results['attitude'].keys():

        estimator.update_magnetometer(simulator_results['magnetometer'][time])
        estimator.update_gyroscope(simulator_results['gyroscope'][time])
        estimator.update_accelerometer(simulator_results['accelerometer'][time])

        if time == 0:
            estimator.initialize_attitude()
        else:
            estimator.update_attitude()

        estimated_attitude[time] = estimator._attitude.to_array()

    # Print results
    print_every = 10

    for idx, time in enumerate(simulator_results['attitude'].keys()):
        if idx % print_every == 0:
            est_q = estimated_attitude[time]
            actual_q = simulator_results['attitude'][time]

            print(f"Time      - {time:.3f}")
            print(f"Actual    - {actual_q}")
            print(f"Estimated - {est_q}\n")

if __name__ == "__main__":
    main()