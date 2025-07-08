import numpy as np
from scipy.spatial.transform import Rotation as R
from attitude_simulator import AttitudeSimulator, AngularAccelerationCommand
from quaternion import Quaternion

class AttitudeEstimatorQ():

    def __init__(self, mag_dec_angle_deg, accel_error, mag_error, gyro_error, dt):

        # Initialize state variables and sensor data
        self._attitude = Quaternion(1, 0, 0, 0)
        self._attitude_cov = np.zeros((4, 4))
        self._mag = np.zeros(3)
        self._gyro = {'previous': np.zeros(3), 'current': np.zeros(3)}
        self._accel = np.zeros(3)

        # Initialize parameters
        self._gyro_error = gyro_error
        self._acc_error = accel_error
        self._mag_error = mag_error
        self._magnetic_declination_angle = mag_dec_angle_deg
        self._dt = dt

        # Tune accordingly
        self._GYRO_NORM = 4 * np.pi
        self._ACC_NORM = 10
        self._MAG_NORM = 5e-5
        
        # Helps with initialization
        self._initialized = False

    def get_attitude_from_measurements(self):
        """ 
        Obtains the attitude based on the accelerometer and magnetometer readings.
        """
        # Get world k-hat vector via gravity vector
        if np.linalg.norm(self._accel) == 0:
            raise ValueError("Cannot detect gravity vector. Accelerometer data is zero.")
        k_world = -(self._accel) / np.linalg.norm(self._accel)
    
        # Get world j-hat vector via magnetic north vector
        # Remove inclination
        mag = self._mag - np.dot(self._mag, k_world) * k_world 
        if np.linalg.norm(mag) == 0:
            raise ValueError("Magnetometer is either zero or incorrectly aligned with gravity vector.")
        # Remove declination
        dec_q = Quaternion.from_axis_angle(k_world, np.radians(self._magnetic_declination_angle))
        mag = dec_q.rotate_vector(mag)
        # Normalize the adjusted mag vector
        j_world = mag / np.linalg.norm(mag)

        # Get world i-hat vector by using the cross product
        i_world = np.cross(j_world, k_world)

        # Create direction cosine matrix (DCM) that maps bodq_y frame to world frame
        dcm = np.array([i_world, j_world, k_world])

        # Compute the attitude quaternion from the DCM
        x, y, z, w = R.from_matrix(dcm).as_quat()
        return Quaternion(w, x, y, z)

    def update_magnetometer(self, data):
        self._mag = data
        
    def update_gyroscope(self, data):
        self._gyro['previous'] = self._gyro['current']
        self._gyro['current'] = data

    def update_accelerometer(self, data):
        self._accel = data

    def update_attitude(self, initial_attitude=None):
        """
        Attitude update step uses a modified Kalman Filter.

            Motion Model: The update equations use angular velocity to perform regular quaternion integration.

            Measurement Model: The sensor measurments are used to indirectly measure the state 
                               and construct the innovation vector in state space instead of measurment space.
                               This helps bypass the non-linearities involved in computing direct measurments 
                               from the attitude.

            The noise matrices are constructed by appropriately scaling noise parameters to be relative 
            to the magnitude of the measurement as determined by [_ACC/_GYRO/_MAG]_NORM variables.
        """
        # Initialize the attitude at time t=0
        if not self._initialized:
            # If initial attitude has been provided, then use it, otherwise determine from sensor measurements
            if initial_attitude is not None:
                self._attitude = initial_attitude
                self._attitude_cov = np.zeros((4, 4))
            else:
                self._attitude = self.get_attitude_from_measurements()
                self._attitude_cov = np.eye(4) * (gamma(self._acc_error / self._ACC_NORM) + gamma(self._mag_error / self._MAG_NORM))
            self._initialized = True
            return

        # Compute delta quaternion using angular velocity at the previous time step
        gyro_prev = self._gyro['previous']
        delta_q = Quaternion.from_axis_angle(gyro_prev, self._dt * np.linalg.norm(gyro_prev))

        # Compute the state update matrix
        dq_w, dq_x, dq_y, dq_z = delta_q.to_array()
        A = np.array([[dq_w, -dq_x, -dq_y, -dq_z], 
                      [dq_x,  dq_w,  dq_z, -dq_y], 
                      [dq_y, -dq_z,  dq_w,  dq_x], 
                      [dq_z,  dq_y, -dq_x,  dq_w]])
        
        # Process noise covariance matrix
        Q = np.diag([1, 1, 1, 1]) * gamma(self._gyro_error / self._GYRO_NORM)

        # EKF prediction step
        X = (self._attitude * delta_q).to_array()
        P = A @ self._attitude_cov @ A.T + Q

        # Measurement model
        H = np.eye(4)

        # Measurement noise covariance matrix
        R = np.eye(4) * (gamma(self._acc_error / self._ACC_NORM) + gamma(self._mag_error / self._MAG_NORM))
        
        # Compute the innovation vector Y by choosing the appropriate quaternion (q or -q) closer to prediction
        pred = H @ X
        measurement = self.get_attitude_from_measurements().to_array()
        neg_measurement = -measurement
        Y_pos = measurement - pred
        Y_neg = neg_measurement - pred
        Y = Y_pos if np.linalg.norm(Y_pos) < np.linalg.norm(Y_neg) else Y_neg

        # Compute the Kalman Gain K
        S = H @ P @ H.T + R
        if np.linalg.det(S) == 0:
            K = np.zeros((4, 4))
        else:
            K = P @ H.T @ np.linalg.inv(S)

        # EKF Measurement update step
        X = X + K @ Y
        P = (np.eye(4) - K @ H) @ P

        # Renormalize and store results in state variables
        self._attitude = Quaternion.from_array(X / np.linalg.norm(X))
        self._attitude_cov = P
        return


def gamma(x: float) -> float:
    """
    Function to help scale noise parameters.
    gamma(x) goes from 0 to INF as x goes from 0 to 1
    GAMMA_FACTOR determines how quickly the function increases
    """
    # Tune accordingly
    INF = 1e5 
    GAMMA_FACTOR = 2
    
    return INF * x ** (1/GAMMA_FACTOR)
