import numpy as np
from scipy.spatial.transform import Rotation as R
from simulator import AttitudeSimulator
from quaternion import *

class AttitudeEstimatorQ():

    def __init__(self, dt, mag_dec_angle):
        self._attitude = np.array([1, 0, 0, 0])
        self._attitude_covariance = np.eye(4)

        self._mag = np.zeros(3)
        self._gyro = np.zeros(3)
        self._accel = np.zeros(3)

        self._dt = dt
        self._gyro_error = 0
        self._acc_error = 10000
        self._mag_error = 100000
        self.magnetic_declination_angle = mag_dec_angle

    def initialize_attitude(self):
        k = -(self._accel)
        k = k / np.linalg.norm(k)
        declination_quaternion = np.concatenate(([np.cos(self.magnetic_declination_angle)], np.sin(self.magnetic_declination_angle) * k))
        i = self._mag - k * np.dot(self._mag, k)
        i = i / np.linalg.norm(i)
        i = quaternion_conjugate(declination_quaternion, i)
        j = np.cross(k, i)
        j = quaternion_conjugate(declination_quaternion, j)
        dcm = np.array([i, j, k])
        self._attitude = R.from_matrix(dcm).as_quat()

    def update_magnetometer(self, data):
        self._mag = data
        
    def update_gyroscope(self, data):
        self._gyro = data

    def update_accelerometer(self, data):
        self._accel = data

    def update_attitude(self):

        angle = self._dt * np.linalg.norm(self._gyro)
        revolve_vec = self._gyro / np.linalg.norm(self._gyro)
        q_gyro = np.concatenate(([np.cos(angle)], np.sin(angle) * revolve_vec))
        a, b, c, d = q_gyro
        m_gyro = np.array([[a, -b, -c, -d], 
                           [b,  a,  d, -c], 
                           [c, -d,  a,  b], 
                           [d,  c, -b,  a]])
        Q = np.array([[0, 0, 0, 0],
                      [0, 0, 1, 0], 
                      [0, 1, 0, 0], 
                      [0, 0, 0, 1]]) * np.random.normal(0, self._gyro_error)
        
        x = quaternion_multiply(self._attitude, q_gyro)
        P = m_gyro @ self._attitude_covariance @ m_gyro.T + Q

        R1 = np.eye(3)*np.random.normal(0, self._acc_error)
        R2 = np.eye(3)*np.random.normal(0, self._mag_error)
        R = np.block([[R1, np.zeros((3,3))], [np.zeros((3,3)), R2]])
        k_observed = -self._accel / np.linalg.norm(self._accel)
        q_declination = np.concatenate(([np.cos(self.magnetic_declination_angle)], np.sin(self.magnetic_declination_angle) * k_observed))
        dec_correction = quaternion_conjugate(q_declination, self._mag)
        i_observed = dec_correction / np.linalg.norm(dec_correction)

        measurement = np.concatenate((i_observed, k_observed))
        w, x, y, z = self._attitude
        H = 2 * np.array([[w, x, -y, -z], 
                          [-z, y, x, -w], 
                          [y, z, w, x], 
                          [-y, z, -w, x], 
                          [x, w, z, y], 
                          [w, -x, -y, z]])

        i_pred = quaternion_conjugate(quaternion_inverse(self._attitude), np.array([1, 0, 0]))
        k_pred = quaternion_conjugate(quaternion_inverse(self._attitude), np.array([0, 0, 1]))
        pred = np.concatenate((i_pred, k_pred))
        
        y = measurement - pred
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        x = x + K @ y
        self._attitude = x / np.linalg.norm(x)
        self._attitude_covariance = (np.eye(4) - K @ H) @ P

        return


def main():

    estimator = AttitudeEstimatorQ(dt=0.1, mag_dec_angle=2.0)
    
    simulator = AttitudeSimulator(dt=0.1, 
                                  initial_attitude=np.array([1, 0, 0, 0]), 
                                  ang_vel=np.array([1, 1, 1]), 
                                  mag_dec_angle=2.0)

    for i in range(10):
        accel, mag, gyro = simulator.simulate_measurements()
        estimator.update_magnetometer(mag)
        estimator.update_gyroscope(gyro)
        estimator.update_accelerometer(accel)
        estimator.update_attitude()
        print(simulator._attitude)
        print(estimator._attitude)
        print()
        # print(np.linalg.norm(simulator._attitude - estimator._attitude))


if __name__ == "__main__":
    main()