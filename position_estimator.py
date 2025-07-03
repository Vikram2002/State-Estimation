import numpy as np
from quaternion import *

class PositionEstimator():

    def __init__(self, dt, initial_position, initial_velocity):
        self.dt = dt
        self.time = 0
        self.x = np.concatenate((initial_position, initial_velocity))
        self.P = np.eye(6) * 0.1
        self.A = np.array([[1, 0, 0, dt, 0, 0], 
                           [0, 1, 0, 0, dt, 0],
                           [0, 0, 1, 0, 0, dt],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
        self.B = np.array([[0.5*dt**2, 0, 0],
                           [0, 0.5*dt**2, 0],
                           [0, 0, 0.5*dt**2],
                           [dt, 0, 0],
                           [0, dt, 0],
                           [0, 0, dt]])
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0]])


    # DRONE STATE: XYZ POSITION AND VELOCITY IN WORLD FRAME COORDINATES
    def update_state(self, accel_data, accel_error, gps_data, gps_error):
        # MOTION UPDATE STEP
        self.x = self.A @ self.x + self.B @ accel_data
        self.P = self.A @ self.P @ self.A.T + accel_error * np.eye(6)

        # MEASUREMENT STEP
        S = self.H @ self.P @ self.H.T + gps_error * np.eye(3)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (gps_data - self.H @ self.x)
        self.P = (np.eye(6) - K @ self.H) @ self.P


    # GET ACCELEROMETER MEASUREMENT OF GLOBAL POSITION
    # Currently simulating drone moving towards positive BF i-axis with constant acceleration of 1m/s^2
    def get_acceleration_data(self):
        accel_data = np.array([1, 0, 9.81])
        accel_error = 0.1
        accel_data = np.random.multivariate_normal(accel_data, accel_error * np.eye(3))
        return accel_data, accel_error
    

    # GET GPS MEASUREMENT OF GLOBAL POSITION
    # Currently simulating drone moving towards positive WF j-axis with constant acceleration of 1m/s^2
    def get_gps_data(self):
        gps_data = np.array([0, 0.5*self.time**2, 0])
        gps_error = 0.1
        gps_data = np.random.multivariate_normal(gps_data, gps_error * np.eye(3))
        return gps_data, gps_error
    

    # GET DRONE'S ATTITUDE QUATERNION (BODY FRAME -> WORLD FRAME)
    # Currently simulating a 90 degree rotation around k-hat
    def get_attitude_data(self):
        return np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])


    # CORRECTS ACCELEROMETER MEASUREMENTS:
    # 1. Removes the upward normal force measured by drone due to gravity
    # 2. Conjugates accelerometer readings by attitude quaternion to get in terms of world frame coordinates
    def correct_acceleration_data(self, acceleration):
        acceleration -= np.array([0, 0, 9.81])
        attitude = self.get_attitude_data()
        acceleration = quaternion_conjugate(attitude, acceleration)
        return acceleration


    # RUN KALMAN FILTER FOR SPECIFIED AMOUNT OF TIME IN SECONDS
    def run(self, time):
        output = dict()
        for t in np.arange(0, time, self.dt):
            self.time = t
            output[t] = {"Position": self.x[0:3], "Velocity": self.x[3:]}
            accel_data, accel_error = self.get_acceleration_data()
            ground_frame_accel = self.correct_acceleration_data(accel_data)
            gps_data, gps_error = self.get_gps_data()
            self.update_state(ground_frame_accel, accel_error, gps_data, gps_error)
        return output


def main():
    estimator = PositionEstimator(dt=0.1, 
                                  initial_position=[0, 0, 0], 
                                  initial_velocity=[0, 0, 0])
    
    output = estimator.run(10)
    for t in output:
        print("Time:", t, 
              "\n   Position:", output[t]['Position'], 
              "\n  Velocity:", output[t]['Velocity'],
              "\n")


if __name__ == "__main__":
    main()



    