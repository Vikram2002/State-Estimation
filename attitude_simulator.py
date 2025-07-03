import numpy as np
from quaternion import *

class AttitudeSimulator():

    def __init__(self, dt, initial_attitude, ang_vel, mag_dec_angle, accel_error, mag_error, gyro_error):
        self._time_step = dt
        self._attitude = initial_attitude
        self._ang_vel = None
        self._ang_vel_q = None
        self.set_angular_velocity(ang_vel)
        self._wrld_mag_vec = self.get_magnetic_north_vector(mag_dec_angle)
        self._accel_error = accel_error
        self._mag_error = mag_error
        self._gyro_error = gyro_error

    def get_magnetic_north_vector(self, mag_dec_angle):
        mag_dec_q = np.concatenate(([np.cos(-mag_dec_angle/2*np.pi/180)], np.sin(-mag_dec_angle/2*np.pi/180) * np.array([0, 0, 1])))
        return quaternion_conjugate(mag_dec_q, np.array([1, 0, 0]))

    def set_angular_velocity(self, ang_vel):
        if ang_vel is not None:
            self._ang_vel = ang_vel
            revolve_vec = ang_vel / np.linalg.norm(ang_vel)
            angle = self._time_step * np.linalg.norm(ang_vel)
            self._ang_vel_q = np.concatenate(([np.cos(angle/2)], np.sin(angle/2) * revolve_vec))

    def update_state(self):
        self._attitude = quaternion_multiply(self._ang_vel_q, self._attitude)
        self._attitude /= np.linalg.norm(self._attitude)

    def get_measurements(self):
        accel_mu = quaternion_conjugate(quaternion_inverse(self._attitude), np.array([0, 0, 9.8]))
        mag_mu = quaternion_conjugate(quaternion_inverse(self._attitude), self._wrld_mag_vec)
        gyro_mu = quaternion_conjugate(quaternion_inverse(self._attitude), self._ang_vel)
        accel = np.random.multivariate_normal(accel_mu, self._accel_error * np.eye(3))
        mag = np.random.multivariate_normal(mag_mu, self._mag_error * np.eye(3))
        gyro = np.random.multivariate_normal(gyro_mu, self._gyro_error * np.eye(3))
        return accel, mag, gyro
    
    

def main():

    dt = 0.01
    accelerometer_data = dict() 
    magnetometer_data = dict() 
    gyroscope_data = dict()
    attitude_data = dict()

    # Initialize simulator with:
    #   1. Time step: dt
    #   2. Initial Attitude: (1, 0, 0, 0)
    #   3. No angular velocity
    #   4. Magnetic declination angle of 4 degrees
    #   5. Accelerometer Measurement Variance: 0.3
    #   6. Magnetometer Measurement Variance: 0.1
    #   7. Gyroscope Measurement Variance: 0.2

    simulator = AttitudeSimulator(dt=dt, 
                                  initial_attitude=np.array([1, 0, 0, 0]), 
                                  ang_vel=None, 
                                  mag_dec_angle=4.0,
                                  accel_error = 0.0,
                                  mag_error = 0.0,
                                  gyro_error = 0.0)
    
    # Simulate the following series of actions:
        # 1. Rotation with angular velocity (PI, 0, 0) for 0.5 second
        # 2. Rotation with angular velocity (0, -PI/2, 0) for 1 second
        # 3. Rotation with angular velocity (0, 0, PI) for 0.5 second
        # 4. Rotation with angular velocity (-PI, 0, 0) for 0.5 second
        # 5. Rotation with angular velocity (0, 0, -PI) for 0.5 second 

    # Note: Angular velocity is given in world frame coordinates
    
    commands = [{"angular velocity": np.array((np.pi, 0, 0)), "seconds": 0.5},
                {"angular velocity": np.array((0, -np.pi/2, 0)), "seconds": 1},
                {"angular velocity": np.array((0, 0, np.pi)), "seconds": 0.5},
                {"angular velocity": np.array((-np.pi, 0, 0)), "seconds": 0.5},
                {"angular velocity": np.array((0, 0, -np.pi)), "seconds": 0.5},]

    time = 0.0
    count = 0
    measurement_frequency = (int)(1.0/dt)  # How often to read sensors (Hz)

    for command in commands:

        simulator.set_angular_velocity(command["angular velocity"])

        for n in range((int) (command["seconds"] / dt)):
            attitude_data[time] = simulator._attitude

            if time >= count * 1 / measurement_frequency:
                a, m, g = simulator.get_measurements()
                accelerometer_data[time] = np.round(a, 3)
                magnetometer_data[time] = np.round(m, 3)
                gyroscope_data[time] = np.round(g, 3)
                count += 1

            simulator.update_state()
            time += dt
            time = round(time, 5)

    for t in attitude_data:
        print("Time:", t)
        print("    Attitude:", attitude_data[t])
        print("    Accelerometer:", accelerometer_data[t] if t in accelerometer_data else None)
        print("    Magnetometer:", magnetometer_data[t] if t in magnetometer_data else None)
        print("    Gyroscope:", gyroscope_data[t] if t in gyroscope_data else None)
        print()


if __name__ == "__main__":
    main()

        
