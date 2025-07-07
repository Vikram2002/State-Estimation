import numpy as np
from quaternion import Quaternion
from utils import transform_vector_euler

"""
Attitude Simulator Class takes as input a series of angular acceleration commands 
and simulates the attitude of a rigid body in space while keeping track of sensor measurements.

The attitude is a quaternion that represents an SO3 transformation from body frame to world frame.

Measurements include: 
    1. Gyroscope: To keep track of angular velocity 
    2. Accelerometer: To keep track of the gravitational down vector
    3. Magnetometer: To keep track of the magnetic north vector

All reference frames are assumed to be East, North, Up (ENU).

"""

class AngularAccelerationCommand:
    def __init__(self, angular_accel: np.ndarray, duration: float):
        """
        Represents an angular acceleration command for a certain duration in seconds.
        """
        if not isinstance(angular_accel, np.ndarray) or angular_accel.shape != (3,):
            raise ValueError("angular_accel must be a 3D numpy array")
        if duration <= 0:
            raise ValueError("duration must be a positive float")

        self.angular_accel = angular_accel
        self.duration = duration


class AttitudeSimulator():

    def __init__(self, mag_dec_angle_deg: float, mag_inc_angle_deg: float, 
                 accel_error: float, mag_error: float, gyro_error: float):
        """
        Initialize the attitude simulator with the following parameters:
            mag_dec_angle_deg: Magnetic declination angle in degrees
            mag_inc_angle_deg: Magnetic inclination angle in degrees  
            accel_error: Accelerometer measurement variance
            mag_error: Magnetometer measurement variance
            gyro_error: Gyroscope measurement variance
        """

        # Create world magnetic north vector with declination and inclination
        self._world_mag_north_unit_vec = transform_vector_euler(
            np.array([0, 1, 0]), 
            np.radians(-mag_inc_angle_deg), 
            0, 
            np.radians(-mag_dec_angle_deg)
        )
        self._world_gravity_vec = np.array([0, 0, -9.81])

        self._accel_error = accel_error
        self._mag_error = mag_error
        self._gyro_error = gyro_error

    def simulate(self, initial_attitude: Quaternion, initial_ang_vel: np.ndarray, 
                angular_acc_commands: list[AngularAccelerationCommand], dt: float) -> dict[str, dict[float, np.ndarray]]:
        """
        Simulate attitude dynamics based on angular acceleration commands.
        
        Args:
            initial_attitude: Initial attitude quaternion [w, x, y, z]
            initial_ang_vel: Initial angular velocity [w_x, w_y, w_z]
            angular_acc_commands: List of angular acceleration commands
            dt: Time step for simulation
            
        Returns:
            Dictionary containing time series of attitude and sensor measurements
        """
        # Initialize state
        attitude = initial_attitude
        ang_vel = initial_ang_vel
        
        # Initialize output data structures
        attitude_data = {}
        accelerometer_data = {}
        magnetometer_data = {}
        gyroscope_data = {}
        
        time = 0.0
        round_digits = max(0, -int(np.floor(np.log10(dt)))) + 1  # To help with rounding time values

        # Obtain sensor measurements at current time step
        measurements = self.get_measurements(attitude, ang_vel) 

        # Store data
        attitude_data[time] = attitude.to_array()
        accelerometer_data[time] = measurements['accelerometer']
        magnetometer_data[time] = measurements['magnetometer']
        gyroscope_data[time] = measurements['gyroscope']
        
        # Process each angular acceleration command
        for command in angular_acc_commands:
            
            for _ in range(int(command.duration / dt)):

                # Update attitude using quaternion integration
                attitude = attitude * Quaternion.from_axis_angle(ang_vel, np.linalg.norm(ang_vel) * dt)
                attitude = attitude.normalize()

                # Update angular velocity using Euler integration
                ang_vel += command.angular_accel * dt

                # Update time
                time += dt
                time = round(time, round_digits) # Round time to avoid floating point precision issues

                # Obtain sensor measurements at current time step
                measurements = self.get_measurements(attitude, ang_vel) 

                # Store data
                attitude_data[time] = attitude.to_array()
                accelerometer_data[time] = measurements['accelerometer']
                magnetometer_data[time] = measurements['magnetometer']
                gyroscope_data[time] = measurements['gyroscope']

        return {
            'attitude': attitude_data,
            'accelerometer': accelerometer_data,
            'magnetometer': magnetometer_data,
            'gyroscope': gyroscope_data
        }
    
    def get_measurements(self, attitude: Quaternion, ang_vel: np.ndarray) -> dict[str, np.ndarray]:
        # Transform gravity and magnetic north vectors from world frame to body frame
        accel_true = attitude.inverse().rotate_vector(self._world_gravity_vec)
        mag_true = attitude.inverse().rotate_vector(self._world_mag_north_unit_vec)

        # Gyroscope measurement is the angular velocity in body frame
        gyro_true = ang_vel
        
        # Add measurement noise
        accel_noisy = accel_true + np.random.normal(0, self._accel_error, 3)
        mag_noisy = mag_true + np.random.normal(0, self._mag_error, 3)
        gyro_noisy = gyro_true + np.random.normal(0, self._gyro_error, 3)

        return {
            'accelerometer': accel_noisy,
            'magnetometer': mag_noisy,
            'gyroscope': gyro_noisy
        }


def main():
    """Example usage of the attitude simulator."""

    # Initialize simulator with:
    #   - Magnetic declination angle of 4 degrees
    #   - Magnetic inclination angle of 60 degrees (typical for mid-latitudes)
    #   - Accelerometer Measurement Variance: 0.3
    #   - Magnetometer Measurement Variance: 0.1
    #   - Gyroscope Measurement Variance: 0.2
    
    simulator = AttitudeSimulator(
        mag_dec_angle_deg=4.0,
        mag_inc_angle_deg=60.0,
        accel_error=0.0,
        mag_error=0.0,
        gyro_error=0.0
    )
    
    # Create angular acceleration commands
    commands = [
        AngularAccelerationCommand(np.array([np.pi, 0, 0]), 1),
        AngularAccelerationCommand(np.array([0, 0, 0]), 5),
        AngularAccelerationCommand(np.array([-np.pi, 0, 0]), 1)
    ]

    # Initial conditions
    initial_attitude = Quaternion(1, 0, 0, 0)  # Identity quaternion
    initial_ang_vel = np.array([0.0, 0.0, 0.0])      # No initial angular velocity

    # Time step for simulation in seconds
    dt = 0.01
    
    # Run simulation
    results = simulator.simulate(initial_attitude, initial_ang_vel, commands, dt)
    
    # Print results
    for i, t in enumerate(results['attitude'].keys()):
        if i % 10 == 0:  # Print every 10th sample to avoid too much output
            print(f"Time: {t:.3f}")
            print(f"    Attitude: {results['attitude'][t]}")
            print(f"    Accelerometer: {results['accelerometer'][t]}")
            print(f"    Magnetometer: {results['magnetometer'][t]}")
            print(f"    Gyroscope: {results['gyroscope'][t]}")
            print()


if __name__ == "__main__":
    main()
