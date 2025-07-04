import numpy as np

class Quaternion:

    def __init__(self, w: float, x: float, y: float, z: float):
        """
        Initializes a quaternion with the given components.
        The quaternion is represented as q = w + xi + yj + zk
        """
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        """
        Returns a string representation of the quaternion.
        """
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"

    def to_array(self) -> np.ndarray:
        """
        Converts the quaternion to a numpy array representation.
        """
        return np.array([self.w, self.x, self.y, self.z])

    def multiply(self, other: 'Quaternion') -> 'Quaternion':
        """
        Multiplies this quaternion with another quaternion.
        """
        if not isinstance(other, Quaternion):
            raise TypeError("Can only multiply with another Quaternion")

        w1, x1, y1, z1 = self.to_array()
        w2, x2, y2, z2 = other.to_array()

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return Quaternion(w, x, y, z)
    
    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """
        Overloads the multiplication operator to multiply two quaternions.
        """
        return self.multiply(other)

    def conjugate(self) -> 'Quaternion':
        """
        Returns the conjugate of the quaternion.
        """
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self) -> float:
        """
        Returns the norm (magnitude) of the quaternion.
        """
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def scale(self, scalar: float) -> 'Quaternion':
        """
        Scales the quaternion by a scalar value.
        """
        if not isinstance(scalar, (int, float)):
            raise TypeError("Scalar must be a number")
        return Quaternion(*(scalar * self.to_array()))

    def normalize(self) -> 'Quaternion':
        """
        Normalizes the quaternion to have a unit norm.
        """
        norm = self.norm()
        if norm == 0:
            raise ZeroDivisionError("Cannot normalize a zero quaternion")
        return self.scale(1.0 / norm)

    def inverse(self) -> 'Quaternion':
        """
        Returns the inverse of the quaternion.
        """
        norm_sq = self.norm()**2
        return self.conjugate().scale(1.0 / norm_sq)
    
    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Rotates a 3D vector `v` using this quaternion (must be unit quaternion).
        """
        if not isinstance(v, np.ndarray) or v.shape != (3,):
            raise ValueError("Input vector must be a 3D numpy array")

        if not np.isclose(self.norm(), 1.0, atol=1e-5):
            raise ValueError("Cannot rotate with a non-unit quaternion")
        
        q_vec = Quaternion(0, *v)
        rotated = self * q_vec * self.inverse()
        return np.array([rotated.x, rotated.y, rotated.z])
    
    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle_rad: float) -> 'Quaternion':
        """
        Creates a unit quaternion from an axis and an angle in radians.
        """
        if not isinstance(axis, np.ndarray) or axis.shape != (3,) or np.linalg.norm(axis) == 0:
            raise ValueError("Axis must be a non-zero 3D numpy array")
        if not isinstance(angle_rad, (int, float)):
            raise ValueError("Angle must be a number (in radians)")
        
        axis = axis / np.linalg.norm(axis)
        w = np.cos(angle_rad / 2)
        v = axis * np.sin(angle_rad / 2)
        return Quaternion(w, *v)

