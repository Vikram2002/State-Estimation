import numpy as np
import pytest
from quaternion import Quaternion

def generate_random_quaternion() -> Quaternion:
    q = np.random.uniform(-10, 10, size=4)
    return Quaternion(*q)

def test_quaternion_representation():
    q = Quaternion(1, 2, 3, 4)
    assert repr(q) == "Quaternion(1, 2, 3, 4)"

def test_to_array():
    q = Quaternion(1, 2, 3, 4)
    assert np.array_equal(q.to_array(), np.array([1, 2, 3, 4]))

def test_multiply_identity():
    identity = Quaternion(1, 0, 0, 0)
    for _ in range(3):
        q = generate_random_quaternion()
        result_left = q * identity
        result_right = identity * q
        assert np.allclose(result_left.to_array(), q.to_array(), atol=1e-6)
        assert np.allclose(result_right.to_array(), q.to_array(), atol=1e-6)

def test_multiply_invalid_type():
    q = Quaternion(1, 2, 3, 4)
    with pytest.raises(TypeError):
        q * "not-a-quaternion"

def test_conjugate():
    q = Quaternion(1, 2, 3, 4)
    q_conj = q.conjugate()
    assert np.array_equal(q_conj.to_array(), np.array([1, -2, -3, -4]))

def test_norm():
    q = Quaternion(1, 2, 2, 1)
    expected = np.sqrt(1**2 + 2**2 + 2**2 + 1**2)
    assert np.isclose(q.norm(), expected)

def test_normalize():
    q = Quaternion(1, 2, 3, 4)
    q_normalized = q.normalize()
    assert np.isclose(q_normalized.norm(), 1.0, atol=1e-6)

def test_normalize_zero():
    q = Quaternion(0, 0, 0, 0)
    with pytest.raises(ZeroDivisionError):
        q.normalize()

def test_inverse():
    q = Quaternion(1, 2, 3, 4)
    q_inv = q.inverse()
    q_times_inv_left = q * q_inv
    q_times_inv_right = q_inv * q
    identity = Quaternion(1, 0, 0, 0)
    assert np.allclose(q_times_inv_left.to_array(), identity.to_array(), atol=1e-5)
    assert np.allclose(q_times_inv_right.to_array(), identity.to_array(), atol=1e-5)

def test_rotate_vector():
    axis = np.array([1, 1, 1])
    angle = 2 * np.pi / 3
    q = Quaternion.from_axis_angle(axis, angle)
    i = np.array([1, 0, 0])
    j = np.array([0, 1, 0])
    k = np.array([0, 0, 1])
    assert np.allclose(q.rotate_vector(i), j, atol=1e-6)
    assert np.allclose(q.rotate_vector(j), k, atol=1e-6)
    assert np.allclose(q.rotate_vector(k), i, atol=1e-6)

def test_rotate_vector_invalid_inputs():
    with pytest.raises(ValueError):
        q = Quaternion(1, 0, 0, 0)
        v = np.array([1, 2])
        q.rotate_vector(v)

    with pytest.raises(ValueError):
        q = Quaternion(2, 0, 0, 0)
        v = np.array([1, 0, 0])
        q.rotate_vector(v)    

def test_from_axis_angle_unit_norm():
    axis = np.array([1, 10, -4])
    angle = np.pi / 3
    q = Quaternion.from_axis_angle(axis, angle)
    assert np.isclose(q.norm(), 1.0, atol=1e-6)

def test_scale():
    q = Quaternion(1, 2, 3, 4)
    scaled = q.scale(0.5)
    assert np.allclose(scaled.to_array(), np.array([0.5, 1, 1.5, 2]))

def test_scale_invalid_input():
    q = Quaternion(1, 2, 3, 4)
    with pytest.raises(TypeError):
        q.scale("not-a-number")

def test_invalid_axis_angle_inputs():
    with pytest.raises(ValueError):
        Quaternion.from_axis_angle(np.array([0, 0, 0]), np.pi)

    with pytest.raises(ValueError):
        Quaternion.from_axis_angle(np.array([1, 0]), np.pi)

    with pytest.raises(ValueError):
        Quaternion.from_axis_angle(np.array([1, 0, 0]), "not-a-number")
