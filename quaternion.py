import numpy as np

def quaternion_multiply(q1, q2):
    w1, v1 = q1[0], q1[1:]
    w2, v2 = q2[0], q2[1:]
    w = w1*w2 - np.dot(v1, v2)
    v = w2*v1 + w1*v2 + np.cross(v1, v2)
    return np.concatenate(([w], v))

def quaternion_conjugate(q, v):
    return quaternion_multiply(q, quaternion_multiply(np.concatenate(([0], v)), quaternion_inverse(q)))[1:]

def quaternion_inverse(q):
    return np.concatenate(([q[0]], -q[1:]))
