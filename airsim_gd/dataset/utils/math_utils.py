import numpy as np
import airsimneurips as airsim


def rotation_matrix_from_angles(pry):
    pitch = pry[0]
    roll = pry[1]
    yaw = pry[2]
    sy = np.sin(yaw)
    cy = np.cos(yaw)
    sp = np.sin(pitch)
    cp = np.cos(pitch)
    sr = np.sin(roll)
    cr = np.cos(roll)

    Rx = np.array([
                    [1, 0, 0],
                    [0, cr, -sr],
                    [0, sr, cr]
                    ])

    Ry = np.array([
                    [cp, 0, sp],
                    [0, 1, 0],
                    [-sp, 0, cp]
                    ])

    Rz = np.array([
                    [cy, -sy, 0],
                    [sy, cy, 0],
                    [0, 0, 1]
                    ])

    # Roll is applied first, then pitch, then yaw.
    RyRx = np.matmul(Ry, Rx)
    return np.matmul(Rz, RyRx)


def rotvec2quat(rotaxis, theta):
    rotaxis = np.array(rotaxis)
    rotaxis = rotaxis/np.linalg.norm(rotaxis)
    quat = [np.cos(0.5*theta), rotaxis[0]*np.sin(0.5*theta), rotaxis[1]*np.sin(0.5*theta), rotaxis[2]*np.sin(0.5*theta)]
    return quat


def quat2mat(airsim_quat):
    # convert gate quaternion to rotation matrix.
    # ref: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion; https://www.lfd.uci.edu/~gohlke/code/transformations.py.html

    q = np.array([airsim_quat.w_val, airsim_quat.x_val, airsim_quat.y_val, airsim_quat.z_val], dtype=np.float64)
    n = np.dot(q, q)
    if n < np.finfo(float).eps:
        return airsim.Vector3r(0.0, 1.0, 0.0)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rotation_matrix = np.array([[1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
                                [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
                                [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]])
    # gate_facing_vector = rotation_matrix[:, 1]
    return rotation_matrix
