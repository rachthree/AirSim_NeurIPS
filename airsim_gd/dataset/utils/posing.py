import airsimneurips as airsim
import numpy as np


# Scale 1:
# inner dimensions (width, thickness, height) 1.6m x 0.2 m x 1.6 m
# outer dimensions (width, thickness, height): 2.1333 m x 0.2 m x 2.1333 m
# gate origin is at center

def rotvec2quat(rotaxis, theta):
    rotaxis = np.array(rotaxis)
    rotaxis = rotaxis/np.linalg.norm(rotaxis)
    quat = [np.cos(0.5*theta), rotaxis[0]*np.sin(0.5*theta), rotaxis[1]*np.sin(0.5*theta), rotaxis[2]*np.sin(0.5*theta)]
    return quat


def quat2norm(airsim_quat):
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
    gate_facing_vector = rotation_matrix[:, 1]
    return gate_facing_vector


def pose_object(airsim_client, object_name, xyz, axisrot, theta, teleport=True):
    quat = rotvec2quat(axisrot, theta)
    pose = airsim.Pose(position_val={'x_val': xyz[0], 'y_val': xyz[1], 'z_val': xyz[2]},
                       orientation_val={'w_val': quat[0], 'x_val': quat[1], 'y_val': quat[2], 'z_val': quat[3]})
    pose = airsim_client.simSetObjectPose(object_name, pose, teleport=teleport)
    return pose


def pose_drone(airsim_client, object_name, xyz, axisrot, theta, ignore_collision=False, pause_sim=True):
    # need to pause sim otherwise physics happens
    airsim_client.simPause(pause_sim)

    quat = rotvec2quat(axisrot, theta)
    pose = airsim.Pose(position_val={'x_val': xyz[0], 'y_val': xyz[1], 'z_val': xyz[2]},
                       orientation_val={'w_val': quat[0], 'x_val': quat[1], 'y_val': quat[2], 'z_val': quat[3]})
    pose = airsim_client.simSetVehiclePose(pose, ignore_collision, vehicle_name=object_name)
    return pose


def place_gate(airsim_client, gate_name, h_limit, rx, ry, rz):
    # TODO: Place gate with rotations, also use collision detection
    # based on get_ground_truth_gate_poses from baseline_racer.py
    gate_names_sorted_bad = sorted(self.airsim_client.simListSceneObjects("Gate.*"))
    # gate_names_sorted_bad is of the form `GateN_GARBAGE`. for example:
    # ['Gate0', 'Gate10_21', 'Gate11_23', 'Gate1_3', 'Gate2_5', 'Gate3_7', 'Gate4_9', 'Gate5_11', 'Gate6_13', 'Gate7_15', 'Gate8_17', 'Gate9_19']
    # we sort them by their ibdex of occurence along the race track(N), and ignore the unreal garbage number after the underscore(GARBAGE)
    gate_indices_bad = [int(gate_name.split('_')[0][4:]) for gate_name in gate_names_sorted_bad]
    gate_indices_correct = sorted(range(len(gate_indices_bad)), key=lambda k:gate_indices_bad[k])
    gate_names_sorted = [gate_names_sorted_bad[gate_idx] for gate_idx in gate_indices_correct]
    self.gate_poses_ground_truth = [self.airsim_client.simGetObjectPose(gate_name) for gate_name in gate_names_sorted]

    airsim_client.simSetObjectPose(object_name, pose, teleport=True)
    # airsim.Pose(position_val={ 'x_val': 0.0, 'y_val': 0.0, 'z_val': 0.0}, orientation_val={ 'w_val': 1.0, 'x_val': 0.0, 'y_val': 0.0, 'z_val': 0.0})
    # orientation_val is quaternion

    # Global Coordinate system:
    # +X North, +Y East, +Z Down
    # client.simSetObjectPose('Gate00', airsim.Pose(position_val={'x_val': 0.0, 'y_val': 0.0, 'z_val': 0.0},
    #                                               orientation_val={'w_val': 0.1, 'x_val': 1.0, 'y_val': 1.0,
    #                                                                'z_val': 0.0}), teleport=True)


    pass

def place_drone(airsim_client, drone_name, gate_pose, dx, dy, dz, rx, ry, rz):
    # TODO: Place drone wrt gate
    airsim_client.simSetCameraOrientation(camera_name, orientation, vehicle_name='')

    pass