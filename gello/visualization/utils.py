import numpy as np
from scipy.spatial.transform import Rotation
import rerun as rr


def path_to_link(link: int) -> str:
    # JOINTS = [
    #     "robot/ur10e",
    #     "base/shoulder_pan_joint",
    #     "shoulder_link/shoulder_lift_joint",
    #     "upper_arm_link/elbow_joint",
    #     "forearm_link/wrist_1_joint",
    #     "wrist_1_link/wrist_2_joint",
    #     "wrist_2_link/wrist_3_joint",
    # ]

    JOINTS = [
        "ur10e/base",
        "shoulder_link/shoulder_pan_joint",
        "upper_arm_link/shoulder_lift_joint",
        "forearm_link/elbow_joint",
        "wrist_1_link/wrist_1_joint",
        "wrist_2_link/wrist_2_joint",
        "wrist_3_link/wrist_3_joint",
    ]
    assert link < len(JOINTS), (
        f"Link {link} is out of range. Max link is {len(JOINTS) - 1}"
    )

    return "/".join(JOINTS[: link + 2])


def log_angle_rot(
    entity_to_transform: dict[str, tuple[np.ndarray, np.ndarray]],
    link: int,
    angle_rad: float,
) -> None:
    """Logs an angle for the franka panda robot"""
    entity_path = path_to_link(link)

    start_translation, start_rotation_quat, axis = entity_to_transform[entity_path]
    start_rotation_mat = Rotation.from_quat(start_rotation_quat).as_matrix()

    # All angles describe rotations around the transformed z-axis.
    vec = np.array(axis * angle_rad)

    rot = Rotation.from_rotvec(vec).as_matrix()
    rotation_mat = start_rotation_mat @ rot

    rr.log(
        entity_path, rr.Transform3D(translation=start_translation, mat3x3=rotation_mat)
    )


def link_to_world_transform(
    entity_to_transform: dict[str, tuple[np.ndarray, np.ndarray]],
    joint_angles: list[float],
    link: int,
) -> np.ndarray:
    tot_transform = np.eye(4)
    for i in range(1, link + 1):
        entity_path = path_to_link(i)

        start_translation, start_rotation_quat = entity_to_transform[entity_path]
        start_rotation_mat = Rotation.from_quat(start_rotation_quat).as_matrix()

        if i - 1 >= len(joint_angles):
            angle_rad = 0
        else:
            angle_rad = joint_angles[i - 1]
        vec = np.array(np.array([0, 0, 1]) * angle_rad)

        rot = Rotation.from_rotvec(vec).as_matrix()
        rotation_mat = start_rotation_mat @ rot

        transform = np.eye(4)
        transform[:3, :3] = rotation_mat
        transform[:3, 3] = start_translation
        tot_transform = tot_transform @ transform

    return tot_transform