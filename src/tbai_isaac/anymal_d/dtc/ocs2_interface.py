import os
import tbai_ocs2_interface

from tbai_isaac.anymal_d.info import URDF_PATH, MPC_TASK_FILE, MPC_REFERENCE_FILE, MPC_GAIT_FILE


def get_interface(num_envs, torch, rviz_visualize=False, num_threads=5):
    task_file = MPC_TASK_FILE
    urdf_file = URDF_PATH
    reference_file = MPC_REFERENCE_FILE
    gait_file = MPC_GAIT_FILE
    gait = "trot"

    ig = tbai_ocs2_interface.TbaiIsaacGymInterface(
        task_file,
        urdf_file,
        reference_file,
        gait_file,
        gait,
        num_envs,
        num_threads,
        torch.device("cuda"),
        rviz_visualize,
    )

    return ig
