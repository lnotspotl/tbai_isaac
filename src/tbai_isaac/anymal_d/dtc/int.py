import ig_interface


def get_interface(num_envs, torch, rviz_visualize=False):
    task_file = "/home/kuba/fun/ocs2_project/src/ocs2_fun/dependencies/ocs2/ocs2_robotic_examples/ocs2_legged_robot/config/mpc/task.info"
    urdf_file = (
        "/home/kuba/fun/ocs2_project/src/ocs2_fun/dependencies/ocs2_robotic_assets/resources/anymal_d/urdf/anymal.urdf"
    )
    reference_file = "/home/kuba/fun/ocs2_project/src/ocs2_fun/dependencies/ocs2/ocs2_robotic_examples/ocs2_legged_robot/config/command/reference.info"
    gait_file = "/home/kuba/fun/ocs2_project/src/ocs2_fun/dependencies/ocs2/ocs2_robotic_examples/ocs2_legged_robot/config/command/gait.info"
    gait = "trot"
    num_threads = 1

    ig = ig_interface.TbaiIsaacGymInterface(
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
