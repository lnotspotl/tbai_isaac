#!/usr/bin/env python3

import time

import ig_interface
import torch

task_file = "/home/kuba/fun/ocs2_project/src/ocs2_fun/dependencies/ocs2/ocs2_robotic_examples/ocs2_legged_robot/config/mpc/task.info"
urdf_file = (
    "/home/kuba/fun/ocs2_project/src/ocs2_fun/dependencies/ocs2_robotic_assets/resources/anymal_d/urdf/anymal.urdf"
)
reference_file = "/home/kuba/fun/ocs2_project/src/ocs2_fun/dependencies/ocs2/ocs2_robotic_examples/ocs2_legged_robot/config/command/reference.info"
gait_file = "/home/kuba/fun/ocs2_project/src/ocs2_fun/dependencies/ocs2/ocs2_robotic_examples/ocs2_legged_robot/config/command/gait.info"
gait = "trot"
num_envs = 1024
num_threads = 5
device = torch.device("cpu")

interface = ig_interface.LeggedRobotIsaacGymInterface(
    task_file, urdf_file, reference_file, gait_file, gait, num_envs, num_threads, device
)
interface.reset_all_solvers(0.0)
interface.update_all_states(torch.zeros(num_envs, 12 + 12))
# interface.update_states(torch.randn(num_envs-1, 12+12), torch.arange(0, num_envs-1))

t1 = time.perf_counter()
interface.optimize_trajectories(0.0)
t2 = time.perf_counter()
print(f"Optimizing trajectories took: {(t2-t1) * 1e3:.5f} ms")

t1 = time.perf_counter()
interface.optimize_trajectories(0.0, torch.arange(0, num_envs))
t2 = time.perf_counter()
print(f"Optimizing trajectories took: {(t2-t1) * 1e3:.5f} ms")

t1 = time.perf_counter()
interface.optimize_trajectories(0.0, torch.arange(0, num_envs))
interface.update_desired_contacts(0.5, torch.arange(0, num_envs))
interface.update_time_left_in_phase(0.5, torch.arange(0, num_envs))
interface.update_desired_joint_angles(0.5, torch.arange(0, num_envs))
t2 = time.perf_counter()
print(f"Optimizing trajectories took: {(t2-t1) * 1e3:.5f} ms")

interface.update_optimized_states(0.5)

optimized_states = interface.get_optimized_states()
optimized_states[:, :] = 3.14
optimized_states = interface.get_optimized_states()
print(optimized_states)
interface.update_optimized_states(0.5)
print(optimized_states)


interface.set_current_command(torch.randn(num_envs, 3), torch.arange(0, num_envs))

print("All solvers are reset!")
