from ocs2_interface import get_interface
import os
from collections import OrderedDict

import numpy as np
import torch
import pytorch3d.transforms

from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *
from tbai_isaac.anymal_d.common.central_pattern_generator import CentralPatternGenerator
from tbai_isaac.anymal_d.common.inverse_kinematics import AnymalInverseKinematics
from tbai_isaac.anymal_d.info import URDF_FOLDER
from tbai_isaac.common.base_env import BaseEnv
from tbai_isaac.common.math import quat_apply_yaw, quat_apply_yaw_inverse, wrap_to_pi
from tbai_isaac.common.observation import ObservationManager
from tbai_isaac.common.terrain import Terrain
import tbai_isaac.anymal_d.dtc.config as ac
from tbai_isaac.common.config import select


class LeggedRobot(BaseEnv):
    def __init__(self, yaml_cfg, headless, ocs2_interface_threads):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """

        self.anymal_config = yaml_cfg

        self.asset_config = ac.get_asset_config(self.anymal_config)
        self.domain_randomization_config = ac.get_randomization_config(self.anymal_config)
        self.terrain_config = ac.get_terrain_config(self.anymal_config)
        self.command_config = ac.get_command_config(self.anymal_config)
        self.env_config = ac.get_env_config(self.anymal_config)
        self.control_config = ac.get_control_config(self.anymal_config)
        self.viewer_config = ac.get_viewer_config(self.anymal_config)
        self.normalization_config = ac.get_normalization_config(self.anymal_config)
        self.rewards_config = ac.get_rewards_config(self.anymal_config)
        self.init_state_config = ac.get_init_state_config(self.anymal_config)
        self.noise_config = ac.get_noise_config(self.anymal_config)
        self.sim_config = ac.get_sim_config(self.anymal_config)

        self.add_height_samples = True

        self.current_time = 0.0

        self.sampling_positions_local = None 

        self.step_counter = 0
        self.noise_scale = float(1.0)

        self.observation_managers: OrderedDict[str, ObservationManager] = OrderedDict()

        self.clip_actions = select(self.anymal_config, "environment.normalization.clip_actions")

        self.last_base_lin_vel = None
        self.last_base_ang_vel = None

        self.sim_params = ac.get_sim_params(self.anymal_config)
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg()

        print("HER1")
        num_obs = self.env_config.num_observations
        num_privileged_obs = self.env_config.num_privileged_observations
        if num_privileged_obs is None:
            num_privileged_obs = 0  
        num_actions = self.env_config.num_actions
        device = "cuda"
        num_envs = self.env_config.num_envs

        print("HER2")

        super().__init__(
            num_obs=num_obs,
            num_privileged_obs=num_privileged_obs,
            num_actions=num_actions,
            device=device,
            headless=headless,
            num_envs=num_envs,
            sim_params=self.sim_params,
            physics_engine=gymapi.SIM_PHYSX,
            sim_device="cuda",
        )
        print("HER3")

        print("Creating interface")

        self.rviz_visualize = False
        self.tbai_ocs2_interface = get_interface(
            self.num_envs, torch, self.rviz_visualize, num_threads=ocs2_interface_threads
        )

        self.actor_noise = False
        self.actor_noise = float(self.actor_noise)
        print("Interface created")

        self.tbai_ocs2_interface.reset_all_solvers(self.current_time)
        print("Solvers reset")

        print("States updated")
        self.time_till_optimization = self.tbai_ocs2_interface.updated_in_seconds()

        print(self.privileged_obs_buf)
        print("BUUG" * 10)

        if self.num_privileged_obs == 0:
            self.num_privileged_obs = None

        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        if not self.headless:
            self.set_camera(self.viewer_config.pos, self.viewer_config.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        period = 0.60
        initial_offsets = torch.Tensor([0.0, period / 2, period / 2, 0.0])  # LF, LH, RF, RH

        self.cpg = CentralPatternGenerator(period, initial_offsets, n_envs=self.num_envs, device=self.device)

        assert self.asset_config.name == "anymal_d", f"Unknown robot name: {self.asset_config.name}"
        self.aik = AnymalInverseKinematics(device=self.device)
        self.aik.tt = None

        self.iter = 0.0
        self.ck = 0.0

        self.tracking_reward_ready = torch.ones(self.num_envs, 4, dtype=torch.float32, device=self.device)

        self.aa = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.bb = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.reference_angle = torch.zeros(self.num_envs, device=self.device)

    def add_observation_manager(self, name: str, observation_manager: ObservationManager):
        if name in self.observation_managers:
            raise ValueError(f"Observation manager {name} already exists")
        self.observation_managers[name] = observation_manager
        print("Added observation manager <<===")
        print(self.observation_managers)

    def get_observation_manager(self, name: str) -> ObservationManager:
        if name not in self.observation_managers:
            raise ValueError(f"Observation manager {name} not found")
        return self.observation_managers[name]

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        self.actions = torch.clip(actions, -self.clip_actions, self.clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.control_config.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.normalization_config.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def generate_pts(self, optimized_indices):
        n_indices = optimized_indices.shape[0]

        length_x = 2.5
        length_y = 2.5
        resolution = 0.1
        Nx = int(length_x / resolution)
        Ny = int(length_y / resolution)

        if self.sampling_positions_local is None:
            x_coords = torch.linspace(-1, 1, Nx) * length_x / 2
            y_coords = torch.linspace(-1, 1, Ny) * length_y / 2
            self.sampling_positions_local = torch.zeros((Nx, Ny, 3), device=self.device)
            # Reversed because we want to go from positive to negative values
            for i, x in enumerate(reversed(x_coords)):
                for j, y in enumerate(reversed(y_coords)):
                    ix = y 
                    iy = x
                    self.sampling_positions_local[i, j, 0] = ix
                    self.sampling_positions_local[i, j, 1] = iy
                    self.sampling_positions_local[i, j, 2] = 0.0
                
            self.sampling_positions_local = self.sampling_positions_local.view(1, -1, 3)

        base_coords = self.root_states[optimized_indices, 0:3].view(n_indices, 1, 3)
        sampling_positions = base_coords + self.sampling_positions_local
        sampling_positions[:, :, 2] = 0.0
        if self.terrain_config.mesh_type != "plane":


            points = sampling_positions.clone() + self.terrain_config.border_size
            points = (points / self.terrain_config.horizontal_scale).long()
            px = points[:, :, 0].view(-1)
            py = points[:, :, 1].view(-1)
            px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
            py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
            heights1 = self.height_samples[px, py] * self.terrain_config.vertical_scale
            sampling_positions[:, :, 2] = heights1.view(n_indices, Nx * Ny)

        return sampling_positions, length_x, length_y, resolution

    def generate_flattened_maps(self, optimized_indices):
        sampling_positions, length_x, length_y, resolution = self.generate_pts(optimized_indices)
        return sampling_positions[:, :, 2].cpu(), length_x, length_y, resolution
    
    def update_flattened_maps(self, optimized_indices):
        # Update maps prior to optimization
        flattened_maps, length_x, length_y, resolution = self.generate_flattened_maps(optimized_indices)
        flattened_maps = flattened_maps.cpu()
        x_coords = self.root_states[optimized_indices, 0].cpu()
        y_coords = self.root_states[optimized_indices, 1].cpu()
        optimized_indices_cpu = optimized_indices.cpu()
        self.tbai_ocs2_interface.set_maps_from_flattened(flattened_maps, length_x, length_y, resolution, x_coords, y_coords, optimized_indices_cpu)


    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.last_base_lin_vel = self.root_states[:, 7:10]
        self.last_base_ang_vel = self.root_states[:, 10:13]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.contacts[:] = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) >= 0.3

        self.shank_contacts[:] = torch.norm(self.contact_forces[:, self.shank_indices, :3], dim=2) >= 0.3
        self.thigh_contacts[:] = torch.norm(self.contact_forces[:, self.thigh_indices, :3], dim=2) >= 0.3

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.current_time += self.dt

        self.time_till_optimization -= self.dt

        optimized_indices = (self.time_till_optimization <= 0.0).nonzero(as_tuple=False).flatten()
        self.tbai_ocs2_interface.update_states_perceptive(self.get_ocs2_state_perceptive(optimized_indices), optimized_indices)
        self.update_flattened_maps(optimized_indices)
        self.tbai_ocs2_interface.optimize_trajectories(self.current_time, optimized_indices)

        mask = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        mask[optimized_indices] = True
        mask = torch.logical_and(mask, torch.logical_not(self.tbai_ocs2_interface.get_desired_contacts()))

        self.aa[mask] = True
        self.aa[optimized_indices][
            torch.logical_not(self.tbai_ocs2_interface.get_desired_contacts())[optimized_indices, :]
        ] = True
        self.bb[:, :] = torch.logical_and(
            torch.logical_not(self.tbai_ocs2_interface.get_desired_contacts()),
            torch.logical_and(
                self.tbai_ocs2_interface.get_time_left_in_phase() <= 0.025,
                self.tbai_ocs2_interface.get_time_left_in_phase() >= 0.0,
            ),
        ).float()

        self.tbai_ocs2_interface.update_optimized_states(self.current_time) # TODO: This is of no use
        self.tbai_ocs2_interface.update_desired_contacts(self.current_time, torch.arange(0, self.num_envs))
        self.tbai_ocs2_interface.update_time_left_in_phase(self.current_time, torch.arange(0, self.num_envs))
        self.tbai_ocs2_interface.update_desired_joint_angles(self.current_time, torch.arange(0, self.num_envs))
        self.tbai_ocs2_interface.update_current_desired_joint_angles(
            self.current_time + self.dt, torch.arange(0, self.num_envs)
        )

        # New functions
        self.tbai_ocs2_interface.update_desired_base(self.current_time + self.dt, torch.arange(0, self.num_envs))
        self.tbai_ocs2_interface.move_desired_base_to_gpu()

        if self.control_config.control_type == "CPG_WBC":
            self.tbai_ocs2_interface.update_desired_foot_positions_and_velocities(
                self.current_time, torch.arange(0, self.num_envs)
            )

        # Update last action history buffer
        self.compute_reward()
        self.dof_action_history[self.dof_action_idx, :] = self.actions[:]
        self.dof_action_idx = (self.dof_action_idx + 1) % self.dof_action_history_length
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        if self.rviz_visualize:
            env_id = 0
            ocs2_state = self.get_ocs2_state_perceptive(torch.arange(env_id, env_id + 2))[0]
            current_time = self.current_time
            current_obs = self.obs_buf[env_id]

            self.update_flattened_maps(torch.arange(env_id, env_id + 2))
            self.tbai_ocs2_interface.visualize(current_time, ocs2_state, env_id, current_obs)

        # Compute observation for each observation manager
        for observation_manager in self.observation_managers.values():
            observation_manager.compute()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        # self.last_root_vel[:] = self.root_states[:, 7:13]

        # update DOF residual history buffer
        self.dof_residuals_history[self.dof_residual_idx, :] = self.dof_pos[:] - (
            self.default_dof_pos if self.aik.tt is None else self.aik.tt
        )
        self.dof_residual_idx = (self.dof_residual_idx + 1) % self.dof_residuals_history_length

        # Update DOF velocity history buffer
        self.dof_velocity_history[self.dof_velocity_idx, :] = self.dof_vel[:]
        self.dof_velocity_idx = (self.dof_velocity_idx + 1) % self.dof_velocity_history_length

        self.update_curriculum()

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        self.cpg.step(self.dt)

    def check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.terrain_config.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.command_config.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        self.reference_angle[env_ids] = 0.0

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._reset_cpg(env_ids)

        self._resample_commands(env_ids)

        self.tbai_ocs2_interface.reset_solvers(self.current_time, env_ids)
        self.tbai_ocs2_interface.set_current_command(self.commands[env_ids], env_ids)
        self.tbai_ocs2_interface.update_states_perceptive(self.get_ocs2_state_perceptive(env_ids), env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        # log additional curriculum info
        if self.terrain_config.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.command_config.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges.lin_vel_x[1]
        # send timeout info to the algorithm
        if self.env_config.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.rewards_config.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def get_heights_observation(self, measured_heights):
        heights = (
            torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - measured_heights, -1, 1.0)
            * self.obs_scales.height_measurements
        )
        return heights

    def get_heights_observation_with_noise(self, noise_generator):
        assert noise_generator.n_envs == self.num_envs
        measured_heights, pts = self._get_heights(noise_generator=noise_generator)
        heights = self.get_heights_observation(measured_heights)
        return heights, pts

    def generate_uniform(self, size, low, high, device):
        diff = high - low
        out = torch.randn(*size) * diff + low
        return out.to(device)
    

    def sample_at_position(self, x, y):
        if self.terrain_config.mesh_type == "plane":
            return 0.0
        xp = x + self.terrain_config.border_size
        yp = y + self.terrain_config.border_size
        xp = xp / self.terrain_config.horizontal_scale
        yp = yp / self.terrain_config.horizontal_scale
        z = self.height_samples[xp.long(), yp.long()] * self.terrain_config.vertical_scale
        return z

    def generate_height_samples(self):
        diffs = list()
        foot_positions = [
            self.lf_foot_position,
            self.rf_foot_position,
            self.lh_foot_position,
            self.rh_foot_position,
        ]
        desired_foot_positions = [
            self.tbai_ocs2_interface.get_planar_footholds()[:, 0:2],
            self.tbai_ocs2_interface.get_planar_footholds()[:, 2:4],
            self.tbai_ocs2_interface.get_planar_footholds()[:, 4:6],
            self.tbai_ocs2_interface.get_planar_footholds()[:, 6:8],
        ]

        for fp, dfp in zip(foot_positions, desired_foot_positions):
            dx = fp[:, 0] - dfp[:, 0]
            dy = fp[:, 1] - dfp[:, 1]

            a = torch.linspace(0, 1, 10).to(self.device)
            a = a.view(1, -1)
            x = fp[:, 0].view(-1, 1) + a * dx.view(-1, 1)
            y = fp[:, 1].view(-1, 1) + a * dy.view(-1, 1)

            n_envs = x.shape[0]
            n_samples = x.shape[1]

            heights = self.sample_at_position(x.view(-1), y.view(-1)).view(n_envs, n_samples)
            zdiffs = heights - fp[:, 2].view(-1, 1)
            diffs.append(zdiffs)
        
        return torch.stack(diffs, dim=1)

    def compute_observations(self):
        d_pos = self.default_dof_pos if self.aik.tt is None else self.aik.tt

        # Base linear velocity
        lin_vel_noise = self.generate_uniform(size=(self.num_envs, 3), low=-0.12, high=0.12, device=self.device) * self.obs_scales.lin_vel
        lin_vel = lin_vel = self.base_lin_vel * self.obs_scales.lin_vel

        # print(f"Base linear velocity: {self.base_lin_vel}")

        # Base angular velocity
        ang_vel_noise = self.generate_uniform(size=(self.num_envs, 3), low=-0.22, high=0.22, device=self.device) * self.obs_scales.ang_vel
        ang_vel = self.base_ang_vel * self.obs_scales.ang_vel
        # print(f"Base angular velocity: {self.base_ang_vel}")

        # Projected gravity
        gravity_noise = self.generate_uniform(size=(self.num_envs, 3), low=-0.06, high=0.06, device=self.device)
        projected_gravity = self.projected_gravity
        # print(f"Projected gravity: {projected_gravity}")

        # Joint positions
        dof_pos_noise = self.generate_uniform(size=(self.num_envs, 12), low=-0.1, high=0.1, device=self.device) * self.obs_scales.dof_pos
        dof_pos = (self.dof_pos - d_pos) * self.obs_scales.dof_pos
        # print(f"Joint positions: {dof_pos}")

        # Joint velocities
        dof_vel_noise = self.generate_uniform(size=(self.num_envs, 12), low=-1.5, high=1.5, device=self.device) * self.obs_scales.dof_vel
        dof_vel = self.dof_vel * self.obs_scales.dof_vel
        # print(f"Joint velocities: {dof_vel}")

        # Past actions
        actions = self.actions
        # print(f"Past actions: {actions}")

        # Planar footholds
        planar_footholds_noise = (
            self.generate_uniform(size=(self.num_envs, 4 * 2), low=-0.06, high=0.06, device=self.device)
        )
        planar_footholds = torch.zeros((self.num_envs, 4 * 2), device=self.device)

        for i, leg_pos in enumerate([self.lf_foot_position, self.rf_foot_position, self.lh_foot_position, self.rh_foot_position]):
            temp = torch.zeros((self.num_envs, 3), device=self.device)
            temp[:, 0:2] = self.tbai_ocs2_interface.get_planar_footholds()[:, i * 2 : i * 2 + 2]
            temp[:, 0:2] -= leg_pos[:, 0:2]
            temp = quat_apply(quat_conjugate(self.base_quat), temp)
            planar_footholds[:, i * 2 : i * 2 + 2] += temp[:, :2]

        # print(f"Planar footholds: {planar_footholds}")

        # Desired joint positions
        desired_joint_angles = self.tbai_ocs2_interface.get_desired_joint_positions() - d_pos
        # print(f"Desired joint positions: {desired_joint_angles}")

        # Current desired joint positions
        current_desired_joint_angles = self.tbai_ocs2_interface.get_current_desired_joint_positions() - d_pos

        # Desired contact state
        desired_contacts = self.tbai_ocs2_interface.get_desired_contacts()
        # print(f"Desired contact state: {desired_contacts}")

        # Time left in phase
        time_left_in_phase = self.tbai_ocs2_interface.get_time_left_in_phase()
        # print(f"Time left in phase: {time_left_in_phase}")

        # Command
        command = self.commands[:, :3] * self.commands_scale
        # print(f"Command: {command}")

        # Desired base position
        desired_base_pos = quat_apply(
            quat_conjugate(self.base_quat), self.tbai_ocs2_interface.get_desired_base_positions() - self.root_states[:, 0:3]
        )
        # print(f"Desired base position: {desired_base_pos}")

        base_orientation = self.root_states[:, 3:7]
        base_orientation_desired = self.tbai_ocs2_interface.get_desired_base_orientations().to(self.device)

        orientation_diff = quat_mul(base_orientation_desired, quat_conjugate(base_orientation))

        # Desired base linear velocity
        desired_base_lin_vel = self.tbai_ocs2_interface.get_desired_base_linear_velocities()
        desired_base_lin_vel = quat_apply(quat_conjugate(self.base_quat), desired_base_lin_vel)

        # Desired base angular velocity
        desired_base_ang_vel = self.tbai_ocs2_interface.get_desired_base_angular_velocities()
        desired_base_ang_vel = quat_apply(quat_conjugate(self.base_quat), desired_base_ang_vel)

        #### TODO: Add
        desid_base_lin_acc = self.tbai_ocs2_interface.get_desired_base_linear_accelerations()
        desid_base_lin_acc = quat_apply(quat_conjugate(self.base_quat), desid_base_lin_acc)

        desid_base_ang_acc = self.tbai_ocs2_interface.get_desired_base_angular_accelerations()
        desid_base_ang_acc = quat_apply(quat_conjugate(self.base_quat), desid_base_ang_acc)

        phases = self.tbai_ocs2_interface.get_bobnet_phases(self.current_time, torch.arange(0, self.num_envs))

        if self.control_config.control_type == "CPG" or self.control_config.control_type == "CPG_WBC":
            # LF, RF, LH, RH
            # flip to LF, LH, RF, RH
            phases = phases[:, [0, 2, 1, 3]]
        cpg_obs = self.cpg.get_observation(phases)

        self.obs_buf = torch.cat(
            [
                lin_vel + lin_vel_noise * self.noise_scale,
                ang_vel + ang_vel_noise * self.noise_scale,
                projected_gravity + gravity_noise * self.noise_scale,
                command,  # No noise
                dof_pos + dof_pos_noise * self.noise_scale,
                dof_vel + dof_vel_noise * self.noise_scale,
                actions, # No noise
                planar_footholds + planar_footholds_noise * self.noise_scale,
                desired_joint_angles,
                current_desired_joint_angles,
                desired_contacts,
                time_left_in_phase,
                desired_base_pos,
                orientation_diff,
                desired_base_lin_vel,
                desired_base_ang_vel,
                desid_base_lin_acc,
                desid_base_ang_acc,
                cpg_obs,
            ],
            dim=-1,
        )

        if self.add_height_samples:
            height_samples_noise = self.generate_uniform(size=(self.num_envs, 40), low=-0.06, high=0.06, device=self.device) * self.obs_scales.height_measurements
            height_samples = self.generate_height_samples().view(self.num_envs, 40) * self.obs_scales.height_measurements

        if self.control_config.control_type == "CPG_WBC":
            # LF foot position error
            lf_foot_position_noise = self.generate_uniform(size=(self.num_envs, 3), low=-0.06, high=0.06, device=self.device)
            lf_foot_pos_error = self.lf_foot_position - self.tbai_ocs2_interface.get_desired_foot_positions()[:, :3]
            lf_foot_pos_error = quat_apply(quat_conjugate(self.base_quat), lf_foot_pos_error)

            # RF foot position error
            rf_foot_position_noise = self.generate_uniform(size=(self.num_envs, 3), low=-0.06, high=0.06, device=self.device)
            rf_foot_pos_error = self.rf_foot_position - self.tbai_ocs2_interface.get_desired_foot_positions()[:, 3:6] 
            rf_foot_pos_error = quat_apply(quat_conjugate(self.base_quat), rf_foot_pos_error)

            # LH foot position error
            lh_foot_position_noise = self.generate_uniform(size=(self.num_envs, 3), low=-0.06, high=0.06, device=self.device)
            lh_foot_pos_error = self.lh_foot_position - self.tbai_ocs2_interface.get_desired_foot_positions()[:, 6:9] 
            lh_foot_pos_error = quat_apply(quat_conjugate(self.base_quat), lh_foot_pos_error)

            # RH foot position error
            rh_foot_position_noise = self.generate_uniform(size=(self.num_envs, 3), low=-0.06, high=0.06, device=self.device)
            rh_foot_pos_error = self.rh_foot_position - self.tbai_ocs2_interface.get_desired_foot_positions()[:, 9:12] 

            # LF foot velocity error
            lf_foot_velocity_noise = self.generate_uniform(size=(self.num_envs, 3), low=-0.20, high=0.20, device=self.device)
            lf_foot_vel_error = self.lf_foot_velocity - self.tbai_ocs2_interface.get_desired_foot_velocities()[:, :3] 
            lf_foot_vel_error = quat_apply(quat_conjugate(self.base_quat), lf_foot_vel_error)

            # RF foot velocity error
            rf_foot_velocity_noise = self.generate_uniform(size=(self.num_envs, 3), low=-0.20, high=0.20, device=self.device)
            rf_foot_vel_error = self.rf_foot_velocity - self.tbai_ocs2_interface.get_desired_foot_velocities()[:, 3:6] 
            rf_foot_vel_error = quat_apply(quat_conjugate(self.base_quat), rf_foot_vel_error)

            # LH foot velocity error
            lh_foot_velocity_noise = self.generate_uniform(size=(self.num_envs, 3), low=-0.20, high=0.20, device=self.device)
            lh_foot_vel_error = self.lh_foot_velocity - self.tbai_ocs2_interface.get_desired_foot_velocities()[:, 6:9]
            lh_foot_vel_error = quat_apply(quat_conjugate(self.base_quat), lh_foot_vel_error)

            # RH foot velocity error
            rh_foot_velocity_noise = self.generate_uniform(size=(self.num_envs, 3), low=-0.20, high=0.20, device=self.device)
            rh_foot_vel_error = self.rh_foot_velocity - self.tbai_ocs2_interface.get_desired_foot_velocities()[:, 9:12]
            rh_foot_vel_error = quat_apply(quat_conjugate(self.base_quat), rh_foot_vel_error)

            self.obs_buf = torch.cat(
                [
                    self.obs_buf,
                    lf_foot_pos_error + lf_foot_position_noise * self.noise_scale,
                    lh_foot_pos_error + lh_foot_position_noise * self.noise_scale,
                    rf_foot_pos_error + rf_foot_position_noise * self.noise_scale,
                    rh_foot_pos_error + rh_foot_position_noise * self.noise_scale,
                    lf_foot_vel_error + lf_foot_velocity_noise * self.noise_scale,
                    lh_foot_vel_error + lh_foot_velocity_noise * self.noise_scale,
                    rf_foot_vel_error + rf_foot_velocity_noise * self.noise_scale,
                    rh_foot_vel_error + rh_foot_velocity_noise * self.noise_scale,
                ], dim=-1
            )

            if self.add_height_samples:
                self.obs_buf = torch.cat([self.obs_buf, height_samples + height_samples_noise * self.noise_scale], dim=-1)

        # Privileged observation == without noise
        self.privileged_obs_buf = torch.cat(
            [
                lin_vel,
                ang_vel,
                projected_gravity,
                command,
                dof_pos,
                dof_vel,
                actions,
                planar_footholds,
                desired_joint_angles,
                current_desired_joint_angles,
                desired_contacts,
                time_left_in_phase,
                desired_base_pos,
                orientation_diff,
                desired_base_lin_vel,
                desired_base_ang_vel,
                desid_base_lin_acc,
                desid_base_ang_acc,
                cpg_obs,
            ],
            dim=-1,
        )

        if self.control_config.control_type == "CPG_WBC":
            self.privileged_obs_buf = torch.cat(
                [
                    self.privileged_obs_buf,
                    lf_foot_pos_error,
                    lh_foot_pos_error,
                    rf_foot_pos_error,
                    rh_foot_pos_error,
                    lf_foot_vel_error,
                    lh_foot_vel_error,
                    rf_foot_vel_error,
                    rh_foot_vel_error,
                ], dim=-1
            )

        if self.add_height_samples:
            self.privileged_obs_buf = torch.cat([self.privileged_obs_buf, height_samples], dim=-1)

        for i in range(4):
            contact_force_world = self.contact_forces[:, self.feet_indices[i], :]
            contact_force_body = (
                quat_rotate_inverse(self.base_quat.repeat(1, self.num_envs), contact_force_world)
                * self.obs_scales.contact_forces
            )
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, contact_force_body), dim=-1)

        self.step_counter += 1

    def create_sim(self):
        """Creates simulation, terrain and evironments"""
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params
        )
        mesh_type = self.terrain_config.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = Terrain(self.terrain_config, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
            print("Created trimesh terrain")
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """Set camera position and direction"""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """

        if self.domain_randomization_config.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.domain_randomization_config.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(
                    friction_range[0], friction_range[1], (num_buckets, 1), device="cpu"
                )
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        else:
            if env_id == 0:
                self.friction_coeffs = torch.ones(self.num_envs).unsqueeze(1).unsqueeze(1)

        return props

    def _process_dof_props(self, props, env_id):
        """Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.rewards_config.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.rewards_config.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.domain_randomization_config.randomize_base_mass:
            rng = self.domain_randomization_config.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (
            (self.episode_length_buf % int(self.command_config.resampling_time / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)
        self.tbai_ocs2_interface.set_current_command(self.commands[env_ids], env_ids)
        self.tbai_ocs2_interface.update_states_perceptive(self.get_ocs2_state_perceptive(env_ids), env_ids)
        self.update_flattened_maps(env_ids)
        self.tbai_ocs2_interface.optimize_trajectories(self.current_time, env_ids)
        if self.command_config.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

        if self.terrain_config.measure_heights:
            self.measured_heights, _ = self._get_heights()
        if self.domain_randomization_config.push_robots and (
            self.common_step_counter % self.domain_randomization_config.push_interval == 0
        ):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges.lin_vel_x[0],
            self.command_ranges.lin_vel_x[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges.lin_vel_y[0],
            self.command_ranges.lin_vel_y[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        if self.command_config.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges.heading[0],
                self.command_ranges.heading[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges.ang_vel_yaw[0],
                self.command_ranges.ang_vel_yaw[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.control_config.action_scale
        control_type = self.control_config.control_type
        if control_type == "P":
            torques = (
                self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
            )

            # torques = (
            #     self.p_gains * (actions_scaled + self.tbai_ocs2_interface.get_current_desired_joint_positions() - self.dof_pos) - self.d_gains * self.dof_vel
            # )
        elif control_type == "V":
            torques = (
                self.p_gains * (actions_scaled - self.dof_vel)
                - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_config.dt
            )
        elif control_type == "T":
            torques = actions_scaled
        elif control_type == "CPG":
            dof_residuals = actions_scaled
            phases = self.tbai_ocs2_interface.get_bobnet_phases(self.current_time, torch.arange(0, self.num_envs))
            # LF, RF, LH, RH
            # flip to LF, LH, RF, RH
            phases = phases[:, [0, 2, 1, 3]]
            joint_angles = self.aik.compute_ik(self.cpg.leg_heights(phases))
            self.aik.tt = joint_angles
            torques = (
                self.p_gains[:12] * (dof_residuals + joint_angles - self.dof_pos) - self.d_gains[:12] * self.dof_vel
            )
        elif control_type == "CPG_WBC":
            torques = (
                self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
            )
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
            0.5, 1.5, (len(env_ids), self.num_dof), device=self.device
        )
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _reset_cpg(self, env_ids):
        self.cpg.reset(env_ids)

    def _reset_root_states(self, env_ids):
        """Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        max_vel = self.domain_randomization_config.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )  # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

        env_ids = torch.arange(0, self.num_envs)
        self.tbai_ocs2_interface.set_current_command(self.commands[env_ids], env_ids)
        self.update_flattened_maps(env_ids)
        self.tbai_ocs2_interface.optimize_trajectories(self.current_time, env_ids)

    def _update_terrain_curriculum(self, env_ids):
        """Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 3
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (
            distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        ) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if (
            torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length
            > 0.5 * self.reward_scales["tracking_lin_vel"]
        ):
            self.command_ranges.lin_vel_x[0] = float(
                np.clip(self.command_ranges.lin_vel_x[0] - 0.5, -self.command_config.max_curriculum, 0.0)
            )
            self.command_ranges.lin_vel_x[1] = float(
                np.clip(self.command_ranges.lin_vel_x[1] + 0.5, 0.0, self.command_config.max_curriculum)
            )

    def _get_noise_scale_vec(self):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        # print("Generating observation noise")
        # print(self.obs_buf.shape)
        self.add_noise = self.noise_config.add_noise
        noise_scales = self.noise_config.noise_scales
        noise_level = self.noise_config.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.0  # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0.0  # previous actions
        if self.terrain_config.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)  # kuba

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)  # kuba

        self.lf_foot_position = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, -1)[
            :, self.name2idx["LF_FOOT"], 0:3
        ]
        self.lh_foot_position = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, -1)[
            :, self.name2idx["LH_FOOT"], 0:3
        ]
        self.rf_foot_position = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, -1)[
            :, self.name2idx["RF_FOOT"], 0:3
        ]
        self.rh_foot_position = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, -1)[
            :, self.name2idx["RH_FOOT"], 0:3
        ]

        self.lf_foot_velocity = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, -1)[
            :, self.name2idx["LF_FOOT"], 7:10
        ]
        self.lh_foot_velocity = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, -1)[
            :, self.name2idx["LH_FOOT"], 7:10
        ]
        self.rf_foot_velocity = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, -1)[
            :, self.name2idx["RF_FOOT"], 7:10
        ]
        self.rh_foot_velocity = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, -1)[
            :, self.name2idx["RH_FOOT"], 7:10
        ]

        # contacts
        self.contacts = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
        self.shank_contacts = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
        self.thigh_contacts = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # DOF residuals history
        DOF_RESIDUAL_HISTORY_LENGTH = 3
        DOF_RESIDUAL_BUFFER_SIZE = (DOF_RESIDUAL_HISTORY_LENGTH, self.num_envs, 12)
        self.dof_residuals_history = torch.zeros(
            DOF_RESIDUAL_BUFFER_SIZE, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.dof_residuals_history_length = DOF_RESIDUAL_HISTORY_LENGTH
        self.dof_residual_idx = 0

        # DOF velocity history
        DOF_VELOCITY_HISTORY_LENGTH = 2
        DOF_VELOCITY_BUFFER_SIZE = (DOF_VELOCITY_HISTORY_LENGTH, self.num_envs, 12)
        self.dof_velocity_history = torch.zeros(
            DOF_VELOCITY_BUFFER_SIZE, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.dof_velocity_history_length = DOF_VELOCITY_HISTORY_LENGTH
        self.dof_velocity_idx = 0

        # Action history
        DOF_ACTION_HISTORY_LENGTH = 2
        CRITIC_OUT = 12
        DOF_ACTION_BUFFER_SIZE = (DOF_ACTION_HISTORY_LENGTH, self.num_envs, CRITIC_OUT)
        self.dof_action_history = torch.zeros(
            DOF_ACTION_BUFFER_SIZE, dtype=torch.float, device=self.device, requires_grad=False
        )

        #

        print(self.feet_indices)
        print(self.thigh_indices)
        print(self.shank_indices)

        self.dof_action_history_length = DOF_ACTION_HISTORY_LENGTH
        self.dof_action_idx = 0

        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        # self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(
            self.num_envs, self.command_config.num_commands, dtype=torch.float, device=self.device, requires_grad=False
        )  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
        )  # TODO change this
        self.feet_air_time = torch.zeros(
            self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_contacts = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.sampling_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        print(self.dof_names)

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.anymal_config.environment.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for j, dof_name in enumerate(self.control_config.joints):
                if dof_name in name:
                    self.p_gains[i] = self.control_config.stiffness[j]
                    self.d_gains[i] = self.control_config.damping[j]
                    found = True
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.control_config.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def module_angle_with_reference(self, zyx_angles, reference_angle):
        if zyx_angles.shape[0] == 0:
            return zyx_angles

        z = zyx_angles[:, 0]
        lower_bound = reference_angle - torch.pi
        upper_bound = reference_angle + torch.pi

        ll = lower_bound + torch.fmod(z - lower_bound, 2.0 * torch.pi)
        uu = upper_bound - torch.fmod(upper_bound - z, 2.0 * torch.pi)

        z = torch.where(z > upper_bound, ll, z)
        z = torch.where(z < lower_bound, uu, z)

        zyx_angles[:, 0] = z
        return zyx_angles
    
    def get_ocs2_state_perceptive(self, env_ids):
        # Euler zyx angles
        zyx_euler_angles = pytorch3d.transforms.matrix_to_euler_angles(
            pytorch3d.transforms.quaternion_to_matrix(self.root_states[env_ids][:, [6, 3, 4, 5]]),  # w, x, y, z,
            convention="ZYX",
        )
        zyx_euler_angles = self.module_angle_with_reference(zyx_euler_angles, self.reference_angle[env_ids])
        self.reference_angle[env_ids] = zyx_euler_angles[:, 0]
        zyx_euler_angles = zyx_euler_angles.cpu()

        # Position in world frame
        pos_com = self.root_states[env_ids][:, [0, 1, 2]].cpu()

        # Angular velocity in base frame
        w_com = self.root_states[env_ids][:, 10:13]
        w_com_local = quat_rotate_inverse(self.base_quat[env_ids], w_com).cpu()

        # Linear velocity in base frame
        v_com = self.root_states[env_ids][:, 7:10]
        v_com_local = quat_rotate_inverse(self.base_quat[env_ids], v_com).cpu()

        joint_pos = self.dof_pos[env_ids].cpu()

        return torch.cat((zyx_euler_angles, pos_com, w_com_local, v_com_local, joint_pos), dim=1)

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                print(self.reward_scales[key])
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }

    def _create_ground_plane(self):
        """Adds a ground plane to the simulation, sets friction and restitution based on the cfg."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.terrain_config.static_friction
        plane_params.dynamic_friction = self.terrain_config.dynamic_friction
        plane_params.restitution = self.terrain_config.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """Adds a heightfield terrain to the simulation, sets parameters based on the cfg."""
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain_config.horizontal_scale
        hf_params.row_scale = self.terrain_config.horizontal_scale
        hf_params.vertical_scale = self.terrain_config.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain_config.border_size
        hf_params.transform.p.y = -self.terrain_config.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.terrain_config.static_friction
        hf_params.dynamic_friction = self.terrain_config.dynamic_friction
        hf_params.restitution = self.terrain_config.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        )

    def _create_trimesh(self):
        """Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        #"""
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain_config.border_size
        tm_params.transform.p.y = -self.terrain_config.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.terrain_config.static_friction
        tm_params.dynamic_friction = self.terrain_config.dynamic_friction
        tm_params.restitution = self.terrain_config.restitution
        self.gym.add_triangle_mesh(
            self.sim, self.terrain.vertices.flatten(order="C"), self.terrain.triangles.flatten(order="C"), tm_params
        )
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        )


    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.asset_config.file.format(LEGGED_GYM_ROOT_DIR=URDF_FOLDER)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = ac.get_asset_options(self.anymal_config)

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.name2idx = self.gym.get_asset_rigid_body_dict(robot_asset)

        print(self.name2idx.keys())
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        print(body_names)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.asset_config.foot_name in s]
        penalized_contact_names = []
        for name in self.asset_config.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.asset_config.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = (
            self.init_state_config.pos
            + self.init_state_config.rot
            + self.init_state_config.lin_vel
            + self.init_state_config.ang_vel
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, self.asset_config.name, i, self.asset_config.self_collisions, 0
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )

        # shank names
        shank_names = [s for s in body_names if self.asset_config.shank_name in s]
        self.shank_indices = torch.zeros(len(shank_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(shank_names)):
            self.shank_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], shank_names[i]
            )

        # thigh names
        thigh_names = [s for s in body_names if self.asset_config.thigh_name in s]
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(thigh_names)):
            self.thigh_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], thigh_names[i]
            )

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

        self.friction_coeffs = self.friction_coeffs.to(self.device)

    def _get_env_origins(self):
        """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """
        if self.terrain_config.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.terrain_config.max_init_terrain_level
            if not self.terrain_config.curriculum:
                max_init_level = self.terrain_config.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / self.terrain_config.num_cols),
                rounding_mode="floor",
            ).to(torch.long)
            self.max_terrain_level = self.terrain_config.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.env_config.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0

    def _parse_cfg(self):
        self.dt = self.control_config.decimation * self.sim_config.dt
        self.obs_scales = self.normalization_config.obs_scales
        self.reward_scales = self.anymal_config.environment.rewards.scales

        self.command_ranges = self.command_config.ranges

        if self.terrain_config.mesh_type not in ["heightfield", "trimesh"]:
            assert self.terrain_config.curriculum is False, f"Cannot use curriculum for {self.terrain_config.mesh_type}"
        self.max_episode_length_s = self.env_config.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        print(self.max_episode_length)
        print(self.max_episode_length_s)
        print(self.dt)
        print("===== HERE ======" * 3)

    def _draw_debug_vis(self):
        """Draws visualizations for dubugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
        """

        ### HEIGHTS ###
        # # draw height lines
        # if not self.terrain_config.measure_heights:
        #     return
        # self.gym.clear_lines(self.viewer)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        # sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        # for i in range(self.num_envs):
        #     heights = self.measured_heights[i].cpu().numpy()
        #     points = self.get_height_points2().cpu().numpy()
        #     for j in range(heights.shape[0]):
        #         x = points[i, j, 0]
        #         y = points[i, j, 1]
        #         z = heights[j]
        #         sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        #         gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        ### DESIRED FOOTHOLDS ###
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))

        env_pts = torch.arange(2).long()
        sampling_positions, length_x, length_y, resolution = self.generate_pts(env_pts)

        for j in range(min(5, self.num_envs)):
            for i in range(4):
                desired_foothold = self.tbai_ocs2_interface.get_planar_footholds()[j, i * 2 : i * 2 + 2]

                # temp = torch.zeros((1, 3), device=self.device)
                # temp[0, 0:2] = self.tbai_ocs2_interface.get_planar_footholds()[0, i*2:i*2+2]
                # temp[0, 0:2] -= self.root_states[0, 0:2]
                # temp = quat_apply_yaw_inverse(self.base_quat[0].view(1, -1), temp)

                # # planar_footholds[:, i*2:i*2+2] = temp[:, :2]
                # desired_foothold = temp[0, :2]

                if self.terrain_config.mesh_type != "plane":

                    desired_foothold = desired_foothold.cpu()
                    x = desired_foothold[0] 
                    y = desired_foothold[1]
                    xp, yp = (x + self.terrain_config.border_size)/self.terrain_config.horizontal_scale, (y + self.terrain_config.border_size)/self.terrain_config.horizontal_scale
                    z = self.height_samples[int(xp), int(yp)] * self.terrain_config.vertical_scale
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose)

            if j == 0 and False:
                for i in range(sampling_positions.shape[1]):
                    x = sampling_positions[0].view(-1, 3)[i][0]
                    y = sampling_positions[0].view(-1, 3)[i][1]
                    z = sampling_positions[0].view(-1, 3)[i][2]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose)

        # Draw a sphere above the first robot
        x = self.root_states[0, 0]
        y = self.root_states[0, 1]
        z = self.root_states[0, 2] + 0.5
        sphere_geom = gymutil.WireframeSphereGeometry(0.5, 4, 4, None, color=(0, 1, 0))
        sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose)

    def draw_spheres(self, x, y, z, reset=True, id=0, color=(1, 1, 0)):
        # heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - measured_heights, -1, 1.) * self.obs_scales.height_measurements
        if reset:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=color)

        x = x - self.terrain_config.border_size
        y = y - self.terrain_config.border_size
        z = -((z / self.obs_scales.height_measurements) + 0.5 - self.root_states[id, 2])

        x = x.cpu().numpy()
        y = y.cpu().numpy()
        z = z.cpu().numpy()
        for i in range(x.shape[0]):
            sphere_pose = gymapi.Transform(gymapi.Vec3(x[i], y[i], z[i]), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[id], sphere_pose)

    def _init_height_points(self):
        Ns = np.array([6, 8, 10, 12, 16])
        rs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        assert Ns.size == rs.size

        n_points = Ns.sum()
        sampling_points = np.zeros((n_points, 3))

        i = 0
        for r, N in zip(rs, Ns):
            for idx in range(N):
                angle = np.deg2rad(360 / N * idx)
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                sampling_points[i, :2] = [x, y]
                i += 1
        points = torch.zeros(self.num_envs, n_points, 3, device=self.device, requires_grad=False)
        sampling_points = torch.from_numpy(sampling_points)
        for i in range(self.num_envs):
            points[i] = sampling_points
        return points

    def get_height_points2(self):
        n = self.sampling_points.shape[1]
        points_rotated = quat_apply_yaw(self.base_quat.repeat(1, n), self.sampling_points)
        lf_points = points_rotated + self.lf_foot_position.unsqueeze(1)
        lh_points = points_rotated + self.lh_foot_position.unsqueeze(1)
        rf_points = points_rotated + self.rf_foot_position.unsqueeze(1)
        rh_points = points_rotated + self.rh_foot_position.unsqueeze(1)
        return torch.cat((lf_points, lh_points, rf_points, rh_points), dim=1)

    def _get_heights(self, env_ids=None, noise_generator=None):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.terrain_config.mesh_type == "plane":
            self.num_height_points = 4 * 52
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.terrain_config.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = self.get_height_points2()  # (num_envs, 208, 3)
        # if env_ids:
        #     points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        # else:
        #     points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain_config.border_size

        # add noise
        if noise_generator is not None:
            x_noise, y_noise, z_noise = noise_generator.sample_noise()
            # print("x_noise", x_noise.shape)
            # print(torch.max(torch.abs(x_noise)))
            # print(torch.max(torch.abs(x_noise)))
            # print("z noise", z_noise.shape)
            # print(torch.max(torch.abs(z_noise)))
            points[:, :, 0] += x_noise
            points[:, :, 1] += y_noise

        points = (points / self.terrain_config.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py] * self.terrain_config.vertical_scale
        if noise_generator is not None:
            heights1 += z_noise.view(-1)
        return heights1.view(self.num_envs, -1), points * self.terrain_config.horizontal_scale

    def _get_surface_normals(self):
        surface_normals = list()

        for foot in [self.lf_foot_position, self.lh_foot_position, self.rf_foot_position, self.rh_foot_position]:
            foot_position = foot.clone().squeeze(1)  # (num_envs, 3)
            foot_position[:, [0, 1]] += self.terrain_config.border_size
            foot_point = (foot_position / self.terrain_config.horizontal_scale).long()
            px = foot_point[:, 0]
            px = torch.clip(px, 1, self.height_samples.shape[0] - 2)

            py = foot_point[:, 1]
            py = torch.clip(py, 1, self.height_samples.shape[1] - 2)

            sn = torch.ones(self.num_envs, 3, device=self.device)
            sn[:, 0] = (
                self.terrain_config.vertical_scale
                * (self.height_samples[px + 1, py] - self.height_samples[px - 1, py])
                / (2 * self.terrain_config.horizontal_scale)
            )
            sn[:, 1] = (
                self.terrain_config.vertical_scale
                * (self.height_samples[px, py + 1] - self.height_samples[px, py - 1])
                / (2 * self.terrain_config.horizontal_scale)
            )
            sn = sn / torch.norm(sn, dim=1, keepdim=True)

            surface_normals.append(sn)

        return surface_normals

    # ------------ reward functions----------------

    def _reward_tracking_linear_velocity_new(self):
        current_velocity = self.base_lin_vel[:, :2]  # (num_envs, 2)
        desired_velocity = self.commands[:, :2]  # (num_envs, 2)

        desired_velocity_norm = torch.norm(desired_velocity, dim=1)  # (num_envs, )
        current_velocity_norm = torch.norm(current_velocity, dim=1)  # (num_envs, )

        curdes_dot = (current_velocity * desired_velocity).sum(dim=1)  # (num_envs, )

        reward = torch.where(
            desired_velocity_norm < 1e-3,
            (-(current_velocity_norm**2)).exp(),
            torch.where(curdes_dot > desired_velocity_norm, 1.0, (-((curdes_dot - desired_velocity_norm) ** 2)).exp()),
        )

        return reward

    def _reward_tracking_angular_velocity_new(self):
        current_velocity = self.base_ang_vel[:, 2]  # (num_envs, )
        desired_velocity = self.commands[:, 2]

        desired_velocity_norm = torch.abs(desired_velocity)
        current_velocity_norm = torch.abs(current_velocity)

        reward = torch.where(
            desired_velocity_norm < 1e-3,
            (-(current_velocity_norm**2)).exp(),
            torch.where(
                current_velocity_norm > desired_velocity_norm,
                1.0,
                (-((desired_velocity_norm - current_velocity_norm) ** 2)).exp(),
            ),
        )

        return reward

    def _reward_linear_orthogonal_velocity_new(self):
        current_velocity = self.base_lin_vel[:, :2]  # (num_envs, 2)
        desired_velocity = self.commands[:, :2]  # (num_envs, 2)
        desired_velocity_norm = torch.norm(desired_velocity, dim=1, keepdim=True)  # (num_envs, )
        elig = desired_velocity_norm.squeeze() > 1e-3
        v_0 = torch.zeros_like(current_velocity)
        v_0[elig] = (
            current_velocity[elig]
            - (current_velocity[elig] * desired_velocity[elig]).sum(dim=1, keepdim=True)
            * desired_velocity[elig]
            / desired_velocity_norm[elig] ** 2
        )
        reward = torch.exp(-3.0 * torch.norm(v_0, dim=1) ** 2)
        return reward

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.square(self.base_ang_vel[:, :2]).sum(dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return self.ck * torch.square(base_height - self.rewards_config.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return self.ck * torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return self.ck * torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(
            1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1
        )

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.rewards_config.soft_dof_vel_limit).clip(
                min=0.0, max=1.0
            ),
            dim=1,
        )

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.rewards_config.soft_torque_limit).clip(min=0.0), dim=1
        )

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.rewards_config.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.rewards_config.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.35) * first_contact, dim=1
        )  # reward only on first contact with the ground
        # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(self.contacts, dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        # return (torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1))) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum(
            (
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.rewards_config.max_contact_force
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_smoothness(self):
        diff1 = self.actions[:, :] - self.dof_action_history[0, :, :]
        diff2 = self.actions[:, :] - 2.0 * self.dof_action_history[0, :, :] + self.dof_action_history[1, :, :]
        return self.ck * (diff1.square().sum(dim=1) + diff2.square().sum(dim=1))

    def _reward_swing_height(self):
        allowed_swing_height = 0.2
        max_heights = torch.zeros(self.num_envs, 4, device=self.device)
        #########
        #  #
        #
        ########

        lf_heights = self.lf_foot_position[:, 2].unsqueeze(1)
        lh_heights = self.lh_foot_position[:, 2].unsqueeze(1)
        rf_heights = self.rf_foot_position[:, 2].unsqueeze(1)
        rh_heights = self.rh_foot_position[:, 2].unsqueeze(1)

        foot_heights = max_heights - torch.cat([lf_heights, lh_heights, rf_heights, rh_heights], dim=1)

        penalize = foot_heights <= -allowed_swing_height
        return self.ck * torch.sum(penalize, dim=1).float()

    def _reward_action_norm(self):
        return torch.norm(self.actions, dim=1)

    def _reward_consistency(self):
        c = self.tbai_ocs2_interface.get_consistency_reward().clone()
        self.tbai_ocs2_interface.get_consistency_reward()[:] = 0.0
        return c * self.ck

    def _reward_foot_position_tracking(self):
        eps = 1e-5

        apply_reward = (torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=-1) >= 1.0).float()
        # print(apply_reward.shape)
        # print(self.tracking_reward_ready.shape)
        apply_reward *= self.tracking_reward_ready
        self.tracking_reward_ready *= 1.0 - apply_reward

        tt = torch.logical_and(self.aa, self.bb)

        lf_foot_positions = self.lf_foot_position[:, :2]
        lf_foot_positions_desired = self.tbai_ocs2_interface.get_planar_footholds()[:, 0:2]
        lf_diff = (lf_foot_positions_desired - lf_foot_positions).square().sum(dim=1) + eps

        lh_foot_positions = self.lh_foot_position[:, :2]
        lh_foot_positions_desired = self.tbai_ocs2_interface.get_planar_footholds()[:, 4:6]
        lh_diff = (lh_foot_positions_desired - lh_foot_positions).square().sum(dim=1) + eps

        rf_foot_positions = self.rf_foot_position[:, :2]
        rf_foot_positions_desired = self.tbai_ocs2_interface.get_planar_footholds()[:, 2:4]
        rf_diff = (rf_foot_positions_desired - rf_foot_positions).square().sum(dim=1) + eps

        rh_foot_positions = self.rh_foot_position[:, :2]
        rh_foot_positions_desired = self.tbai_ocs2_interface.get_planar_footholds()[:, 6:8]
        rh_diff = (rh_foot_positions_desired - rh_foot_positions).square().sum(dim=1) + eps

        reward = (torch.stack([lf_diff.log(), lh_diff.log(), rf_diff.log(), rh_diff.log()], dim=-1) * tt).sum(dim=-1)

        self.aa[tt] = False
        self.bb[tt] = False

        return -reward

    def _reward_exp_base_position(self):
        base_position = self.root_states[:, :3]
        base_position_desired = self.tbai_ocs2_interface.get_desired_base_positions().to(
            self.device
        )  # TODO: Why to(device)? Should be already on device

        sigma = 1200.0
        # print(base_position_desired[0])
        # print(base_position[0])
        # print()
        diff = (base_position - base_position_desired).square().sum(dim=1)
        reward = torch.exp(-sigma * diff)
        return reward

    def _reward_exp_base_linear_velocity(self):
        base_velocity = self.root_states[:, 7:10]
        base_velocity_desired = self.tbai_ocs2_interface.get_desired_base_linear_velocities().to(self.device)

        # print("Current base velocity:", base_velocity[0])
        # print("Desired base velocity:", base_velocity_desired[0])
        # print()

        sigma = 10.0
        diff = (base_velocity - base_velocity_desired).square().sum(dim=1)
        reward = torch.exp(-sigma * diff)
        return reward

    def _reward_exp_base_orientation(self):
        base_orientation = self.root_states[:, 3:7]
        base_orientation_desired = self.tbai_ocs2_interface.get_desired_base_orientations().to(self.device)

        xyz1 = base_orientation[:, 0:3]
        w2 = base_orientation_desired[:, 3]

        xyz2 = base_orientation_desired[:, 0:3]
        w1 = base_orientation[:, 3]

        diff = xyz1 * w2.view(-1, 1) - xyz2 * w1.view(-1, 1) + torch.cross(xyz1, xyz2)
        diff = diff.square().sum(dim=1)

        sigma = 90.0
        reward = torch.exp(-sigma * diff)
        return reward

    def _reward_exp_base_angular_velocity(self):
        base_angular_velocity = self.root_states[:, 10:13]
        base_angular_velocity_desired = self.tbai_ocs2_interface.get_desired_base_angular_velocities().to(self.device)

        sigma = 1.0
        diff = (base_angular_velocity - base_angular_velocity_desired).square().sum(dim=1)
        reward = torch.exp(-sigma * diff)
        return reward

    def _reward_exp_base_linear_acceleration(self):
        base_acceleration = (self.root_states[:, 7:10] - self.last_base_lin_vel) / self.dt
        base_acceleration_desired = self.tbai_ocs2_interface.get_desired_base_linear_accelerations().to(self.device)

        sigma = 0.05
        diff = (base_acceleration - base_acceleration_desired).square().sum(dim=1)
        reward = torch.exp(-sigma * diff)
        return reward

    def _reward_exp_base_angular_acceleration(self):
        base_angular_acceleration = (self.root_states[:, 10:13] - self.last_base_ang_vel) / self.dt
        base_angular_acceleration_desired = self.tbai_ocs2_interface.get_desired_base_angular_accelerations().to(
            self.device
        )

        sigma = 0.005
        diff = (base_angular_acceleration - base_angular_acceleration_desired).square().sum(dim=1)
        reward = torch.exp(-sigma * diff)
        return reward

    def _reward_foot_power(self):
        lf_velocity = self.lf_foot_velocity
        lf_force = self.contact_forces[:, self.feet_indices[0], :] * self.obs_scales.contact_forces
        lf_power = (lf_velocity * lf_force).norm(dim=1)

        lh_velocity = self.lh_foot_velocity
        lh_force = self.contact_forces[:, self.feet_indices[1], :] * self.obs_scales.contact_forces
        lh_power = (lh_velocity * lh_force).norm(dim=1)

        rf_velocity = self.rf_foot_velocity
        rf_force = self.contact_forces[:, self.feet_indices[2], :] * self.obs_scales.contact_forces
        rf_power = (rf_velocity * rf_force).norm(dim=1)

        rh_velocity = self.rh_foot_velocity
        rh_force = self.contact_forces[:, self.feet_indices[3], :] * self.obs_scales.contact_forces
        rh_power = (rh_velocity * rh_force).norm(dim=1)

        return (lf_power + lh_power + rf_power + rh_power).view(-1)

    def _reward_z_height(self):
        desired_height = 0.55 + self.sample_at_position(self.root_states[:, 0], self.root_states[:, 1])
        current_height = self.root_states[:, 2]
        diff = (desired_height - current_height).square()

        return (-diff) * (1.0 - self.ck)

    def _reward_follow_joint_trajectory(self):
        # Penalize deviation from joint trajectory

        des = self.tbai_ocs2_interface.get_current_desired_joint_positions()
        cur = self.dof_pos

        error = (des - cur).square().sum(dim=1)
        return error

    def update_curriculum(self):
        CURRICULUM_UPDATES = 10000.0

        if self.iter == CURRICULUM_UPDATES:
            print("Done curriculum")

        if self.iter <= CURRICULUM_UPDATES:
            self.ck = min(self.iter / CURRICULUM_UPDATES, 1.0)
            self.iter += 1.0

    def _reward_desired_foot_position_wbc_tracking(self):
        lf_error = torch.norm(
            self.lf_foot_position - self.tbai_ocs2_interface.get_desired_foot_positions()[:, :3], dim=1
        )
        rf_error = torch.norm(
            self.rf_foot_position - self.tbai_ocs2_interface.get_desired_foot_positions()[:, 3:6], dim=1
        )
        lh_error = torch.norm(
            self.lh_foot_position - self.tbai_ocs2_interface.get_desired_foot_positions()[:, 6:9], dim=1
        )
        rh_error = torch.norm(
            self.rh_foot_position - self.tbai_ocs2_interface.get_desired_foot_positions()[:, 9:], dim=1
        )

        return (lf_error + rf_error + lh_error + rh_error) * self.ck

    def _reward_desired_foot_velocity_wbc_tracking(self):
        lf_error = torch.norm(
            self.lf_foot_velocity - self.tbai_ocs2_interface.get_desired_foot_velocities()[:, :3], dim=1
        )
        rf_error = torch.norm(
            self.rf_foot_velocity - self.tbai_ocs2_interface.get_desired_foot_velocities()[:, 3:6], dim=1
        )
        lh_error = torch.norm(
            self.lh_foot_velocity - self.tbai_ocs2_interface.get_desired_foot_velocities()[:, 6:9], dim=1
        )
        rh_error = torch.norm(
            self.rh_foot_velocity - self.tbai_ocs2_interface.get_desired_foot_velocities()[:, 9:], dim=1
        )

        return (lf_error + rf_error + lh_error + rh_error) * self.ck
