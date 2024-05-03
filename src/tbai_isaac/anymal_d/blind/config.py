#!/usr/bin/env python3

import numpy as np
from isaacgym import gymapi
from omegaconf import OmegaConf
from tbai_isaac.anymal_d.info import URDF_PATH
from tbai_isaac.common.config import select


def get_asset_options(config: OmegaConf) -> gymapi.AssetOptions:
    asset_options = gymapi.AssetOptions()
    print(config)
    asset_options.default_dof_drive_mode = int(config.environment.asset.default_dof_drive_mode)

    asset_options.collapse_fixed_joints = bool(config.environment.asset.collapse_fixed_joints)
    asset_options.replace_cylinder_with_capsule = bool(config.environment.asset.replace_cylinder_with_capsule)
    asset_options.flip_visual_attachments = bool(config.environment.asset.flip_visual_attachments)
    asset_options.fix_base_link = bool(config.environment.asset.fix_base_link)
    asset_options.density = float(config.environment.asset.density)
    asset_options.angular_damping = float(config.environment.asset.angular_damping)
    asset_options.linear_damping = float(config.environment.asset.linear_damping)
    asset_options.max_angular_velocity = float(config.environment.asset.max_angular_velocity)
    asset_options.max_linear_velocity = float(config.environment.asset.max_linear_velocity)
    asset_options.armature = float(config.environment.asset.armature)
    asset_options.thickness = float(config.environment.asset.thickness)
    asset_options.disable_gravity = bool(config.environment.asset.disable_gravity)
    return asset_options


def get_asset_config(config: OmegaConf):
    config.environment.asset.file = URDF_PATH
    return select(config, "environment.asset")


def get_randomization_config(config: OmegaConf):
    dt = config.environment.sim.dt * config.environment.control.decimation
    push_interval_s = config.environment.domain_randomization.push_interval_s
    print(type(np.ceil(push_interval_s / dt)))
    config.environment.domain_randomization.push_interval = float(np.ceil(push_interval_s / dt))
    return select(config, "environment.domain_randomization")


def get_terrain_config(config: OmegaConf):
    return select(config, "environment.terrain")


def get_command_config(config: OmegaConf):
    return select(config, "environment.command")


def get_env_config(config: OmegaConf):
    return select(config, "environment.env")


def get_control_config(config: OmegaConf):
    return select(config, "environment.control")


def get_viewer_config(config: OmegaConf):
    return select(config, "environment.viewer")


def get_normalization_config(config: OmegaConf):
    return select(config, "environment.normalization")


def get_rewards_config(config: OmegaConf):
    return select(config, "environment.rewards")


def get_init_state_config(config: OmegaConf):
    return select(config, "environment.init_state")


def get_noise_config(config: OmegaConf):
    return select(config, "environment.noise")


def get_sim_config(config: OmegaConf):
    return select(config, "environment.sim")


def get_sim_params(config: OmegaConf) -> gymapi.SimParams:
    sim_config = select(config, "environment.sim")

    sim_params = gymapi.SimParams()

    assert sim_config.physics_engine == "physx", "Only PhysX is supported"
    sim_params.dt = float(sim_config.dt)
    sim_params.substeps = int(sim_config.substeps)
    sim_params.up_axis = gymapi.UpAxis(sim_config.up_axis)
    sim_params.gravity = gymapi.Vec3(sim_config.gravity[0], sim_config.gravity[1], sim_config.gravity[2])
    sim_params.use_gpu_pipeline = True  # Only GPU is supported

    # physx specific
    sim_params.physx.use_gpu = True  # Only GPU is supported
    sim_params.physx.num_threads = int(sim_config.physx.num_threads)
    sim_params.physx.num_position_iterations = int(sim_config.physx.num_position_iterations)
    sim_params.physx.num_velocity_iterations = int(sim_config.physx.num_velocity_iterations)
    sim_params.physx.contact_offset = float(sim_config.physx.contact_offset)
    sim_params.physx.rest_offset = float(sim_config.physx.rest_offset)

    sim_params.physx.bounce_threshold_velocity = float(sim_config.physx.bounce_threshold_velocity)
    sim_params.physx.max_depenetration_velocity = float(sim_config.physx.max_depenetration_velocity)
    sim_params.physx.max_gpu_contact_pairs = int(sim_config.physx.max_gpu_contact_pairs)
    sim_params.physx.default_buffer_size_multiplier = int(sim_config.physx.default_buffer_size_multiplier)
    sim_params.physx.contact_collection = gymapi.ContactCollection(int(sim_config.physx.contact_collection))

    return sim_params
