#!/usr/bin/env python3

import numpy as np
from isaacgym import gymapi
from tbai_isaac.anymal_d.info import URDF_PATH
from tbai_isaac.common.config import YamlConfig


class AnymalConfig(YamlConfig):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    def get_asset_options(self) -> gymapi.AssetOptions:
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self("environment/asset/default_dof_drive_mode", int)
        asset_options.collapse_fixed_joints = self("environment/asset/collapse_fixed_joints", bool)
        asset_options.replace_cylinder_with_capsule = self("environment/asset/replace_cylinder_with_capsule", bool)
        asset_options.flip_visual_attachments = self("environment/asset/flip_visual_attachments", bool)
        asset_options.fix_base_link = self("environment/asset/fix_base_link", bool)
        asset_options.density = self("environment/asset/density", float)
        asset_options.angular_damping = self("environment/asset/angular_damping", float)
        asset_options.linear_damping = self("environment/asset/linear_damping", float)
        asset_options.max_angular_velocity = self("environment/asset/max_angular_velocity", float)
        asset_options.max_linear_velocity = self("environment/asset/max_linear_velocity", float)
        asset_options.armature = self("environment/asset/armature", float)
        asset_options.thickness = self("environment/asset/thickness", float)
        asset_options.disable_gravity = self("environment/asset/disable_gravity", bool)

        return asset_options

    def get_asset_config(self):
        self["environment/asset/file"] = URDF_PATH
        return self.as_dataclass("environment/asset")

    def get_randomization_config(self):
        dt = self["environment/sim/dt", float] * self["environment/control/decimation", int]
        push_interval_s = self["environment/domain_randomization/push_interval_s", float]
        self["environment/domain_randomization/push_interval"] = np.ceil(push_interval_s / dt)
        return self.as_dataclass("environment/domain_randomization")

    def get_terrain_config(self):
        return self.as_dataclass("environment/terrain")

    def get_command_config(self):
        return self.as_dataclass("environment/command")

    def get_env_config(self):
        return self.as_dataclass("environment/env")

    def get_control_config(self):
        return self.as_dataclass("environment/control")

    def get_viewer_config(self):
        return self.as_dataclass("environment/viewer")

    def get_normalization_config(self):
        return self.as_dataclass("environment/normalization")

    def get_rewards_config(self):
        return self.as_dataclass("environment/rewards")

    def get_init_state_config(self):
        return self.as_dataclass("environment/init_state")

    def get_noise_config(self):
        return self.as_dataclass("environment/noise")

    def get_sim_config(self):
        return self.as_dataclass("environment/sim")

    def get_sim_params(self):
        sim_config = self.get_sim_config()

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
