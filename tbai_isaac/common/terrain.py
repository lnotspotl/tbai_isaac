#!/usr/bin/env python3

import numpy as np
from isaacgym import terrain_utils


class Terrain:
    def __init__(self, terrain_config, num_robots) -> None:
        self.terrain_config = terrain_config
        self.num_sub_terrains = terrain_config.num_rows * terrain_config.num_cols

        self.num_robots = num_robots
        self.type = self.terrain_config.mesh_type
        if self.type in ["none", "plane"]:
            return
        self.env_length = self.terrain_config.terrain_length
        self.env_width = self.terrain_config.terrain_width
        self.proportions = [
            np.sum(self.terrain_config.terrain_proportions[: i + 1])
            for i in range(len(self.terrain_config.terrain_proportions))
        ]

        self.env_origins = np.zeros((self.terrain_config.num_rows, self.terrain_config.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.terrain_config.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.terrain_config.horizontal_scale)

        self.border = int(self.terrain_config.border_size / self.terrain_config.horizontal_scale)
        self.tot_cols = int(self.terrain_config.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.terrain_config.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        if self.terrain_config.curriculum:
            self.curiculum()
        else:
            self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.terrain_config.horizontal_scale,
                self.terrain_config.vertical_scale,
                self.terrain_config.slope_treshold,
            )

    def randomized_terrain(self):
        for k in range(self.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.terrain_config.num_rows, self.terrain_config.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.001, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self):
        for j in range(self.terrain_config.num_cols):
            for i in range(self.terrain_config.num_rows):
                difficulty = 0.7 * i / self.terrain_config.num_rows
                choice = j / self.terrain_config.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.terrain_config.vertical_scale,
            horizontal_scale=self.terrain_config.horizontal_scale,
        )
        slope = difficulty * 0.4
        step_height = 0.05 + self.terrain_config.step_height_multiplier * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 0.8 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1.0 * difficulty
        pit_depth = 1.0 * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.0)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.0)
            terrain_utils.random_uniform_terrain(
                terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2
            )
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.0)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.0
            rectangle_max_size = 2.0
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                discrete_obstacles_height,
                rectangle_min_size,
                rectangle_max_size,
                num_rectangles,
                platform_size=3.0,
            )
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(
                terrain,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                max_height=0.0,
                platform_size=4.0,
            )
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.0)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.0)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length / 2.0 - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2.0 + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2.0 - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2.0 + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def gap_terrain(terrain, gap_size, platform_size=1.0):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x - x2 : center_x + x2, center_y - y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x - x1 : center_x + x1, center_y - y1 : center_y + y1] = 0


def pit_terrain(terrain, depth, platform_size=1.0):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth