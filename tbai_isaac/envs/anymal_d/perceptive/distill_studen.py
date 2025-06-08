#!/usr/bin/env python3

import tbai_isaac.envs.anymal_d.perceptive.config as ac
import tbai_isaac.envs.anymal_d.perceptive.student as student
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from tbai_isaac.envs.anymal_d.common.noise_model import ExteroceptiveNoiseGenerator
from tbai_isaac.envs.anymal_d.perceptive.env import LeggedRobot
from tbai_isaac.envs.anymal_d.perceptive.student import StudentPolicy, StudentPolicyJitted
from tbai_isaac.envs.anymal_d.perceptive.teacher import TeacherNetwork
from tbai_isaac.common.config import load_config
from tbai_isaac.common.utils import parse_args
from tbai_isaac.ppo.coach import Coach
from torch.utils.tensorboard import SummaryWriter


class Distiller:
    def __init__(self, config, teacher, student, env, n_iters, headless, device, logs_dir=None):
        self.config = config
        self.teacher = teacher
        self.teacher_policy = teacher.act_inference
        self.student = student
        self.env = env
        self.device = device
        self.n_iters = n_iters
        self.headless = headless

        self.noise_config = ac.get_noise_config(self.config)
        self.normalization_config = ac.get_normalization_config(self.config)

        self.start_lr = 3e-3
        self.end_lr = 1e-9
        self.batch_size = 32

        self.prepare_noise()
        self.prepare_optimizer()

        self.writer = SummaryWriter(logs_dir)

    def prepare_noise(self):
        noise_level = self.noise_config.noise_level
        noise_scales = self.noise_config.noise_scales
        observation_scales = self.normalization_config.obs_scales

        proprioceptive_size = student.proprioceptive_size
        self.proprioceptive_noise = torch.zeros(proprioceptive_size, device=self.device)
        self.proprioceptive_noise[0:3] = 0.0  # command
        self.proprioceptive_noise[3:6] = noise_scales.gravity * noise_level  # gravity
        self.proprioceptive_noise[6:9] = noise_scales.lin_vel * noise_level * observation_scales.lin_vel  # lin_vel
        self.proprioceptive_noise[9:12] = noise_scales.ang_vel * noise_level * observation_scales.ang_vel  # ang_vel
        self.proprioceptive_noise[12:24] = noise_scales.dof_pos * noise_level * observation_scales.dof_pos  # joint_pos
        self.proprioceptive_noise[24:36] = noise_scales.dof_vel * noise_level * observation_scales.dof_vel  # joint_vel
        self.proprioceptive_noise.to(self.device)

        self.exteroceptive_noise_generator = ExteroceptiveNoiseGenerator(
            points_per_leg=52,
            n_envs=self.config.environment.env.num_envs,
            env_max_steps=self.env.max_episode_length,
            n_legs=4,
        )

    def prepare_optimizer(self):
        self.optimizer = optim.Adam(self.student.parameters(), lr=self.start_lr)
        self.lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.n_iters // self.batch_size, eta_min=self.end_lr
        )

    def distill(self):
        # Get initial observations
        observation = self.env.get_observations()
        total_loss = 0.0

        loss_acum = 0.0

        trange = tqdm.tqdm(range(self.n_iters))
        for i in trange:
            # Unpack proprioceptive observation -- teacher
            proprioceptive = student.proprioceptive_from_observation(observation)

            # Introduce noise -- student
            student_proprioceptive = (
                proprioceptive + (2 * torch.rand_like(proprioceptive) - 1) * self.proprioceptive_noise
            )

            # Unpack exteroceptive observation -- teacher
            exteroceptive = student.exteroceptive_from_observation(observation)

            # Introduce noise -- student
            student_exteroceptive, points = self.env.get_heights_observation_with_noise(
                self.exteroceptive_noise_generator
            )

            # Update exteroceptive noise model
            self.exteroceptive_noise_generator.step()

            if not self.headless:
                x_points = points[0, :, 0].view(-1)
                y_points = points[0, :, 1].view(-1)
                z_points = student_exteroceptive[0, :].view(-1)

            # Unpack privileged observation -- teacher
            privileged = student.priviliged_from_observation(observation)

            reconstructed_target = torch.cat((exteroceptive, privileged), dim=-1)

            action_student, reconstructed_student = self.student(student_proprioceptive, student_exteroceptive)

            if not self.headless:
                self.env.draw_spheres(x_points, y_points, z_points, reset=True)
                self.env.draw_spheres(
                    x_points,
                    y_points,
                    student.exteroceptive_from_decoded(reconstructed_student.detach())[0, :].view(-1),
                    reset=False,
                    color=(0, 0, 1),
                )

            # What would the teached do?
            with torch.no_grad():
                teacher_actions = self.teacher_policy(observation)

            # Step environment
            observation, _, rewards, dones, infos = self.env.step(action_student.detach())

            # Reset envs
            self.exteroceptive_noise_generator.reset(dones.nonzero().squeeze())

            assert not teacher_actions.requires_grad
            assert action_student.requires_grad
            assert not reconstructed_target.requires_grad
            assert reconstructed_student.requires_grad

            # Calculate action reconstruction loss
            action_loss = nn.functional.mse_loss(action_student, teacher_actions)

            # Calculate exteroceptive and priviliged information loss
            reconstruction_loss = nn.functional.mse_loss(reconstructed_student, reconstructed_target)

            # Calculate total loss
            loss = action_loss + 0.5 * reconstruction_loss
            loss_acum += loss

            self.writer.add_scalar("action_loss", action_loss.item(), i)
            self.writer.add_scalar("reconstruction_loss", reconstruction_loss.item(), i)

            # Book keeping
            # loss.backward(retain_graph=True)

            total_loss += loss.item()
            if i % self.batch_size == (self.batch_size - 1):
                loss_acum.backward()
                total_loss /= self.batch_size
                trange.set_description(f"step: {i} | loss {total_loss} | lr {self.optimizer.param_groups[0]['lr']}")
                self.optimizer.param_groups[0]["lr"] /= self.batch_size
                self.optimizer.step()
                self.optimizer.param_groups[0]["lr"] *= self.batch_size
                self.optimizer.zero_grad()
                self.student.reset_graph()
                self.lr_schedule.step()
                self.writer.add_scalar("batch_loss", total_loss, i)
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], i)
                total_loss = 0.0
                loss_acum = 0.0

            self.student.reset(dones)


def distill(args):
    config = load_config(args.config)
    config.environment.env.num_envs = min(config.environment.env.num_envs, 4096)
    config.environment.terrain.curriculum = False
    config.environment.noise.add_noise = False
    config.environment.domain_randomization.randomize_friction = True
    config.environment.domain_randomization.push_robots = True

    env = LeggedRobot(config, args.headless)

    model_path = "./logs/model_3000.pt"

    actor_critic = TeacherNetwork(config)
    coach = Coach(env, config, actor_critic, "./logs", "cuda")
    coach.load(model_path)

    actor_critic = coach.alg.actor_critic
    student = StudentPolicy(config.environment.env.num_envs, actor_critic)

    distiller = Distiller(config, actor_critic, student, env, 50_000, args.headless, "cuda", "./logs/distill_blind2")
    distiller.distill()

    student_policy = StudentPolicyJitted(student)
    student_policy.export("./logs/student_jitted.pt")


if __name__ == "__main__":
    args = parse_args()
    distill(args)
