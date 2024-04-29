#!/usr/bin/env python3

from torch.utils.tensorboard import SummaryWriter


class AbstractWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def upload(self):
        raise NotImplementedError


class TensorboardWriter(AbstractWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def upload(self):
        pass


class WandbWriter(AbstractWriter):
    def __init__(self, project_name):
        super().__init__()
        import wandb

        wandb.init(project=project_name)
        self.wandb = wandb

        self.data = {}

    def add_scalar(self, name, value, step):
        self.step = int(step)
        self.data[name] = value

    def upload(self):
        self.wandb.log(self.data, step=int(self.step))
        self.data = {}


class WandbWriter(SummaryWriter):
    def __init__(self, project_name):
        import wandb

        wandb.init(project=project_name)
        self.wandb = wandb

        self.data = {}

    def add_scalar(self, name, value, step):
        self.step = int(step)
        self.data[name] = value

    def upload(self):
        self.wandb.log(self.data, step=int(self.step))
        self.data = {}
