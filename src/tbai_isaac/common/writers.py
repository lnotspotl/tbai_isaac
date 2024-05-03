#!/usr/bin/env python3

class Writer:
    def __init__(self, type):
        assert type in ["tensorboard", "wandb", "none"], f"Invalid writer type: {type}"
        self.type = type

        if self.type == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter

        if self.type == "wandb":
            import wandb
            self.wandb = wandb
            self.buffer = dict() 

        self.step = -1

    def add_scalar(self, tag: str, value: float, global_step: int):
        if self.type == "tensorboard":
            self.writer.add_scalar(tag, value, global_step)

        if self.type == "wandb":
            if self.step < global_step and len(self.buffer) > 0:
                self.wandb.log(self.buffer, step=self.step)
                self.buffer = dict()
            self.buffer[tag] = value
            self.step = global_step
        
        if self.type == "none":
            pass  # do nothing
