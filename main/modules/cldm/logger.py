import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
# from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        # root = os.path.join(save_dir, "image_log", split+"_titan")
        path_image = {}
        for k in images:
            value = images[k]
            if not isinstance(value, torch.Tensor):
                try:
                    value = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in value])
                except Exception:
                    value = torch.tensor(value)
            grid = torchvision.utils.make_grid(value, nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            grid = torch.clamp(grid.detach().cpu(), 0.0, 1.0)
            torchvision.utils.save_image(grid, path)
            path_image[k] = grid
        if "control_mask" in path_image and "samples_cfg_scale_9.00_mask" in path_image:
            grid_control = path_image["control_mask"]
            grid_samples = path_image["samples_cfg_scale_9.00_mask"]
            blended_filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format("blended", global_step, current_epoch, batch_idx)
            blended_path = os.path.join(root, blended_filename)
            blended = torch.clamp(0.5 * grid_control + 0.5 * grid_samples, 0.0, 1.0)
            torchvision.utils.save_image(blended, blended_path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            path_image = {}
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if self.log_first_step and check_idx == 0:
            return True
        return check_idx % self.batch_freq == 0

    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            # if pl_module.current_epoch % 20 == 0:
            self.log_img(pl_module, batch, batch_idx, split="train")
