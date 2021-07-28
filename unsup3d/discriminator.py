import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from . import networks
from . import utils
from .renderer import Renderer
import lpips
from PIL import Image

EPS = 1e-7


class Unsup3D_Discriminator:
  def __init__(self, cfgs):
    self.device = cfgs.get('device', 'cpu')
    self.image_size = cfgs.get('image_size', 64)
    self.depthmap_size = cfgs.get('depthmap_size', 32)
    self.lr = cfgs.get('lr_discriminator', 1e-4)
    self.model = networks.DCDiscriminator(in_dim=3, n_feat=512, img_size=self.image_size)
        # self.netG_discriminator_flipped = networks.DCDiscriminator(in_dim=3, n_feat=512, img_size=self.image_size)
    self.make_optimizer = lambda model: torch.optim.Adam(
      filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

  def toggle_grad(self, requires_grad):
    for p in self.model.parameters():
      p.requires_grad_(requires_grad)
    
  def to_device(self, device):
    self.model.to(device)

  def init_optimizers(self):
    self.optimizer_names = []
    optimizer = self.make_optimizer(self.model)
    optim_name = "optimizerD"
    setattr(self, optim_name, optimizer)
    self.optimizer_names += [optim_name]
    print(f"The following optimizers were initialised: {self.optimizer_names}")

  def set_train(self):
    self.model.train()
  
  def set_eval(self):
    self.model.eval()

  def reset_optimizer(self):
    for optim_name in self.optimizer_names:
      getattr(self, optim_name).zero_grad()
    
  def optimizer_step(self):
    for optim_name in self.optimizer_names:
      getattr(self, optim_name).step()

  def forward(self,input):
    self.input_im = input.to(self.device)
    self.output = self.model(self.input_im)
    return self.output

  def visualize(self, logger, total_iter, fake):
    output_name = "fake" if fake == True else "real"
    output = F.sigmoid(self.output) 
    logger.add_histogram(f"Discriminator/discriminator_output_{output_name}", output, total_iter)