import math
import torch
import torch.nn as nn
import numpy as np
import torchvision
from . import networks
from . import utils
from .renderer import Renderer
import lpips
from PIL import Image

EPS = 1e-7


class Unsup3D_Discriminator():
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
    return self.model(self.input_im)

        #         #### GAN losses ####
        # # Generator loss & generator optimization
        # x_fake = self.recon_im
        # with torch.no_grad():
        #     d_fake = self.netG_discriminator(x_fake)
        # loss_gen = compute_bce(d_fake, 1)

        # # Discriminator loss & discriminator optimization
        # x_real = self.input_im
        # self.canon_depth*(1-depth_border) + depth_border *self.border_depth
        # x_real = x_real*self.recon_im_mask_both + tensor.ones_like(self.input_im)*(1-self.recon_im_mask_both)
        # # mask input_image
        # loss_dis_full = 0.

        # d_real = self.netG_discriminator(x_real)
        # loss_dis_real = compute_bce(d_real, 1)
        # loss_dis_full += loss_dis_real

        # # reg = 10. * compute_grad2(d_real, x_real).mean()
        # # loss_d_full += reg

        # x_fake = self.recon_im
        # x_fake.requires_grad = False
        # d_fake = self.netG_discriminator(x_fake)
        # loss_dis_fake = compute_bce(d_fake, 0)
        # loss_dis_full += loss_dis_fake


        # self.total_loss += loss_gen + loss_dis_full

        # # loss_d_full.backward()
        # # self.optimizer_d.step()
