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

class Unsup3d_Generator():
    def __init__(self, cfgs):
        self.model_name = cfgs.get('model_name', self.__class__.__name__)
        self.device = cfgs.get('device', 'cpu')
        self.image_size = cfgs.get('image_size', 64)
        self.depthmap_size = cfgs.get('depthmap_size', 32)
        self.min_depth = cfgs.get('min_depth', 0.9)
        self.max_depth = cfgs.get('max_depth', 1.1)
        self.border_depth = cfgs.get('border_depth', (0.6*self.max_depth + 0.4*self.min_depth))
        self.xyz_rotation_range = cfgs.get('xyz_rotation_range', 60)
        self.xy_translation_range = cfgs.get('xy_translation_range', 0.1)
        self.z_translation_range = cfgs.get('z_translation_range', 0.1)
        self.lam_perc = cfgs.get('lam_perc', 1)
        self.lam_flip = cfgs.get('lam_flip', 0.5)
        self.lr = cfgs.get('lr_generator', 1e-4)
        self.load_gt_depth = cfgs.get('load_gt_depth', False)
        self.depthmap_mode = cfgs.get('depth_network', 'resnet')
        self.lam_perc_decrease_start_epoch = cfgs.get('lam_perc_decrease_start_epoch', 2)
        self.use_lpips = cfgs.get('use_lpips', False)
        self.conf_map_enabled = cfgs.get('conf_map_enabled', True)
        self.use_depthmap_prior = cfgs.get('use_depthmap_prior', True)
        self.renderer = Renderer(cfgs)
        self.alebdo_mode = cfgs.get('use_resnet_albedonet', False)
        self.depthmap_mode = cfgs.get('use_resnet_depthmapnet', False)

        ## networks and optimizers
        if self.depthmap_size == 64:
            if self.depthmap_mode:
                self.netD = networks.DepthMapResNet64(cin=3, cout=1, nf=64,activation=None)
            else:
                self.netD = networks.DepthMapNet64(cin=3, cout=1, nf=64,zdim=256,activation=None)
        else:
            if self.depthmap_mode:
                self.netD = networks.DepthMapResNet(cin=3, cout=1, nf=64,activation=None)
            else:
                self.netD = networks.DepthMapNet(cin=3, cout=1, nf=64,zdim=256,activation=None)

        if self.alebdo_mode:
            self.netA = networks.AlbedoMapResNet(cin=3, cout=3, nf=64)
        else:
            self.netA = networks.AlbedoMapNet(cin=3, cout=3, nf=64, zdim=256)

        self.netL = networks.Encoder(cin=3, cout=4, nf=32)
        self.netV = networks.Encoder(cin=3, cout=6, nf=32)
        self.netC = networks.ConfNet(cin=3, cout=2, nf=64, zdim=128)

        self.network_names = [k for k in vars(self) if 'net' in k]
        self.make_optimizer = lambda model: torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

        ## depth rescaler: -1~1 -> min_deph~max_deph
        self.depth_rescaler = lambda d : (1+d)/2 *self.max_depth + (1-d)/2 *self.min_depth

    def toggle_grad(self, requires_grad):
        for net_name in self.network_names:
            model = getattr(self, net_name)
            for p in model.parameters():
                p.requires_grad_(requires_grad)

    def init_optimizers(self):
        self.optimizer_names = []
        for net_name in self.network_names:
            optimizer = self.make_optimizer(getattr(self, net_name))
            optim_name = net_name.replace('net','optimizer')
            setattr(self, optim_name, optimizer)
            self.optimizer_names += [optim_name]
        print(f"The following optimizers were initialised: {self.optimizer_names}")

    def load_model_state(self, cp):
        for k in cp:
            if k and k in self.network_names:
                getattr(self, k).load_state_dict(cp[k])

    def load_optimizer_state(self, cp):
        for k in cp:
            if k and k in self.optimizer_names:
                getattr(self, k).load_state_dict(cp[k])

    def get_model_state(self):
        states = {}
        for net_name in self.network_names:
            states[net_name] = getattr(self, net_name).state_dict()
        return states

    def get_optimizer_state(self):
        states = {}
        for optim_name in self.optimizer_names:
            states[optim_name] = getattr(self, optim_name).state_dict()
        return states

    def to_device(self, device):
        self.device = device
        for net_name in self.network_names:
            setattr(self, net_name, getattr(self, net_name).to(device))

    def set_train(self):
        for net_name in self.network_names:
            getattr(self, net_name).train()

    def set_eval(self):
        for net_name in self.network_names:
            getattr(self, net_name).eval()

    def reset_optimizer(self):
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).zero_grad()
    
    def optimizer_step(self):
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).step()
    
    def forward(self, input):
        """Feedforward once."""
        if self.load_gt_depth:
            input, depth_gt = input
        self.input_im = input.to(self.device)
        b, c, h, w = self.input_im.shape

        ## predict canonical depth
        self.canon_depth_raw = self.netD(self.input_im).squeeze(1)  # BxHxW

        self.canon_depth = self.canon_depth_raw - self.canon_depth_raw.view(b,-1).mean(1).view(b,1,1)
        if self.use_depthmap_prior:
            # weak prior for the depth map ensures that the depth map sticks out of the image plane
            depthmap_prior = torch.from_numpy(np.load(f'/users/janhr/unsup3d_extended/unsup3d/depth_map_prior/64x64.npy')).to(self.device)
            depthmap_prior = depthmap_prior.unsqueeze(0).unsqueeze(0)
            depthmap_prior = torch.nn.functional.interpolate(depthmap_prior, size=[self.depthmap_size,self.depthmap_size], mode='nearest', align_corners=None)[0,...]
            self.canon_depth = self.canon_depth + 2*depthmap_prior
        self.canon_depth = self.canon_depth.tanh()
        self.canon_depth = self.depth_rescaler(self.canon_depth)

        ## clamp border depth
        _, h_depth, w_depth = self.canon_depth_raw.shape 
        if self.depthmap_size == 32:
            border_width = 4
        elif self.depthmap_size == 64:
            border_width = 8
        depth_border = torch.zeros(1,h_depth,w_depth-border_width).to(self.input_im.device)
        depth_border = nn.functional.pad(depth_border, (int(border_width/2),int(border_width/2)), mode='constant', value=1)
        self.canon_depth = self.canon_depth*(1-depth_border) + depth_border *self.border_depth
        self.canon_depth = torch.cat([self.canon_depth, self.canon_depth.flip(2)], 0)  # flip

        # canon_depth = self.canon_depth.detach()[0].cpu().numpy()*255
        # img = Image.fromarray(np.uint8(canon_depth))
        # img.save('canon_depth.png')


        ## predict canonical albedo
        self.canon_albedo = self.netA(self.input_im)  # Bx3xHxW
        self.canon_albedo = torch.cat([self.canon_albedo, self.canon_albedo.flip(3)], 0)  # flip

        # canon_albedo = self.canon_albedo.detach().permute(0,2,3,1)[b+1].cpu().numpy()*255
        # img = Image.fromarray(np.uint8(canon_albedo)).convert('RGB')
        # img.save('canon_albedo.png')

        ## predict confidence map
        self.conf_sigma_l1, self.conf_sigma_percl = self.netC(self.input_im)  # Bx2xHxW

        ## predict lighting
        canon_light = self.netL(self.input_im).repeat(2,1)  # Bx4
        self.canon_light_a = canon_light[:,:1] # ambience term
        self.canon_light_b = canon_light[:,1:2] # diffuse term
        canon_light_dxy = canon_light[:,2:]
        self.canon_light_d = torch.cat([canon_light_dxy, -torch.ones(b*2,1).to(self.input_im.device)], 1) # pytorch special: light direction in same direction as normals
        self.canon_light_d = self.canon_light_d / ((self.canon_light_d**2).sum(1, keepdim=True))**0.5  # diffuse light direction
        self.canon_light_b = torch.clamp(self.canon_light_b, min=-0.8, max=1)
        self.lighting = { "ambient": self.canon_light_a, "diffuse": self.canon_light_b, "direction": self.canon_light_d}

        ## predict viewpoint transformation
        self.view = self.netV(self.input_im).repeat(2,1)
        self.view = torch.cat([
            self.view[:,:3] *math.pi/180 *self.xyz_rotation_range,
            self.view[:,3:5] *self.xy_translation_range,
            self.view[:,5:] *self.z_translation_range], 1)

        ## reconstruct input view
        self.meshes = self.renderer.create_meshes_from_depth_map(self.canon_depth) # create meshes from vertices and faces
        recon_im = self.renderer(self.meshes, self.canon_albedo, self.view, self.lighting)
        self.recon_im = recon_im[...,:3]
        self.alpha_mask = recon_im[...,3]

        self.recon_im_mask = (self.alpha_mask > 0).type(torch.float32).unsqueeze(1)
        self.recon_im = self.recon_im.permute(0,3,1,2)
        self.alpha_mask = self.alpha_mask.unsqueeze(1)

        # recon_im_mask_both = self.recon_im_mask[:b] * self.recon_im_mask[b:]
        # detached_mask = recon_im_mask_both.repeat(2,1,1,1).detach()

        return self.recon_im, self.recon_im_mask, self.conf_sigma_l1, self.conf_sigma_percl

        # if self.conf_map_enabled:
        #     self.loss_l1_im = self.photometric_loss(self.recon_im[:b], self.input_im, mask=detached_mask[:b], conf_sigma=self.conf_sigma_l1[:,:1])
        #     self.loss_l1_im_flip = self.photometric_loss(self.recon_im[b:], self.input_im, mask=detached_mask[b:], conf_sigma=self.conf_sigma_l1[:,1:])
    
        #     if self.use_lpips:
        #         self.loss_perc_im = torch.mean(self.PerceptualLoss.forward(self.recon_im[:b], self.input_im))
        #         self.loss_perc_im_flip = torch.mean(self.PerceptualLoss.forward(self.recon_im[b:],self.input_im))
        #     else:
        #         self.loss_perc_im = self.PerceptualLoss(self.recon_im[:b], self.input_im, mask=detached_mask[:b], conf_sigma=self.conf_sigma_percl[:,:1])
        #         self.loss_perc_im_flip = self.PerceptualLoss(self.recon_im[b:],self.input_im ,mask=detached_mask[:b], conf_sigma=self.conf_sigma_percl[:,1:])
        # else:
        #     self.loss_l1_im = self.photometric_loss(self.recon_im[:b], self.input_im, mask=detached_mask[:b])
        #     self.loss_l1_im_flip = self.photometric_loss(self.recon_im[b:], self.input_im, mask=detached_mask[b:])

        #     if self.use_lpips:
        #         self.loss_perc_im = torch.mean(self.PerceptualLoss.forward(self.recon_im[:b], self.input_im))
        #         self.loss_perc_im_flip = torch.mean(self.PerceptualLoss.forward(self.recon_im[b:],self.input_im))
        #     else:
        #         self.loss_perc_im = self.PerceptualLoss(self.recon_im[:b], self.input_im, mask=detached_mask[:b])
        #         self.loss_perc_im_flip = self.PerceptualLoss(self.recon_im[b:],self.input_im ,mask=detached_mask[:b])

        
        # self.lam_perc = 1 if self.trainer.current_epoch > self.lam_perc_decrease_start_epoch else self.lam_perc 
        # self.loss_total = self.loss_l1_im + self.lam_flip*self.loss_l1_im_flip + self.lam_perc*(self.loss_perc_im + self.lam_flip*self.loss_perc_im_flip)


        # metrics = {'loss': self.loss_total}

        ## compute accuracy if gt depth is available
        # if self.load_gt_depth:
        #     self.depth_gt = depth_gt[:,0,:,:].to(self.input_im.device)
        #     self.depth_gt = (1-self.depth_gt)*2-1
        #     self.depth_gt = self.depth_rescaler(self.depth_gt)
        #     self.normal_gt = self.renderer.get_normal_from_depth(self.depth_gt)

        #     # mask out background
        #     mask_gt = (self.depth_gt<self.depth_gt.max()).float()
        #     mask_gt = (nn.functional.avg_pool2d(mask_gt.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0.99).float()  # erode by 1 pixel
        #     mask_pred = (nn.functional.avg_pool2d(self.recon_im_mask[:b].unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0.99).float()  # erode by 1 pixel
        #     mask = mask_gt * mask_pred
        #     self.acc_mae_masked = ((self.recon_depth[:b] - self.depth_gt[:b]).abs() *mask).view(b,-1).sum(1) / mask.view(b,-1).sum(1)
        #     self.acc_mse_masked = (((self.recon_depth[:b] - self.depth_gt[:b])**2) *mask).view(b,-1).sum(1) / mask.view(b,-1).sum(1)
        #     self.sie_map_masked = utils.compute_sc_inv_err(self.recon_depth[:b].log(), self.depth_gt[:b].log(), mask=mask)
        #     self.acc_sie_masked = (self.sie_map_masked.view(b,-1).sum(1) / mask.view(b,-1).sum(1))**0.5
        #     self.norm_err_map_masked = utils.compute_angular_distance(self.recon_normal[:b], self.normal_gt[:b], mask=mask)
        #     self.acc_normal_masked = self.norm_err_map_masked.view(b,-1).sum(1) / mask.view(b,-1).sum(1)

        #     metrics['SIE_masked'] = self.acc_sie_masked.mean()
        #     metrics['NorErr_masked'] = self.acc_normal_masked.mean()

        # return metrics

    def visualize(self, logger, total_iter, max_bs=25):
        b, c, h, w = self.input_im.shape
        b0 = min(max_bs, b)

        # create shading image
        white_albedo = torch.ones_like(self.canon_albedo).to(self.device)
        new_light = { "ambient": -1*torch.ones_like(self.canon_light_a), "diffuse": torch.ones_like(self.canon_light_b), "direction": self.canon_light_d}
        self.shading_img = self.renderer(self.meshes, white_albedo, self.view, new_light)
        self.shading_img = self.shading_img[...,0].clamp(min=0).unsqueeze(3).permute(0,3,1,2)

        # create side view of shading image
        side_view = utils.get_side_view(self.view)
        self.shading_img_side_view = self.renderer(self.meshes, white_albedo,side_view,new_light)
        self.shading_img_side_view = self.shading_img_side_view[...,0].clamp(min=0).unsqueeze(3).permute(0,3,1,2)

        # sude view zoomed out
        side_view_zoom_out = utils.get_side_view(self.view, zoom_mode=1)
        self.shading_img_side_view_zoom_out = self.renderer(self.meshes, white_albedo,side_view_zoom_out,new_light)
        self.shading_img_side_view_zoom_out = self.shading_img_side_view_zoom_out[...,0].clamp(min=0).unsqueeze(3).permute(0,3,1,2)

        # render rotations for shadding image
        num_rotated_frames = 12
        self.rotated_views = utils.calculate_views_for_360_video(self.view, num_frames=num_rotated_frames).to(self.device)
        shading_img_rotated_video = []
        for i in range(num_rotated_frames):
            shading_img_rotated = self.renderer(self.meshes, white_albedo,self.rotated_views[i], new_light)
            shading_img_rotated = shading_img_rotated[...,0].clamp(min=0).unsqueeze(3).permute(0,3,1,2)
            shading_img_rotated = shading_img_rotated[:b0].detach()
            shading_img_rotated_video.append(shading_img_rotated)
        shading_img_rotated_video = torch.stack(shading_img_rotated_video).permute(1,0,2,3,4)
        
        # render rotations for reconstructed image
        reconstructed_img_rotated_video = []
        for i in range(num_rotated_frames):
            reconstructed_img_rotated = self.renderer(self.meshes,self.canon_albedo,self.rotated_views[i], self.lighting)
            reconstructed_img_rotated = reconstructed_img_rotated[...,:3].permute(0,3,1,2)
            reconstructed_img_rotated = reconstructed_img_rotated[:b0].detach()
            reconstructed_img_rotated_video.append(reconstructed_img_rotated)
        reconstructed_img_rotated_video = torch.stack(reconstructed_img_rotated_video).permute(1,0,2,3,4)

        input_im = self.input_im[:b0].detach().cpu() 
        canon_albedo = self.canon_albedo[:b0].detach().cpu() /2.+0.5
        recon_im = self.recon_im[:b0].detach().cpu()
        recon_im_flip = self.recon_im[b:b+b0].detach().cpu()
        alpha_mask = self.alpha_mask[:b0].detach().cpu()
        shading_im = self.shading_img[:b0].detach().cpu()
        shading_im_side_view = self.shading_img_side_view[:b0].detach().cpu()
        shading_im_side_view_zoom_out = self.shading_img_side_view_zoom_out[:b0].detach().cpu()
        canon_depth_raw_hist = self.canon_depth_raw.detach().unsqueeze(1).cpu()
        canon_depth_raw = self.canon_depth_raw[:b0].detach().unsqueeze(1).cpu() /2.+0.5 # flip(1) is necessary since pytorch3d uses different y axis orientation
        canon_depth = ((self.canon_depth[:b0] -self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().unsqueeze(1)
        canon_light_a = self.canon_light_a/2.+0.5
        canon_light_b = self.canon_light_b/2.+0.5

        shadding_im_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b0**0.5))) for img in torch.unbind(shading_img_rotated_video, 1)]  # [(C,H,W)]*T
        shadding_im_rotate_grid = torch.stack(shadding_im_rotate_grid, 0).unsqueeze(0).cpu()  # (1,T,C,H,W)

        reconstructed_im_rotate_grid = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b0**0.5))) for img in torch.unbind(reconstructed_img_rotated_video, 1)]  # [(C,H,W)]*T
        reconstructed_im_rotate_grid = torch.stack(reconstructed_im_rotate_grid , 0).unsqueeze(0).cpu()  # (1,T,C,H,W)

        ## write summary
        # logger.add_scalar('Loss/loss_total', self.loss_total, total_iter)
        # logger.add_scalar('Loss/loss_l1_im', self.loss_l1_im, total_iter)
        # logger.add_scalar('Loss/loss_l1_im_flip', self.loss_l1_im_flip, total_iter)

        # logger.add_scalar(f'Loss/loss_perc_im', self.loss_perc_im, total_iter)
        # logger.add_scalar(f'Loss/loss_perc_im_flipped', self.loss_perc_im_flip, total_iter)

        logger.add_histogram('Depth/canon_depth_raw_hist', canon_depth_raw_hist, total_iter)
        vlist = ['view_rx', 'view_ry', 'view_rz', 'view_tx', 'view_ty', 'view_tz']
        for i in range(self.view.shape[1]):
            logger.add_histogram('View/'+vlist[i], self.view[:,i], total_iter)
        logger.add_histogram('Light/canon_light_a', canon_light_a, total_iter)
        logger.add_histogram('Light/canon_light_b', canon_light_b, total_iter)
        llist = ['canon_light_dx', 'canon_light_dy', 'canon_light_dz']
        for i in range(self.canon_light_d.shape[1]):
            logger.add_histogram('Light/'+llist[i], self.canon_light_d[:,i], total_iter)

        def log_grid_image(label, im, nrow=int(math.ceil(b0**0.5)), iter=total_iter):
            im_grid = torchvision.utils.make_grid(im, nrow=nrow)
            logger.add_image(label, im_grid, iter)

        log_grid_image('Image/input_image', input_im)
        log_grid_image('Image/canonical_albedo', canon_albedo)
        log_grid_image('Image/recon_image', recon_im)
        log_grid_image('Image/recon_image_flip', recon_im_flip)
        log_grid_image('Image/alpha_mask', alpha_mask)
        log_grid_image('Depth/canonical_depth_raw', canon_depth_raw)
        log_grid_image('Depth/canonical_depth', canon_depth)
        log_grid_image('Depth/diffuse_shading', shading_im)
        log_grid_image('Depth/diffuse_shading_side_view', shading_im_side_view)
        log_grid_image('Depth/diffuse_shading_side_view_zoom_out', shading_im_side_view_zoom_out)

        logger.add_histogram('Image/canonical_albedo_hist', canon_albedo, total_iter)

        if self.conf_map_enabled:
            conf_map_l1 = 1/(1+self.conf_sigma_l1[:b0,:1].detach().cpu()+EPS)
            conf_map_l1_flip = 1/(1+self.conf_sigma_l1[:b0,1:].detach().cpu()+EPS)
            conf_map_percl = 1/(1+self.conf_sigma_percl[:b0,:1].detach().cpu()+EPS)
            conf_map_percl_flip = 1/(1+self.conf_sigma_percl[:b0,1:].detach().cpu()+EPS)

            log_grid_image('Conf/conf_map_l1', conf_map_l1)
            logger.add_histogram('Conf/conf_sigma_l1_hist', self.conf_sigma_l1[:,:1], total_iter)
            log_grid_image('Conf/conf_map_l1_flip', conf_map_l1_flip)
            logger.add_histogram('Conf/conf_sigma_l1_flip_hist', self.conf_sigma_l1[:,1:], total_iter)
            log_grid_image('Conf/conf_map_percl', conf_map_percl)
            logger.add_histogram('Conf/conf_sigma_percl_hist', self.conf_sigma_percl[:,:1], total_iter)
            log_grid_image('Conf/conf_map_percl_flip', conf_map_percl_flip)
            logger.add_histogram('Conf/conf_sigma_percl_flip_hist', self.conf_sigma_percl[:,1:], total_iter)

        logger.add_video('Image_rotate/shadding_rotate', shadding_im_rotate_grid, total_iter, fps=2)
        logger.add_video('Image_rotate/recon_rotate', reconstructed_im_rotate_grid, total_iter, fps=2)

        # visualize images and accuracy if gt is loaded
        if self.load_gt_depth:
            depth_gt = ((self.depth_gt[:b0] -self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().unsqueeze(1)
            normal_gt = self.normal_gt.permute(0,3,1,2)[:b0].detach().cpu() /2+0.5
            sie_map_masked = self.sie_map_masked[:b0].detach().unsqueeze(1).cpu() *1000
            norm_err_map_masked = self.norm_err_map_masked[:b0].detach().unsqueeze(1).cpu() /100

            logger.add_scalar('Acc_masked/MAE_masked', self.acc_mae_masked.mean(), total_iter)
            logger.add_scalar('Acc_masked/MSE_masked', self.acc_mse_masked.mean(), total_iter)
            logger.add_scalar('Acc_masked/SIE_masked', self.acc_sie_masked.mean(), total_iter)
            logger.add_scalar('Acc_masked/NorErr_masked', self.acc_normal_masked.mean(), total_iter)

            log_grid_image('Depth_gt/depth_gt', depth_gt)
            log_grid_image('Depth_gt/normal_gt', normal_gt)
            log_grid_image('Depth_gt/sie_map_masked', sie_map_masked)
            log_grid_image('Depth_gt/norm_err_map_masked', norm_err_map_masked)

    def save_results(self, save_dir):
        b, c, h, w = self.input_im.shape

        input_im = self.input_im[:b].detach().cpu().numpy() /2+0.5
        canon_albedo = self.canon_albedo[:b].detach().cpu().numpy() /2+0.5
        recon_im = self.recon_im[:b].clamp(-1,1).detach().cpu().numpy() 
        recon_im_flip = self.recon_im[b:].clamp(-1,1).detach().cpu().numpy() 
        canon_depth = ((self.canon_depth[:b] -self.min_depth)/(self.max_depth-self.min_depth)).clamp(0,1).detach().cpu().unsqueeze(1).numpy()
        canon_light = torch.cat([self.canon_light_a, self.canon_light_b, self.canon_light_d], 1)[:b].detach().cpu().numpy()
        view = self.view[:b].detach().cpu().numpy()

        sep_folder = True
        utils.save_images(save_dir, input_im, suffix='input_image', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_albedo, suffix='canonical_albedo', sep_folder=sep_folder)
        utils.save_images(save_dir, recon_im, suffix='recon_image', sep_folder=sep_folder)
        utils.save_images(save_dir, recon_im_flip, suffix='recon_image_flip', sep_folder=sep_folder)
        utils.save_images(save_dir, canon_depth, suffix='canonical_depth', sep_folder=sep_folder)
        utils.save_txt(save_dir, canon_light, suffix='canonical_light', sep_folder=sep_folder)
        utils.save_txt(save_dir, view, suffix='viewpoint', sep_folder=sep_folder)

        if self.conf_map_enabled:
            conf_map_l1 = 1/(1+self.conf_sigma_l1[:b,:1].detach().cpu().numpy()+EPS)
            conf_map_l1_flip = 1/(1+self.conf_sigma_l1[:b,1:].detach().cpu().numpy()+EPS)
            conf_map_percl = 1/(1+self.conf_sigma_percl[:b,:1].detach().cpu().numpy()+EPS)
            conf_map_percl_flip = 1/(1+self.conf_sigma_percl[:b,1:].detach().cpu().numpy()+EPS)
            utils.save_images(save_dir, conf_map_l1, suffix='conf_map_l1', sep_folder=sep_folder)
            utils.save_images(save_dir, conf_map_l1_flip, suffix='conf_map_l1_flip', sep_folder=sep_folder)
            utils.save_images(save_dir, conf_map_percl, suffix='conf_map_percl', sep_folder=sep_folder)
            utils.save_images(save_dir, conf_map_percl_flip, suffix='conf_map_percl_flip', sep_folder=sep_folder)


        # save scores if gt is loaded
        if self.load_gt_depth:
            depth_gt = ((self.depth_gt[:b] -self.min_depth)/(self.max_depth-self.min_depth)).clamp(0,1).detach().cpu().unsqueeze(1).numpy()
            normal_gt = self.normal_gt[:b].permute(0,3,1,2).detach().cpu().numpy() /2+0.5
            utils.save_images(save_dir, depth_gt, suffix='depth_gt', sep_folder=sep_folder)
            utils.save_images(save_dir, normal_gt, suffix='normal_gt', sep_folder=sep_folder)

            all_scores = torch.stack([
                self.acc_mae_masked.detach().cpu(),
                self.acc_mse_masked.detach().cpu(),
                self.acc_sie_masked.detach().cpu(),
                self.acc_normal_masked.detach().cpu()], 1)
            if not hasattr(self, 'all_scores'):
                self.all_scores = torch.FloatTensor()
            self.all_scores = torch.cat([self.all_scores, all_scores], 0)

    def save_scores(self, path):
        # save scores if gt is loaded
        if self.load_gt_depth:
            header = 'MAE_masked, \
                      MSE_masked, \
                      SIE_masked, \
                      NorErr_masked'
            mean = self.all_scores.mean(0)
            std = self.all_scores.std(0)
            header = header + '\nMean: ' + ',\t'.join(['%.8f'%x for x in mean])
            header = header + '\nStd: ' + ',\t'.join(['%.8f'%x for x in std])
            utils.save_scores(path, self.all_scores, header=header)
