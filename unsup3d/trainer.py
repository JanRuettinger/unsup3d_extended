import os, json
import pathlib
from datetime import datetime
import torch
from . import meters
from . import utils
from . import losses
from . import networks
from .dataloaders import get_data_loaders
import lpips
from PIL import Image
import numpy as np


class Trainer():
    def __init__(self, cfgs, Generator, Discriminator):
        self.device = cfgs.get('device', 'cpu')
        self.num_epochs = cfgs.get('num_epochs', 30)
        self.batch_size = cfgs.get('batch_size', 64)
        self.checkpoint_dir = cfgs.get('checkpoint_dir', 'results')
        self.save_checkpoint_freq = cfgs.get('save_checkpoint_freq', 1)
        self.keep_num_checkpoint = cfgs.get('keep_num_checkpoint', 2)  # -1 for keeping all checkpoints
        self.resume = cfgs.get('resume', True)
        self.use_logger = cfgs.get('use_logger', True)
        self.log_freq = cfgs.get('log_freq', 1000)
        self.archive_code = cfgs.get('archive_code', True)
        self.checkpoint_name = cfgs.get('checkpoint_name', None)
        self.test_result_dir = cfgs.get('test_result_dir', None)
        self.cfgs = cfgs
        self.lam_perc = cfgs.get('lam_perc', 1)
        self.lam_flip = cfgs.get('lam_flip', 0.5)
        self.discriminator_loss = cfgs.get('discriminator_loss', 0.1)
        self.conf_map_enabled = cfgs.get('conf_map_enabled', True)
        self.use_lpips = cfgs.get('use_lpips', False)
        self.discriminator_loss_start_epoch = cfgs.get('discriminator_loss_start_epoch', False)
        self.lam_perc_decrease_start_epoch = cfgs.get('lam_perc_decrease_start_epoch', 2)
        self.discriminator_loss_type = cfgs.get('discriminator_loss_type', "bce")

        self.metrics_trace = meters.MetricsTrace()
        self.make_metrics = lambda m=None: meters.StandardMetrics(m)
        self.generator = Generator(cfgs)
        self.generator.trainer = self
        self.discriminator = Discriminator(cfgs)
        self.discriminator.trainer = self
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(cfgs)
        loss_fn = lpips.LPIPS(net='alex')

        if self.use_lpips:
            self.PerceptualLoss = loss_fn.to(device=self.device)
        else:
            self.PerceptualLoss = networks.PerceptualLoss().to(device=self.device)
        #         self.loss_perc_im = self.PerceptualLoss(self.recon_im[:b], self.input_im, mask=detached_mask[:b], conf_sigma=self.conf_sigma_percl[:,:1])
        #         self.loss_perc_im_flip = self.PerceptualLoss(self.recon_im[b:],self.input_im ,mask=detached_mask[:b], conf_sigma=self.conf_sigma_percl[:,1:])


    # def load_checkpoint(self, optim=True):
    #     """Search the specified/latest checkpoint in checkpoint_dir and load the model and optimizer."""
    #     if self.checkpoint_name is not None:
    #         checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)
    #     else:
    #         checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, '*.pth')))
    #         if len(checkpoints) == 0:
    #             return 0
    #         checkpoint_path = checkpoints[-1]
    #         self.checkpoint_name = os.path.basename(checkpoint_path)
    #     print(f"Loading checkpoint from {checkpoint_path}")
    #     cp = torch.load(checkpoint_path, map_location=self.device)
    #     print(cp) # doesn't work 
    #     self.generator.load_model_state(cp) # probably doesn;t work like that
    #     self.discriminator.load_model_state(cp) # probably doesn;t work like that
    #     if optim:
    #         self.model.load_optimizer_state(cp)
    #     self.metrics_trace = cp['metrics_trace']
    #     epoch = cp['epoch']
    #     return epoch

    # def save_checkpoint(self, epoch, optim=True):
    #     """Save model, optimizer, and metrics state to a checkpoint in checkpoint_dir for the specified epoch."""
    #     utils.xmkdir(self.checkpoint_dir)
    #     checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint{epoch:03}.pth')
    #     state_dict = self.model.get_model_state()
    #     if optim:
    #         optimizer_state = self.model.get_optimizer_state()
    #         state_dict = {**state_dict, **optimizer_state}
    #     state_dict['metrics_trace'] = self.metrics_trace
    #     state_dict['epoch'] = epoch
    #     print(f"Saving checkpoint to {checkpoint_path}")
    #     torch.save(state_dict, checkpoint_path)
    #     if self.keep_num_checkpoint > 0:
    #         utils.clean_checkpoint(self.checkpoint_dir, keep_num=self.keep_num_checkpoint)

    def save_clean_checkpoint(self, path):
        """Save model state only to specified path."""
        torch.save(self.model.get_model_state(), path)

    def test(self):
        """Perform testing."""
        self.model.to_device(self.device)
        # self.current_epoch = self.load_checkpoint(optim=False)
        if self.test_result_dir is None:
            self.test_result_dir = os.path.join(self.checkpoint_dir, f'test_results_{self.checkpoint_name}'.replace('.pth',''))
        print(f"Saving testing results to {self.test_result_dir}")

        with torch.no_grad():
            m = self.run_epoch(self.test_loader, epoch=self.current_epoch, is_test=True)

        score_path = os.path.join(self.test_result_dir, 'eval_scores.txt')
        self.model.save_scores(score_path)

    def train(self):
        """Perform training."""
        ## archive code and configs
        if self.archive_code:
            utils.archive_code(os.path.join(self.checkpoint_dir, 'archived_code.zip'), filetypes=['.py', '.yml'])
        utils.dump_yaml(os.path.join(self.checkpoint_dir, 'configs.yml'), self.cfgs)

        ## initialize
        start_epoch = 0
        self.metrics_trace.reset()
        self.train_iter_per_epoch = len(self.train_loader)
        self.generator.to_device(self.device)
        self.generator.init_optimizers()
        self.discriminator.to_device(self.device)
        self.discriminator.init_optimizers()

        ## resume from checkpoint
        if self.resume:
            pass
            # start_epoch = self.load_checkpoint(optim=True)

        ## initialize tensorboardX logger
        if self.use_logger:
            from tensorboardX import SummaryWriter
            self.logger = SummaryWriter(os.path.join(self.checkpoint_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")))
            self.logger.add_text("config", json.dumps(self.cfgs))

            ## cache one batch for visualization
            self.val_loader.__iter__().__next__() # skip first
            self.viz_input = self.val_loader.__iter__().__next__()

        ## run epochs
        print(f"Optimizing to {self.num_epochs} epochs")
        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch
            if epoch % self.lam_perc_decrease_start_epoch == 0:
                self.lam_perc *= 0.80
            metrics = self.run_epoch(self.train_loader, epoch)
            self.metrics_trace.append("train", metrics)

            with torch.no_grad():
                metrics = self.run_epoch(self.val_loader, epoch, is_validation=True)
                self.metrics_trace.append("val", metrics)

            if (epoch+1) % self.save_checkpoint_freq == 0:
                pass
                # self.save_checkpoint(epoch+1, optim=True)
            self.metrics_trace.plot(pdf_path=os.path.join(self.checkpoint_dir, 'metrics.pdf'))
            self.metrics_trace.save(os.path.join(self.checkpoint_dir, 'metrics.json'))

        print(f"Training completed after {epoch+1} epochs.")

    def run_epoch(self, loader, epoch=0, is_validation=False, is_test=False):
        """Run one epoch."""
        is_train = not is_validation and not is_test
        metrics = self.make_metrics()

        if is_train:
            print(f"Starting training epoch {epoch}")
            self.generator.set_train()
            self.discriminator.set_train()
        else:
            print(f"Starting validation epoch {epoch}")
            self.generator.set_eval()
            self.discriminator.set_eval()

        for iter, input in enumerate(loader):
            if is_train:
                m_gen = self.train_step_generator(input, epoch)
                m_dis = self.train_step_discriminator(input)
            if is_validation:
                m_gen = self.validation_step_generator(input, epoch)
                m_dis = self.validation_step_discriminator(input)
                if iter < 5:
                    # save predictions
                    self.save_example_predictions(input, iter)
            
            m = {**m_gen, **m_dis}
            metrics.update(m, self.batch_size)
            print(f"{'T' if is_train else 'V'}{epoch:02}/{iter:05}/{metrics}")
            if self.use_logger and is_train:
                total_iter = iter + epoch*self.train_iter_per_epoch
                if total_iter % self.log_freq == 0:
                    m_gen = self.validation_step_generator(self.viz_input, epoch)
                    m_dis = self.validation_step_discriminator(self.viz_input)
                    self.log_loss_generator(self.logger,m_gen,total_iter)
                    self.log_loss_discriminator(self.logger,m_dis,total_iter)
                    self.generator.visualize(self.logger, total_iter=total_iter, max_bs=25)
                    self.visualization_helper_discriminator(self.viz_input, self.logger, total_iter=total_iter, max_bs=25)
                    # self.discriminator.visualize(self.logger, total_iter=total_iter, max_bs=25)

        return metrics

    def log_loss_generator(self,logger,metrics, total_iter):
        loss_total = metrics["loss"]
        loss_l1_im = metrics["loss_l1_im"]
        loss_perc_im = metrics["loss_perc_im"]
        gloss = metrics["gloss"]
        logger.add_scalar('Loss_Gen/loss_total', loss_total, total_iter)
        logger.add_scalar('Loss_Gen/loss_l1_im', loss_l1_im, total_iter)
        logger.add_scalar('Loss_Gen/loss_perc_im', loss_perc_im, total_iter)
        logger.add_scalar('Loss_Gen/loss_generator', gloss, total_iter)
    
    def log_loss_discriminator(self,logger,metrics, total_iter):
        d_loss = metrics["d_loss"]
        d_loss_fake = metrics["d_loss_fake"]
        d_loss_real = metrics["d_loss_real"]
        logger.add_scalar('Loss_Dis/d_loss', d_loss, total_iter)
        logger.add_scalar('Loss_Dis/d_loss_real', d_loss_real, total_iter)
        logger.add_scalar('Loss_Dis/d_loss_fake', d_loss_fake, total_iter)

    def train_step_generator(self,input, epoch):
        generator = self.generator
        discriminator = self.discriminator
        input_im = input.to(self.device)
        b  = input_im.shape[0] # batch_size

        generator.toggle_grad(True)
        discriminator.toggle_grad(False)
        generator.set_train()
        discriminator.set_train() # train/eval mode -> no difference since dis doesn't use BN

        fake_recon_im, recon_im_mask, conf_sigma_l1, conf_sigma_percl = generator.forward(input)

        # print recon_im and mask
        # detached_x_fake = fake_recon_im.detach().permute(0,2,3,1)[0].cpu().numpy()*255
        # img = Image.fromarray(np.uint8(detached_x_fake)).convert('RGB')
        # img.save('recon.png')

        # detached_x_fake = fake_recon_im.detach().permute(0,2,3,1)[b+1].cpu().numpy()*255
        # img = Image.fromarray(np.uint8(detached_x_fake)).convert('RGB')
        # img.save('recon_flipped.png')

        # detached_im_mask = recon_im_mask.detach().permute(0,2,3,1)[0].cpu().numpy()*255
        # img = Image.fromarray(np.uint8(detached_im_mask[:,:,0]))
        # img.save('im_mask.png')

        # detached_im_mask = recon_im_mask.detach().permute(0,2,3,1)[b+1].cpu().numpy()*255
        # img = Image.fromarray(np.uint8(detached_im_mask[:,:,0]))
        # img.save('im_mask_flipped.png')

        # masked_input_im = input_im*recon_im_mask[:b] + (1-recon_im_mask[:b])
        # detached_input_im = masked_input_im.permute(0,2,3,1)
        # detached_input_im = detached_input_im.detach()[0].cpu().numpy()*255
        # img = Image.fromarray(np.uint8(detached_input_im)).convert('RGB')
        # img.save('masked_img.png')

        fake_recon_im_disc = torch.clamp(fake_recon_im,0,1)*2 -1 
        d_fake = discriminator.forward(fake_recon_im_disc)
        if self.discriminator_loss_type == "bce":
            gloss = losses.compute_bce(d_fake, 1)
        else:
            gloss = losses.compute_lse(d_fake, 1)

        # input_im wrong

        if self.conf_map_enabled:
            loss_l1_im = losses.photometric_loss(fake_recon_im[:b], input_im, mask=recon_im_mask[:b], conf_sigma=conf_sigma_l1[:,:1])
            loss_l1_im_flip = losses.photometric_loss(fake_recon_im[b:], input_im, mask=recon_im_mask[b:], conf_sigma=conf_sigma_l1[:,1:])
        else:
            loss_l1_im = losses.photometric_loss(fake_recon_im[:b], input_im, mask=recon_im_mask[:b])
            loss_l1_im_flip = losses.photometric_loss(fake_recon_im[b:], input_im, mask=recon_im_mask[b:])

        if self.use_lpips:
            loss_perc_im = torch.mean(self.PerceptualLoss.forward(fake_recon_im[:b], input_im))
            loss_perc_im_flip = torch.mean(self.PerceptualLoss.forward(fake_recon_im[b:],input_im))
        else:
            if self.conf_map_enabled:
                loss_perc_im = self.PerceptualLoss(fake_recon_im[:b], input_im, mask=recon_im_mask[:b], conf_sigma=conf_sigma_percl[:,:1])
                loss_perc_im_flip = self.PerceptualLoss(fake_recon_im[b:],input_im ,mask=recon_im_mask[:b], conf_sigma=conf_sigma_percl[:,1:])
            else:
                loss_perc_im = self.PerceptualLoss(fake_recon_im[:b], input_im, mask=recon_im_mask[:b])
                loss_perc_im_flip = self.PerceptualLoss(fake_recon_im[b:],input_im ,mask=recon_im_mask[:b])

        loss_conventional = loss_l1_im + self.lam_flip*loss_l1_im_flip + self.lam_perc*(loss_perc_im + self.lam_flip*loss_perc_im_flip)
        if epoch > self.discriminator_loss_start_epoch:
            loss_total = loss_conventional + self.discriminator_loss*gloss
        else:
            gloss *= 0
            loss_total = loss_conventional


        generator.reset_optimizer()
        loss_total.backward()
        generator.optimizer_step()

        # if self.generator_test is not None:
        #     update_average(self.generator_test, generator, beta=0.999)

        metrics = {'loss': loss_total.item(), "loss_l1_im": loss_l1_im.item(), "loss_l1_flip": loss_l1_im_flip.item(), "loss_perc_im": loss_perc_im.item(),"loss_perc_im_flip": loss_perc_im_flip.item(), "gloss": gloss.item() }
        return metrics
    
    def validation_step_generator(self,input, epoch):
        generator = self.generator
        discriminator = self.discriminator
        input_im = input.to(self.device)
        b  = input_im.shape[0] # batch_size

        generator.toggle_grad(False) 
        discriminator.toggle_grad(False) 
        generator.set_eval()
        discriminator.set_eval()
        fake_recon_im, recon_im_mask, conf_sigma_l1, conf_sigma_percl = generator.forward(input)

        fake_recon_im_disc = torch.clamp(fake_recon_im,0,1)*2 -1 
        d_fake = discriminator.forward(fake_recon_im_disc)
        if self.discriminator_loss_type == "bce":
            gloss = losses.compute_bce(d_fake, 1)
        else:
            gloss = losses.compute_lse(d_fake, 1)

        # input_im wrong
        if self.conf_map_enabled:
            loss_l1_im = losses.photometric_loss(fake_recon_im[:b], input_im, mask=recon_im_mask[:b], conf_sigma=conf_sigma_l1[:,:1])
            loss_l1_im_flip = losses.photometric_loss(fake_recon_im[b:], input_im, mask=recon_im_mask[b:], conf_sigma=conf_sigma_l1[:,1:])
        else:
            loss_l1_im = losses.photometric_loss(fake_recon_im[:b], input_im, mask=recon_im_mask[:b])
            loss_l1_im_flip = losses.photometric_loss(fake_recon_im[b:], input_im, mask=recon_im_mask[b:])

        if self.use_lpips:
            loss_perc_im = torch.mean(self.PerceptualLoss.forward(fake_recon_im[:b], input_im))
            loss_perc_im_flip = torch.mean(self.PerceptualLoss.forward(fake_recon_im[b:],input_im))
        else:
            if self.conf_map_enabled:
                loss_perc_im = self.PerceptualLoss(fake_recon_im[:b], input_im, mask=recon_im_mask[:b], conf_sigma=conf_sigma_percl[:,:1])
                loss_perc_im_flip = self.PerceptualLoss(fake_recon_im[b:],input_im ,mask=recon_im_mask[:b], conf_sigma=conf_sigma_percl[:,1:])
            else:
                loss_perc_im = self.PerceptualLoss(fake_recon_im[:b], input_im, mask=recon_im_mask[:b])
                loss_perc_im_flip = self.PerceptualLoss(fake_recon_im[b:],input_im ,mask=recon_im_mask[:b])

        loss_conventional = loss_l1_im + self.lam_flip*loss_l1_im_flip + self.lam_perc*(loss_perc_im + self.lam_flip*loss_perc_im_flip)

        if epoch > self.discriminator_loss_start_epoch:
            loss_total = loss_conventional + self.discriminator_loss*gloss
        else:
            gloss *= 0
            loss_total = loss_conventional

        metrics = {'loss': loss_total.item(), "loss_l1_im": loss_l1_im.item(), "loss_l1_flip": loss_l1_im_flip.item(), "loss_perc_im": loss_perc_im.item(),"loss_perc_im_flip": loss_perc_im_flip.item(), "gloss": gloss.item() }
        return metrics

    def train_step_discriminator(self,input):
        generator = self.generator
        discriminator = self.discriminator
        generator.toggle_grad(False)
        discriminator.toggle_grad(True)
        generator.set_train()
        discriminator.set_train()
        d_full_loss = 0.
        b = input.shape[0]

        input_im = input.to(self.device)
        random_view = torch.rand(b, 6)
        random_view[:,3:] = 0
        random_view[:,:2] = 0
        random_view[:,2] -= 0.5
        x_fake, recon_im_mask, _, _ = generator.forward(input_im, random_view) # no grad loop -> no grad reuqired here

        # print recon_im and mask
        # detached_x_fake = x_fake.detach().permute(0,2,3,1)[0].cpu().numpy()*255
        # img = Image.fromarray(np.uint8(detached_x_fake)).convert('RGB')
        # img.save('random_view_fakes.png')

        x_fake = torch.clamp(x_fake,0,1)*2 -1 
        d_fake = discriminator.forward(x_fake)
        if self.discriminator_loss_type == "bce":
            d_loss_fake = losses.compute_bce(d_fake, 0)
        else:
            d_loss_fake = losses.compute_lse(d_fake, 0)

        input_with_flipped = torch.cat([input_im, input_im.flip(2)], 0)  # flip
        masked_input_im = input_with_flipped*recon_im_mask + (1-recon_im_mask)
        masked_input_im = torch.clamp(masked_input_im, 0, 1) *2 -1
        d_real = discriminator.forward(masked_input_im)
        if self.discriminator_loss_type == "bce":
            d_loss_real = losses.compute_bce(d_real, 1)
        else:
            d_loss_real = losses.compute_lse(d_real, 1)

        # input_im.requires_grad_() # QUESTION: Why required true on input tensor?
        # reg = 10. * losses.compute_grad2(d_real, input_im).mean()
        # d_full_loss += reg

        d_full_loss += d_loss_fake + d_loss_real

        discriminator.reset_optimizer()
        d_full_loss.backward()
        discriminator.optimizer_step()
        
        # return (d_loss.item(), reg.item(), d_loss_fake.item(), d_loss_real.item())
        metrics = {"d_loss": d_full_loss.item(), "d_loss_fake": d_loss_fake.item(), "d_loss_real": d_loss_real.item()}
        return metrics


    def validation_step_discriminator(self, input):
        generator = self.generator
        discriminator = self.discriminator
        generator.toggle_grad(False)
        discriminator.toggle_grad(False)
        generator.set_eval()
        discriminator.set_eval()

        input_im = input.to(self.device)

        x_fake, recon_im_mask,_, _ = generator.forward(input)
        x_fake = torch.clamp(x_fake,0,1)*2 -1
        d_fake = discriminator.forward(x_fake)
        if self.discriminator_loss_type == "bce":
            d_loss_fake = losses.compute_bce(d_fake, 0)
        else:
            d_loss_fake = losses.compute_lse(d_fake, 0)

        # x_real.requires_grad_()
        input_with_flipped = torch.cat([input_im, input_im.flip(2)], 0)  # flip
        masked_input_im = input_with_flipped*recon_im_mask + (1-recon_im_mask)
        masked_input_im = torch.clamp(masked_input_im, 0, 1) *2 -1
        d_real = discriminator.forward(masked_input_im)
        if self.discriminator_loss_type == "bce":
            d_loss_real = losses.compute_bce(d_real, 1)
        else:
            d_loss_real = losses.compute_lse(d_real, 1)

        # reg = 10. * losses.compute_grad2(d_real, input_im).mean()
        # loss_d_full += reg

        d_loss = d_loss_fake + d_loss_real

        # return (d_loss.item(), reg.item(), d_loss_fake.item(), d_loss_real.item())
        metrics = {"d_loss": d_loss.item(), "d_loss_fake": d_loss_fake.item(), "d_loss_real": d_loss_real.item()}
        return metrics


    def visualization_helper_discriminator(self, input, logger, total_iter, max_bs=25):
        generator = self.generator
        discriminator = self.discriminator
        generator.toggle_grad(False)
        discriminator.toggle_grad(False)
        generator.set_eval()
        discriminator.set_eval()

        input_im = input.to(self.device)

        x_fake, recon_im_mask,_, _ = generator.forward(input)
        discriminator.forward(x_fake)
        discriminator.visualize(logger, total_iter, fake=True)

        # x_real.requires_grad_()
        input_with_flipped = torch.cat([input_im, input_im.flip(2)], 0)  # flip
        masked_input_im = input_with_flipped*recon_im_mask + (1-recon_im_mask)
        discriminator.forward(masked_input_im)
        discriminator.visualize(logger, total_iter, fake=False)


    def save_example_predictions(self, input, iter):
        b  = input.shape[0] # batch_siz
        recon_im, recon_im_mask,_, _ = self.generator.forward(input)
        img_dir = pathlib.Path(self.checkpoint_dir) / 'imgs'
        img_dir.mkdir(parents=True, exist_ok=True) 

        # print recon_im and mask
        detached_mask = recon_im.detach().permute(0,2,3,1).cpu().numpy()*255
        for key,value in enumerate(detached_mask):
            img = Image.fromarray(np.uint8(value)).convert('RGB')
            img_path = img_dir / f'recon_im_{iter}_{key}.png'
            img.save(img_path)

        # detached_im_mask = recon_im_mask.detach().permute(0,2,3,1)[0].cpu().numpy()*255
        # img = Image.fromarray(np.uint8(detached_im_mask[:,:,0]))
        # img.save('im_mask.png')

