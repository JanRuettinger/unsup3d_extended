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


class Trainer:
    def __init__(self, cfgs, model):
        self.device = cfgs.get('device', 'cpu')
        self.num_epochs = cfgs.get('num_epochs', 30)
        self.batch_size = cfgs.get('batch_size', 64)
        self.checkpoint_dir = cfgs.get('checkpoint_dir', 'results')
        self.save_checkpoint_freq = cfgs.get('save_checkpoint_freq', 1)
        self.keep_num_checkpoint = cfgs.get('keep_num_checkpoint', 2)  # -1 for keeping all checkpoints
        self.resume = cfgs.get('resume', True)
        self.use_logger = cfgs.get('use_logger', True)
        self.log_freq_train = cfgs.get('log_freq_train', 1000)
        self.log_freq_val = cfgs.get('log_freq_val', 1000)
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
        self.view_loss_weight = cfgs.get('view_loss_weight', 1)

        self.metrics_trace = meters.MetricsTrace()
        self.make_metrics = lambda m=None: meters.StandardMetrics(m)
        self.model = model(cfgs)
        self.model.trainer = self
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
        self.val_iter_per_epoch = len(self.val_loader)
        self.model.to_device(self.device)
        self.model.init_optimizers()

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
            # self.val_loader.__iter__().__next__() # skip first
            # self.train_loader.__iter__().__next__() # skip first
            self.viz_input_val = self.val_loader.__iter__().__next__()
            self.viz_input_train = self.train_loader.__iter__().__next__()

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
            # self.metrics_trace.plot(pdf_path=os.path.join(self.checkpoint_dir, 'metrics.pdf'))
            # self.metrics_trace.save(os.path.join(self.checkpoint_dir, 'metrics.json'))

        print(f"Training completed after {epoch+1} epochs.")

    def run_epoch(self, loader, epoch=0, is_validation=False, is_test=False):
        """Run one epoch."""
        is_train = not is_validation and not is_test
        metrics = self.make_metrics()

        if is_train:
            print(f"Starting training epoch {epoch}")
            self.model.set_train()
        else:
            print(f"Starting validation epoch {epoch}")
            self.model.set_eval()

        if is_train:
            for iter, input in enumerate(loader):
                # if iter > 100:
                #     break
                # self.model.reset_optimizer()
                m = self.model.forward(input)
                self.model.backward(epoch)
                metrics.update(m, self.batch_size)
                print(f"T{epoch:02}/{iter:05}/{metrics}")
                if self.use_logger:
                    total_iter = iter + epoch*self.train_iter_per_epoch
                    if total_iter % self.log_freq_train == 0:
                        with torch.no_grad():
                        #if self.log_fix_sample:
                            self.model.forward(self.viz_input_train)
                        self.model.visualize(self.logger, total_iter=total_iter,is_train=True,max_bs=25)

        if is_validation:
            with torch.no_grad():
                validation_metrics = {}
                total_iter = (epoch+1)*self.train_iter_per_epoch
                for iter, input in enumerate(loader):
                    m1 = self.model.forward(input)
                    m2 = self.model.backward_without_paramter_update(epoch)
                    m = {**m1, **m2}
                    metrics.update(m, self.batch_size)
                    print(f"V{epoch:02}/{iter:05}/{metrics}")
                    for key, item in m.items():
                        if key in validation_metrics:
                            validation_metrics[key] += item
                        else:
                            validation_metrics[key] = item

                for key in validation_metrics:
                    validation_metrics[key] /= (iter+1)

                with torch.no_grad():
                    self.model.forward(self.viz_input_val)
                self.model.visualize(self.logger, total_iter=total_iter, is_train=False, max_bs=25)

        return metrics

    # def save_example_predictions(self, input, iter):
    #     b  = input.shape[0] # batch_siz
    #     recon_im, recon_im_mask,_, _, _ = self.generator.forward(input)
    #     img_dir = pathlib.Path(self.checkpoint_dir) / 'imgs'
    #     img_dir.mkdir(parents=True, exist_ok=True) 

    #     # print recon_im and mask
    #     detached_im = recon_im.detach().permute(0,2,3,1).cpu().numpy()*255
    #     for key,value in enumerate(detached_im):
    #         img = Image.fromarray(np.uint8(value)).convert('RGB')
    #         img_path = img_dir / f'recon_im_{iter}_{key}.png'
    #         img.save(img_path)