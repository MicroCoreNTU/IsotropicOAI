from typing import Dict, List, Optional, Union, Any

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import yaml
import torchvision.models as models
import torchvision.transforms as transforms
import os

from networks.networks import get_scheduler
from networks.loss import GANLoss
from networks.registry import network_registry
from utils.data_utils import *


def _weights_init(m: nn.Module) -> None:
    """Initialize network weights using normal distribution.
    
    Args:
        m: PyTorch module whose weights will be initialized
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


class Vgg19(torch.nn.Module):
    """VGG19 model for feature extraction used in perceptual loss calculations.
    
    This class implements a modified VGG19 network that extracts features from
    different layers of the network for use in perceptual loss calculations.
    """
    
    def __init__(self, requires_grad=False):
        """Initialize the VGG19 model with pretrained weights.
        
        Args:
            requires_grad: Whether to compute gradients for the model parameters
        """
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        """Extract features from different layers of the VGG19 network.
        
        Args:
            X: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            List of feature maps from different layers of the network
        """
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    """Perceptual loss using VGG19 features.
    
    This loss computes the L1 distance between features extracted by the VGG19
    network from the generated and target images, weighted by importance.
    """
    
    def __init__(self):
        """Initialize the VGG loss with a pretrained VGG19 model."""
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        """Calculate the weighted VGG perceptual loss.
        
        Args:
            x: Generated image tensor
            y: Target image tensor
            
        Returns:
            Weighted sum of L1 losses between VGG features
        """
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class BaseModel(pl.LightningModule):
    """Base model for GAN training using PyTorch Lightning.
    
    This class provides the foundation for implementing various GAN architectures
    by handling common operations like optimization, loss calculation, checkpointing,
    and training/validation loops.
    """
    
    def __init__(self, hparams: Any, train_loader: Any, eval_loader: Any, checkpoints: str) -> None:
        """Initialize the base GAN model.
        
        Args:
            hparams: Hyperparameters for the model
            train_loader: DataLoader for training data
            eval_loader: DataLoader for evaluation data
            checkpoints: Directory to save model checkpoints
        """
        super().__init__()
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.epoch = 0
        self.dir_checkpoints = checkpoints

        # Model and loss names
        self.netg_names = {'net_g': 'netG'}
        self.netd_names = {'net_d': 'netD'}
        self.loss_g_names = ['loss_g']
        self.loss_d_names = ['loss_d']

        # Process hyperparameters
        hparams_dict = {x: vars(hparams)[x] for x in vars(hparams).keys() 
                       if x not in hparams.not_tracking_hparams}
        hparams_dict.pop('not_tracking_hparams', None)
        self.hparams.update(hparams_dict)
        self.save_hyperparameters(self.hparams)

        # Initialize loss functions
        self._init_loss_functions()

        # Final
        self.hparams.update(vars(self.hparams))   # updated hparams to be logged in tensorboard
        #self.train_loader.dataset.shuffle_images()  # !!! shuffle again just to make sure
        #self.train_loader.dataset.shuffle_images()  # !!! shuffle again just to make sure

        self.all_label = []
        self.all_out = []
        self.all_loss = []

        self.log_image = {}

        self.buffer = {}

    def _init_loss_functions(self) -> None:
        """Initialize loss functions used in the model.
        
        This method should be overridden by subclasses to initialize
        specific loss functions needed for their implementation.
        """
        self.criterionL1 = nn.L1Loss()
        self.criterionL2 = nn.MSELoss()
        if self.hparams.gan_mode == 'vanilla':
            self.criterionGAN = nn.BCEWithLogitsLoss()
        else:
            self.criterionGAN = GANLoss(self.hparams.gan_mode)

    def save_tensor_to_png(self, tensor, path):
        """Save a tensor as a PNG image.
        
        Args:
            tensor: Image tensor to save
            path: Path where the image will be saved
        """
        if len(tensor.shape) == 4:  # B, C, H, W
            if tensor.shape[1] == 1:
                tensor = tensor.repeat(1, 3, 1, 1)
            tensor = tensor[0].detach().cpu()
        elif len(tensor.shape) == 3:  # C, H, W
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            tensor = tensor.detach().cpu()

        tensor = (tensor + 1) / 2.0  # [-1, 1] -> [0, 1]
        tensor = torch.clamp(tensor, 0, 1)

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        transforms.ToPILImage()(tensor).save(path)
        return path

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            Dictionary containing optimizers and schedulers for generator and discriminator
        """
        # Generator optimizer
        self.optimizer_g = optim.Adam(getattr(self, list(self.netg_names.keys())[0]).parameters(),
                                      lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        
        # Discriminator optimizer
        self.optimizer_d = optim.Adam(getattr(self, list(self.netd_names.keys())[0]).parameters(),
                                      lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        
        # Learning rate schedulers
        self.net_g_scheduler = get_scheduler(self.optimizer_g, self.hparams)
        self.net_d_scheduler = get_scheduler(self.optimizer_d, self.hparams)
        
        return [self.optimizer_g, self.optimizer_d], []

    def add_loss_adv(self, a, net_d, truth):
        """Calculate adversarial loss.
        
        Args:
            a: Generated image tensor
            net_d: Discriminator network
            truth: Whether the target is real (True) or fake (False)
            
        Returns:
            Adversarial loss value
        """
        if isinstance(truth, bool):
            b = torch.ones_like(a) if truth else torch.zeros_like(a)
        else:
            b = truth
            
        pred = net_d(a)
        return self.criterionGAN(pred, b)

    def add_loss_l1(self, a, b):
        """Calculate L1 loss between two tensors.
        
        Args:
            a: First tensor
            b: Second tensor
            
        Returns:
            L1 loss value
        """
        return torch.mean(torch.abs(a - b))

    def add_loss_l2(self, a, b):
        """Calculate L2 (MSE) loss between two tensors.
        
        Args:
            a: First tensor
            b: Second tensor
            
        Returns:
            L2 loss value
        """
        return torch.mean((a - b) ** 2)

    def save_auc_csv(self, auc, epoch):
        """Save AUC metrics to a CSV file.
        
        Args:
            auc: AUC value to save
            epoch: Current epoch number
        """
        import csv
        path = self.dir_checkpoints + '/auc.csv'
        with open(path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, auc])

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Perform a single training step.
        
        Args:
            batch: Batch of training data
            batch_idx: Index of the current batch
            optimizer_idx: Index of the optimizer to use (0 for generator, 1 for discriminator)
            
        Returns:
            Loss value for the current step
        """
        # Train generator
        if optimizer_idx == 0:
            # Set discriminator gradients to False
            self.set_requires_grad(getattr(self, list(self.netd_names.keys())[0]), False)
            
            # Calculate generator loss
            loss_g = self.backward_g()
            for k in list(loss_g.keys()):
                if k != 'sum':
                    self.log(k, loss_g[k], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            return loss_g['sum']
        
        # Train discriminator
        elif optimizer_idx == 1:
            # Set discriminator gradients to True
            self.set_requires_grad(getattr(self, list(self.netd_names.keys())[0]), True)
            
            # Calculate discriminator loss
            loss_d = self.backward_d()
            for k in list(loss_d.keys()):
                if k != 'sum':
                    self.log(k, loss_d[k], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            return loss_d['sum']
        else:
            return None

    def training_epoch_end(self, outputs):
        """Perform operations at the end of each training epoch.
        
        Args:
            outputs: List of outputs from training_step
        """
        # Save checkpoints
        if self.epoch % self.hparams.epoch_save == 0:
            for name in self.netg_names.keys():
                path_g = self.dir_checkpoints + ('/' + self.netg_names[name] + '_model_epoch_{}.pth').format(self.epoch)
                torch.save(getattr(self, name), path_g)
                print("Checkpoint saved to {}".format(path_g))

            if self.hparams.save_d:
                for name in self.netd_names.keys():
                    path_d = self.dir_checkpoints + ('/' + self.netd_names[name] + '_model_epoch_{}.pth').format(self.epoch)
                    torch.save(getattr(self, name), path_d)
                    print("Checkpoint saved to {}".format(path_d))

        # Step learning rate schedulers
        self.net_g_scheduler.step()
        self.net_d_scheduler.step()

        # Save generated images
        for k in self.log_image.keys():
            self.save_tensor_to_png(self.log_image[k], self.dir_checkpoints + os.path.join(str(self.epoch).zfill(4) + k + '.png'))

        # Reset metrics and increment epoch counter
        self.reset_metrics()
        self.epoch += 1

    def get_metrics(self):
        """Get evaluation metrics.
        
        This method should be overridden by subclasses to return
        specific evaluation metrics for their implementation.
        """
        pass

    def reset_metrics(self):
        """Reset evaluation metrics.
        
        This method should be overridden by subclasses to reset
        specific evaluation metrics for their implementation.
        """
        pass

    def testing_step(self, batch, batch_idx):
        """Perform a single testing step.
        
        Args:
            batch: Batch of test data
            batch_idx: Index of the current batch
        """
        self.generation(batch)

    def validation_epoch_end(self, x):
        """Perform operations at the end of each validation epoch.
        
        Args:
            x: List of outputs from validation_step
            
        Returns:
            None
        """
        #self.log_helper.print(logger=self.logger, epoch=self.epoch)
        #self.log_helper.clear()
        return None

    def get_progress_bar_dict(self):
        """Customize the progress bar displayed during training.
        
        Returns:
            Dictionary of values to display in the progress bar
        """
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        if 'loss' in tqdm_dict:
            del tqdm_dict['loss']
        return tqdm_dict

    def generation(self, batch):
        """Generate outputs for a batch of data.
        
        This method should be overridden by subclasses to implement
        the generation process for their specific model architecture.
        
        Args:
            batch: Batch of data to generate outputs for
        """
        pass

    def backward_g(self):
        """Calculate generator losses.
        
        This method should be overridden by subclasses to implement
        the loss calculation for the generator.
        
        Returns:
            Dictionary of generator losses
        """
        pass

    def backward_d(self):
        """Calculate discriminator losses.
        
        This method should be overridden by subclasses to implement
        the loss calculation for the discriminator.
        
        Returns:
            Dictionary of discriminator losses
        """
        pass

    def set_networks(self, net='all'):
        """Create network instances using the network registry.
        
        Args:
            net: Which networks to return ('all', 'g', or 'd')
            
        Returns:
            Generator and/or discriminator networks based on the net parameter
        """
        # Common parameters for both networks
        common_params = {
            'input_nc': self.hparams.input_nc,
            'output_nc': self.hparams.output_nc,
            'norm': self.hparams.norm,
            'norm_layer': nn.BatchNorm2d
        }
        
        # Generator-specific parameters
        generator_params = {
            **common_params,
            'ngf': self.hparams.ngf,
            'use_dropout': self.hparams.mc,
            'final': self.hparams.final,
            'mc': self.hparams.mc
        }
        
        # Discriminator-specific parameters
        discriminator_params = {
            **common_params,
            'ndf': self.hparams.ndf
        }
        
        # Create networks
        net_g = network_registry.get_generator(self.hparams.netG, **generator_params)
        net_d = network_registry.get_discriminator(self.hparams.netD, **discriminator_params)
        
        # Return requested networks
        if net == 'all':
            return net_g, net_d
        elif net == 'g':
            return net_g
        elif net == 'd':
            return net_d

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad flag for network parameters.
        
        This method controls whether gradients are computed for the specified networks,
        which is useful for selectively training parts of the model.
        
        Args:
            nets: Single network or list of networks
            requires_grad: Whether to compute gradients for the networks
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
