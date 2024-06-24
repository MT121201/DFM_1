import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from diffusers.models import AutoencoderKL

from .DenoisingDiffusionProcess import *

class AutoEncoder(nn.Module):
    def __init__(self,
                 model_type ="stabilityai/sd-vae-ft-ema"#@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
                 ):
        """
            A wrapper for an AutoEncoder model
            
            By default, a pretrained AutoencoderKL is used from stabilitai
            
            A custom AutoEncoder could be trained and used with the same interface.
            Yet, this model works quite well for many tasks out of the box!
        """

        super().__init__()
        self.model = AutoencoderKL.from_pretrained(model_type)
    
    def forward(self, x):
        return self.model(x).sample
    
    def encode(self, x, mode=False):
        dist = self.model.encode(x).latent_dist
        if mode:
            return dist.mode()
        else:
            return dist.sample()
        
    def decode(self, z):
        return self.model.decode(z).sample
    
class LatentDiffusion(pl.LightningModule):
    def __init__(self,
                 train_dataset,
                 valid_dataset=None,
                 num_timesteps=1000,
                 latent_scale_factor=0.1,
                 batch_size = 1,
                 lr = 1e-4):
        """
        This is very simple version
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr =lr
        self.register_buffer('latent_scale_factor', torch.tensor(latent_scale_factor))
        self.batch_size = batch_size

        self.ae = AutoEncoder()

        with torch.no_grad():
            self.latent_dim = self.ae.encode(torch.one(1,3,256,256)).shape[1]

        self.model = DenoisingDiffusionProcess(generated_channels=self.latent_dim,
                                               num_timesteps=num_timesteps)
        
    @torch.no_grad()
    def forward(self, *args, **kwargs):
        # return self.model(*args, **kwargs)
        return self.output_T(self.ae.decode(self.model(*args, **kwargs)/self.latent_scale_factor))
    
    def input_T(self, input):
        # by default, model acept input in [01] range and transform them
        return (input.clip(0,1).mul(2)).sub(1)
    
    def output_T(self, output):
        return (output.add(1)).div(2)
    
    def training_step(self, batch, batch_idx):
        latents = self.ae.encode(self.input_T(batch)).detach()*self.latent_scale_factor
        loss = self.model.p_loss(latents)

        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        latents = self.ae.encode(self.input_T(batch)).detach()*self.latent_scale_factor
        loss = self.model.p_loss(latents)

        self.log('valid_loss', loss)
        return loss
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    def val_dataloader(self):
        if self.valid_dataset is not None:
            return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        else:
            return None
        
    def configure_optimizers(self):
        return torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.model.parameters())), lr=self.lr)
    
class LatentDiffusionConditional(LatentDiffusion):
    def __init__(self,
                 train_dataset,
                 valid_dataset=None,
                 num_timesteps=1000,
                 latent_scale_factor=0.1,
                 batch_size = 1,
                 lr = 1e-4):
        """
        This is very simple version
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr =lr
        self.register_buffer('latent_scale_factor', torch.tensor(latent_scale_factor))
        self.batch_size = batch_size

        self.ae = AutoEncoder()

        with torch.no_grad():
            self.latent_dim = self.ae.encode(torch.one(1,3,256,256)).shape[1]

        self.model = DenoisingDiffusionConditionalProcess(generated_channels=self.latent_dim,
                                                            condition_channels=self.latent_dim,
                                                            num_timesteps=num_timesteps)
        
    @torch.no_grad()
    def forward(self, condition, *args, **kwargs):
        condition_latent = self.ae.encode(self.input_T(condition).to(self.device)).detach()*self.latent_scale_factor
        output_code = self.model(condition_latent, *args, **kwargs)/self.latent_scale_factor
        return self.output_T(self.ae.decode(output_code))
    
    def traning_step(self, batch, batch_idx):
        condition, output =batch
        with torch.no_grad():
            latents = self.ae.encode(self.input_T(output)).detach()*self.latent_scale_factor
            letents_condition = self.ae.encode(self.input_T(condition)).detach()*self.latent_scale_factor
        loss = self.model.p_loss(latents, letents_condition)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        condition, output =batch
        with torch.no_grad():
            latents = self.ae.encode(self.input_T(output)).detach()*self.latent_scale_factor
            letents_condition = self.ae.encode(self.input_T(condition)).detach()*self.latent_scale_factor
            loss = self.model.p_loss(latents, letents_condition)
        self.log('val_loss', loss)
        return loss
    

        