"""
This is DDIM sampler class for a diffusion process
"""

import torch
from torch import nn
from ..beta_schedules import *

class DDIM_Sampler(nn.Module):
    def __init__(self,
                 num_timesteps=1000,
                 train_timesteps =  1000,
                 clip_sample = True,
                 schedule='linear'):
        super().__init__()

        self.num_timesteps=num_timesteps
        self.train_timesteps=train_timesteps
        self.ratio=self.train_timesteps // self.num_timesteps
        self.final_alpha_cumprod=torch.tensor([1.0])
        self.clip_sample=clip_sample
        self.schedule=schedule

        self.register_buffer('betas',get_beta_schedule(self.schedule,self.train_timesteps))
        self.register_buffer('betas_sqrt',self.betas.sqrt())
        self.register_buffer('alphas',1-self.betas)
        self.register_buffer('alphas_cumprod',torch.cumprod(self.alphas,0))
        self.register_buffer('alphas_cumprod_sqrt',self.alphas_cumprod.sqrt())
        self.register_buffer('alphas_one_minus_cumprod_sqrt',(1-self.alphas_cumprod).sqrt())
        self.register_buffer('alphas_sqrt',self.alphas.sqrt())
        self.register_buffer('alphas_sqrt_recip',1/(self.alphas_sqrt))

    @torch.no_grad()
    def forward(self,*args,**kwargs):   
        return self.step(*args,**kwargs)
    
    @torch.no_grad()
    def step(self, x_t, t, z_t, eta=0):
        """
        Given approximation of noise z_t in x_t predict x_(t-1)
        """
        assert (t<self.num_timesteps).all()

        b,c,h,w = z_t.shape
        device = z_t.device 

        # compute step base on ratio (number of DDPM step in each DDIM)
        t = t * self.ratio
        t_prev = t - self.ratio

        #params
        alpha_cumprod_prev = self.alphas_cumprod[t_prev].where(t_prev.ge(0), self.final_alpha_cumprod.to(device)) #>=0
        alpha_cumprod_prev = alpha_cumprod_prev.view(b,1,1,1)
        alpha_cumprod_prev_sqrt = self.alpha_cumprod_prev_sqrt[t_prev]

        # estimate origin
        x_0_pred = self.estimate_original(x_t, t, z_t)
        if self.clip_sample:
            x_0_pred = torch.clamp(x_0_pred, -1, 1)

        std_dev_t = eta * self.estimate_std(t, t_prev).view(b,1,1,1)
        x_0_grad = (1- alpha_cumprod_prev -std_dev_t ** 2).sqrt() * z_t
        prev_sample = alpha_cumprod_prev_sqrt * x_0_pred + x_0_grad

        if eta > 0:
            noise = torch.randn(prev_sample.shape, dtype=prev_sample.dtype)
            prev_sample = prev_sample + eta * noise * std_dev_t

        return prev_sample

    def estimate_std(self, t, t_prev):
        alpha_cumprod = self.alphas_cumprod[t]
        alpha_cumprod_prev = self.alphas_cumprod[t_prev].where(t_prev.ge(0), self.final_alpha_cumprod.to(alpha_cumprod.device))
        one_minus_alpha_cumprod = 1 - alpha_cumprod
        one_minus_alpha_cumprod_prev = 1 - alpha_cumprod_prev
        var = (one_minus_alpha_cumprod_prev/one_minus_alpha_cumprod) * (1-alpha_cumprod/alpha_cumprod_prev)
        return var.sqrt()

    def estimate_original(self,x_t, t, z_t):
        alpha_cumprod = self.alphas_cumprod[t].view(z_t.shape[0],1,1,1)
        alpha_one_minus_cumprod_sqrt =self.alphas_one_minus_cumprod_sqrt[t]
        return(x_t-alpha_one_minus_cumprod_sqrt*z_t)/alpha_cumprod.sqrt()