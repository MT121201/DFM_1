import torch 
from torch import nn
from DenoisingDiffisionProcess.beta_schedules import *

class ForwardModel(nn.Module):
    """
    Forward process of the denoising diffusion process
    """
    def __init__(self, 
                 num_timesteps,
                 schedule = 'linear'):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.schedule = schedule    

    @torch.no_grad()
    def forward(self,x_0, t):
        """
            Get noisy sample at t given x_0
        """
        raise NotImplemented
        
    def step(self, x_t, t):
        """
            Get next sample in the process
        """
        raise NotImplemented 

class GaussianForwardProcess(ForwardModel):
    """
    Forward process of the denoising diffusion process
    """
    def __init__(self, 
                 num_timesteps=1000,
                 schedule = 'linear'):
        super().__init__(num_timesteps=num_timesteps,
                         schedule=schedule)
        # get process parameters
        # regis as buffer for not be update by optimizer
        self.register_buffer('betas', get_beta_schedule(schedule, num_timesteps))
        self.register_buffer('betas_sqrt', self.betas.sqrt())
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_sqrt', self.alphas_cumprod.sqrt())
        self.register_buffer('alphas_one_minus_cumprod_sqrt', (1-self.alphas_cumprod).sqrt())
        self.register_buffer('alpha_sqrt', self.alphas.sqrt())

    @torch.no_grad()
    def forward(self, x_0, t, return_noise =False):
        """
        Get noisy sample at t given x_0
        q(x_t | x_0)=N(x_t; alphas_cumprod_sqrt(t)*x_0, 1-alpha_cumprod(t)*I)
        """
        assert (t<self.num_timesteps).all()

        b = x_0.shape[0]
        mean = x_0 * self.alphas_cumprod_sqrt[t].view(b,1,1,1)
        std = self.alphas_one_minus_cumprod_sqrt[t].view(b,1,1,1)
        noise = torch.randn_like(x_0)
        out = mean + std * noise
        if return_noise:
            return out, noise
        return out
    
    @torch.no_grad()
    def step(self, x_t, t, return_noise=False):
        """
            Get next sample in the process
            
            q(x_t | x_t-1)=N(x_t; alphas_sqrt(t)*x_t,betas(t)*I)
        """
        assert (t<self.num_timesteps).all()

        mean = x_t * self.alpha_sqrt[t]
        std = self.betas_sqrt[t]
        noise = torch.randn_like(x_t)
        out = mean + std * noise
        if return_noise:
            return out, noise