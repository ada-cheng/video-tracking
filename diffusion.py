import torch
import torch.nn.functional as F
import time
from tqdm import tqdm, trange
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize



def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)



timesteps = 1000

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0) # cumulative product
#alphas_cumprod.shape = (timesteps,) 
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # shift right and pad with 1.0 F是torch.nn.functional


sqrt_recip_alphas = torch.sqrt(1.0 / alphas) #意思是alphas的倒数的开方

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # posterior_variance[t] = beta_t*(1-alpha_{t-1})/(1-alpha_t)

def extract(a, t, x_shape):
    # extract a tensor from a batch of tensors
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())# gather的意思是从a中取出t中的元素 out.shape = (batch_size, 1) out[i] = a[i][t[i]]
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)



# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    # t is a tensor of shape (batch_size,)
    
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape) #sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t]
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise #alpha_t*x_start + sqrt(1-alpha_t)*noise

def get_noisy_image(x_start, t):
  # add noise
  x_noisy = q_sample(x_start, t=t) 
 
  return x_noisy



def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    # 先采样噪声
    if noise is None:
        noise = torch.randn_like(x_start)
   

    # 用采样得到的噪声去加噪图片
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    # the model is mlp
    predicted_noise = denoise_model(x_noisy) # the model predicts the noise
    
    # 根据加噪了的图片去预测采样的噪声
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

@torch.no_grad()
def p_sample(model, x, t, t_index):
    x = x.reshape(x.shape[0], -1)
    betas_t = extract(betas, t, x.shape)
  
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
 
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x) / sqrt_one_minus_alphas_cumprod_t
    )
   

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    
    device = next(model.parameters()).device

    b = shape[0]
   #equals to t finally
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []
    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size)) 

