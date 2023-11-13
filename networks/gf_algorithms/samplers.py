import sys
import os
import torch
import numpy as np

from scipy import integrate
from ipdb import set_trace
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.genpose_utils import get_pose_dim
from utils.misc import normalize_rotation


def global_prior_likelihood(z, sigma_max):
    """The likelihood of a Gaussian distribution with mean zero and 
        standard deviation sigma."""
    # z: [bs, pose_dim]
    shape = z.shape
    N = np.prod(shape[1:]) # pose_dim
    return -N / 2. * torch.log(2*np.pi*sigma_max**2) - torch.sum(z**2, dim=-1) / (2 * sigma_max**2)


def cond_ode_likelihood(
        score_model,
        data,
        prior,
        sde_coeff,
        marginal_prob_fn,
        atol=1e-5, 
        rtol=1e-5, 
        device='cuda', 
        eps=1e-5,
        num_steps=None,
        pose_mode='quat_wxyz', 
        init_x=None,
    ):
    
    pose_dim = get_pose_dim(pose_mode)
    batch_size = data['pts'].shape[0]
    epsilon = prior((batch_size, pose_dim)).to(device)
    init_x = data['sampled_pose'].clone().cpu().numpy() if init_x is None else init_x
    shape = init_x.shape
    init_logp = np.zeros((shape[0],)) # [bs]
    init_inp = np.concatenate([init_x.reshape(-1), init_logp], axis=0)
    
    def score_eval_wrapper(data):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_model(data)
        return score.cpu().numpy().reshape((-1,))

    def divergence_eval(data, epsilon):      
        """Compute the divergence of the score-based model with Skilling-Hutchinson."""
        # save ckpt of sampled_pose
        origin_sampled_pose = data['sampled_pose'].clone()
        with torch.enable_grad():
            # make sampled_pose differentiable
            data['sampled_pose'].requires_grad_(True)
            score = score_model(data)
            score_energy = torch.sum(score * epsilon) # [, ]
            grad_score_energy = torch.autograd.grad(score_energy, data['sampled_pose'])[0] # [bs, pose_dim]
        # reset sampled_pose
        data['sampled_pose'] = origin_sampled_pose
        return torch.sum(grad_score_energy * epsilon, dim=-1) # [bs, 1]
    
    def divergence_eval_wrapper(data):
        """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
        with torch.no_grad(): 
            # Compute likelihood.
            div = divergence_eval(data, epsilon) # [bs, 1]
        return div.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    def ode_func(t, inp):        
        """The ODE function for use by the ODE solver."""
        # split x, logp from inp
        x = inp[:-shape[0]]
        logp = inp[-shape[0]:] # haha, actually we do not need use logp here
        # calc x-grad
        x = torch.tensor(x.reshape(-1, pose_dim), dtype=torch.float32, device=device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        drift, diffusion = sde_coeff(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        data['sampled_pose'] = x
        data['t'] = time_steps
        x_grad = drift - 0.5 * (diffusion**2) * score_eval_wrapper(data)
        # calc logp-grad
        logp_grad = drift - 0.5 * (diffusion**2) * divergence_eval_wrapper(data)
        # concat curr grad
        return  np.concatenate([x_grad, logp_grad], axis=0)
  
    # Run the black-box ODE solver, note the 
    res = integrate.solve_ivp(ode_func, (eps, 1.0), init_inp, rtol=rtol, atol=atol, method='RK45')
    zp = torch.tensor(res.y[:, -1], device=device) # [bs * (pose_dim + 1)]
    z = zp[:-shape[0]].reshape(shape) # [bs, pose_dim]
    delta_logp = zp[-shape[0]:].reshape(shape[0]) # [bs,] logp
    _, sigma_max = marginal_prob_fn(None, torch.tensor(1.).to(device)) # we assume T = 1 
    prior_logp = global_prior_likelihood(z, sigma_max)
    log_likelihoods = (prior_logp + delta_logp) / np.log(2) # negative log-likelihoods (nlls)
    return z, log_likelihoods


def cond_pc_sampler(
        score_model, 
        data,
        prior,
        sde_coeff,
        num_steps=500, 
        snr=0.16,                
        device='cuda',
        eps=1e-5,
        pose_mode='quat_wxyz',
        init_x=None,
    ):
    
    pose_dim = get_pose_dim(pose_mode)
    batch_size = data['pts'].shape[0]
    init_x = prior((batch_size, pose_dim)).to(device) if init_x is None else init_x
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    noise_norm = np.sqrt(pose_dim) 
    x = init_x
    poses = []
    with torch.no_grad():
        for time_step in time_steps:      
            batch_time_step = torch.ones(batch_size, device=device).unsqueeze(-1) * time_step
            # Corrector step (Langevin MCMC)
            data['sampled_pose'] = x
            data['t'] = batch_time_step
            grad = score_model(data)
            grad_norm = torch.norm(grad.reshape(batch_size, -1), dim=-1).mean()
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)  

            # normalisation
            if pose_mode == 'quat_wxyz' or pose_mode == 'quat_xyzw':
                # quat, should be normalised
                x[:, :4] /= torch.norm(x[:, :4], dim=-1, keepdim=True)   
            elif pose_mode == 'euler_xyz':
                pass
            else:
                # rotation(x axis, y axis), should be normalised
                x[:, :3] /= torch.norm(x[:, :3], dim=-1, keepdim=True)
                x[:, 3:6] /= torch.norm(x[:, 3:6], dim=-1, keepdim=True)

            # Predictor step (Euler-Maruyama)
            drift, diffusion = sde_coeff(batch_time_step)
            drift = drift - diffusion**2*grad # R-SDE
            mean_x = x + drift * step_size
            x = mean_x + diffusion * torch.sqrt(step_size) * torch.randn_like(x)
            
            # normalisation
            x[:, :-3] = normalize_rotation(x[:, :-3], pose_mode)
            poses.append(x.unsqueeze(0))
    
    xs = torch.cat(poses, dim=0)
    xs[:, :, -3:] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
    mean_x[:, -3:] += data['pts_center']
    mean_x[:, :-3] = normalize_rotation(mean_x[:, :-3], pose_mode)
    # The last step does not include any noise
    return xs.permute(1, 0, 2), mean_x 


def cond_ode_sampler(
        score_model,
        data,
        prior,
        sde_coeff,
        atol=1e-5, 
        rtol=1e-5, 
        device='cuda', 
        eps=1e-5,
        T=1.0,
        num_steps=None,
        pose_mode='quat_wxyz', 
        denoise=True,
        init_x=None,
    ):
    pose_dim = get_pose_dim(pose_mode)
    batch_size=data['pts'].shape[0]
    init_x = prior((batch_size, pose_dim), T=T).to(device) if init_x is None else init_x + prior((batch_size, pose_dim), T=T).to(device)
    shape = init_x.shape
    
    def score_eval_wrapper(data):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_model(data)
        return score.cpu().numpy().reshape((-1,))
    
    def ode_func(t, x):      
        """The ODE function for use by the ODE solver."""
        x = torch.tensor(x.reshape(-1, pose_dim), dtype=torch.float32, device=device)
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        drift, diffusion = sde_coeff(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        data['sampled_pose'] = x
        data['t'] = time_steps
        return drift - 0.5 * (diffusion**2) * score_eval_wrapper(data)
  
    # Run the black-box ODE solver, note the 
    t_eval = None
    if num_steps is not None:
        # num_steps, from T -> eps
        t_eval = np.linspace(T, eps, num_steps)
    res = integrate.solve_ivp(ode_func, (T, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45', t_eval=t_eval)
    xs = torch.tensor(res.y, device=device).T.view(-1, batch_size, pose_dim) # [num_steps, bs, pose_dim]
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape) # [bs, pose_dim]
    # denoise, using the predictor step in P-C sampler
    if denoise:
        # Reverse diffusion predictor for denoising
        vec_eps = torch.ones((x.shape[0], 1), device=x.device) * eps
        drift, diffusion = sde_coeff(vec_eps)
        data['sampled_pose'] = x.float()
        data['t'] = vec_eps
        grad = score_model(data)
        drift = drift - diffusion**2*grad       # R-SDE
        mean_x = x + drift * ((1-eps)/(1000 if num_steps is None else num_steps))
        x = mean_x
    
    num_steps = xs.shape[0]
    xs = xs.reshape(batch_size*num_steps, -1)
    xs[:, :-3] = normalize_rotation(xs[:, :-3], pose_mode)
    xs = xs.reshape(num_steps, batch_size, -1)
    xs[:, :, -3:] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
    x[:, :-3] = normalize_rotation(x[:, :-3], pose_mode)
    x[:, -3:] += data['pts_center']
    return xs.permute(1, 0, 2), x


def cond_edm_sampler(
    decoder_model, data, prior_fn, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    pose_mode='quat_wxyz', device='cuda'
):
    pose_dim = get_pose_dim(pose_mode)
    batch_size = data['pts'].shape[0]
    latents = prior_fn((batch_size, pose_dim)).to(device)

    # Time step discretization. note that sigma and t is interchangable
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    def decoder_wrapper(decoder, data, x, t):
        # save temp
        x_, t_= data['sampled_pose'], data['t']
        # init data
        data['sampled_pose'], data['t'] = x, t
        # denoise
        data, denoised = decoder(data)
        # recover data
        data['sampled_pose'], data['t'] = x_, t_
        return denoised.to(torch.float64)

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    xs = []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = torch.as_tensor(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = decoder_wrapper(decoder_model, data, x_hat, t_hat)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = decoder_wrapper(decoder_model, data, x_next, t_next)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        xs.append(x_next.unsqueeze(0))

    xs = torch.stack(xs, dim=0) # [num_steps, bs, pose_dim]
    x = xs[-1] # [bs, pose_dim]

    # post-processing
    xs = xs.reshape(batch_size*num_steps, -1)
    xs[:, :-3] = normalize_rotation(xs[:, :-3], pose_mode)
    xs = xs.reshape(num_steps, batch_size, -1)
    xs[:, :, -3:] += data['pts_center'].unsqueeze(0).repeat(xs.shape[0], 1, 1)
    x[:, :-3] = normalize_rotation(x[:, :-3], pose_mode)
    x[:, -3:] += data['pts_center']

    return xs.permute(1, 0, 2), x


