import torch
from torch.optim import Optimizer

class COCOB(Optimizer):
    """Implemention COCOB Algorithm.
    It has been proposed in 'Training Deep Networks without Learning Rates Through Coin Betting'.
    Authors - F. Orabona, T. Tommasi.
    https://arxiv.org/abs/1705.07795

    Args:
        params (torch.nn.Parameter): Parameters of the network.
        alpha (float): Scale of maximum observed range of gradients.
        eps (float): Epsilon value to stop.
    """

    def __init__(self, params, alpha=100, eps=1e-9):
        if not 0.0 < alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(alpha=alpha, eps=eps)

        super(COCOB, self).__init__(params, defaults)


    def step(self, closure=None):
        """Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss=closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Get negative stochastic subgradient grad
                grad = p.grad.data

                state = self.state[p]

                # Initialize the state
                if len(state) == 0:
                    state['max_obs_scale'] = group['eps']*torch.ones_like(p.data, 
                                                  memory_format=torch.preserve_format)
                    state['sum_abs_grad'] = torch.zeros_like(p.data, 
                                                  memory_format=torch.preserve_format)
                    state['reward'] = torch.zeros_like(p.data,
                                                  memory_format=torch.preserve_format)
                    state['sum_grad'] = torch.zeros_like(p.data, 
                                                  memory_format=torch.preserve_format)
                    state['w'] = torch.zeros_like(p.data,
                                                  memory_format=torch.preserve_format)

                alpha = group['alpha']
                max_obs_scale = state['max_obs_scale']
                sum_abs_grad = state['sum_abs_grad']
                reward = state['reward']
                sum_grad = state['sum_grad']
                w_ = state['w']

                # Update the maximum observed scale
                torch.max(input=max_obs_scale, other=torch.abs(grad), out=max_obs_scale)
                state['max_obs_scale'] = max_obs_scale

                # Update the sub of the absolute values of subgradients
                sum_abs_grad += torch.abs(grad)
                state['sum_abs_grad'] = sum_abs_grad
                
                # Update the reward
                torch.max(input=(reward - grad*w_), other=torch.FloatTensor([0.]), out=reward)
                state['reward'] = reward
                
                # Update the sum of the gradients
                sum_grad += grad
                state['sum_grad'] = sum_grad

                # Calculate the parameters
                w = -(sum_grad/(max_obs_scale*torch.max(sum_abs_grad+max_obs_scale, alpha*max_obs_scale)))*(max_obs_scale+reward)
                p.data = p.data - w_ + w
                state['w'] = w

        return loss

