import torch
import torch.nn as nn

class Mask():
    def __init__(self, hard):
        self.hard = hard
        if hard is None:
            self.n_keep = 0
        else:
            self.n_keep = int(torch.sum(hard).item())
        if self.n_keep == 0:
            self.nonzero_hard = None
        else:
            # this code only support batchsize=1 while inference, 
            # nonzero calculation need to be updated in the future
            self.nonzero_hard = torch.nonzero(hard[0][0], as_tuple=True)


class Gumbel(nn.Module):
    ''' 
    Returns differentiable discrete outputs. Applies a Gumbel-Softmax trick on every element of x. 
    '''
    def __init__(self, eps=1e-8):
        super(Gumbel, self).__init__()
        self.eps = eps

    def forward(self, x, gumbel_temp=1.0, gumbel_noise=True):
        if not self.training:  # no Gumbel noise during inference
            hard = (x >= 0).float() 
            ans = Mask(hard)
            return ans

        if gumbel_noise:
            eps = self.eps
            U1, U2 = torch.rand_like(x), torch.rand_like(x)
            g1, g2 = -torch.log(-torch.log(U1 + eps)+eps), - \
                torch.log(-torch.log(U2 + eps)+eps)
            x = x + g1 - g2

        soft = torch.sigmoid(x / gumbel_temp)
        hard = ((soft >= 0.5).float() - soft).detach() + soft
        assert not torch.any(torch.isnan(hard))
        ans = Mask(hard)
        return ans