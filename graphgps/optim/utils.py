import torch
from torch.autograd import Function
from torch.nn import Module

from torch_geometric.utils import degree
from torch_geometric.nn.aggr import SumAggregation


class _GRADTracker(Function):

    @staticmethod
    def forward(ctx, x, norm_buffer):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        grad_norm = grad_output.norm('fro') ** 2
        return grad_output, grad_norm


class _GRADTrackerWithBackwardWeight(Function):

    @staticmethod
    def forward(ctx, x, norm_buffer, grad_weight):
        ctx.save_for_backward(norm_buffer, grad_weight)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        norm_buffer, grad_weight = ctx.saved_tensors
        grad_norm = grad_output.norm('fro') ** 2
        grad_output.mul_(grad_weight)
        return grad_output, grad_norm, None


class _GradClipIn(Function):

    @staticmethod
    def forward(ctx, x, norm_buffer):
        ctx.save_for_backward(norm_buffer)
        return x, norm_buffer

    @staticmethod
    def backward(ctx, grad_output, grad_norm_out):
        grad_norm_in = grad_output.norm('fro')
        max_norm, =  ctx.saved_tensors
        max_norm.mul_(grad_norm_out)
        #max_norm.clamp_(0.0, grad_norm_in)
        scale = max_norm / (grad_norm_in + 1.0e-12)
        #scale.clamp_(0.0, 1.0)
        grad_output.mul_(scale)
        return grad_output, None


class _GradClipOut(Function):

    @staticmethod
    def forward(ctx, x, norm_buffer):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_norm_out = grad_output.norm('fro')
        return grad_output, grad_norm_out



_grad_tracker = _GRADTracker.apply
_grad_tracker_with_weight = _GRADTrackerWithBackwardWeight.apply
_grad_clip_in = _GradClipIn.apply
_grad_clip_out = _GradClipOut.apply


class GRADTracker(Module):

    def __init__(self, key='grad_l2', device=None):
        super(GRADTracker, self).__init__()
        self.key = key
        self.register_buffer('norm_buffer', torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device))

    def get_last_norm(self):
        return torch.sqrt(self.norm_buffer.grad)

    def forward(self, x):
        with torch.no_grad():
            self.norm_buffer.grad = None
        self.norm_buffer.retain_grad()
        return _grad_tracker(x, self.norm_buffer)



class GradClipper(Module):

    def __init__(self, max_rel_norm=1.0, device=None):
        super(GradClipper, self).__init__()
        self.max_rel_norm = max_rel_norm
        self.register_buffer('norm_buffer', torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device))

    def set_input(self, x):
        with torch.no_grad():
            self.norm_buffer.grad = None
        #self.norm_buffer.retain_grad()
        max_norm = torch.tensor(self.max_rel_norm, dtype=torch.float32, requires_grad=True, device=x.device)
        x, self.norm_buffer = _grad_clip_in(x, max_norm)
        return x

    def set_output(self, x):
        #self.norm_buffer.retain_grad()
        return _grad_clip_out(x, self.norm_buffer)


class GradScaler(Module):

    def __init__(self, scale=1.0, alpha=0.999, eps=1e-12, device=None, **kwargs):
        super(GradScaler, self).__init__(**kwargs)
        self.scale = scale
        self.alpha = alpha
        self.eps = eps

        self.register_buffer('exp_grad_in', torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device))
        self.register_buffer('exp_grad_out', torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device))
        self.register_buffer('exp_ratio', torch.tensor(1.0, dtype=torch.float32, requires_grad=False, device=device))

        self.steps = 0

    def step(self):
        with torch.no_grad():
            grad_in = self.exp_grad_in.grad
            grad_out = self.exp_grad_out.grad

            if grad_in is not None and grad_out is not None:
                ratio = grad_in / (grad_out + self.eps)

                grad_in.sqrt_()
                grad_out.sqrt_()

                if self.steps >= 0:
                    self.exp_grad_in.mul_(self.alpha)
                    self.exp_grad_in.addcmul_(grad_in, grad_in, value=1.0 - self.alpha)
                    self.exp_grad_out.mul_(self.alpha)
                    self.exp_grad_out.addcmul_(grad_out, grad_out, value=1.0 - self.alpha)
                    self.exp_ratio.mul_(self.alpha)
                    self.exp_ratio.addcmul_(ratio, ratio, value=1.0 - self.alpha)
                else:
                    self.exp_grad_in.fill_(grad_in)
                    self.exp_grad_out.fill_(grad_out)
                    self.exp_ratio.fill_(ratio)

                self.exp_grad_in.grad = None
                self.exp_grad_out.grad = None

                self.steps += 1

    def get_scale(self):
        return (self.scale / (self.exp_ratio + self.eps)).clamp_(0.01, 1.0)

    def track_input_grad(self, x):
        if not self.exp_grad_in.retains_grad:
            self.exp_grad_in.retain_grad()
        return _grad_tracker(x, self.exp_grad_in)

    def track_output_grad(self, x):
        if not self.exp_grad_out.retains_grad:
            self.exp_grad_out.retain_grad()
        return _grad_tracker(x, self.exp_grad_out) #_grad_tracker_with_weight(x, self.exp_grad_out, scale)


def collect_all_grad_norms(module):
    grad_norm_dict = {}
    for name, child in module.named_modules():
        if isinstance(child, GRADTracker):
            value = child.get_last_norm()
            if value is not None:
                grad_norm_dict['Tracked Gradients/' + name + f'.{child.key}'] = value
    return grad_norm_dict


def step_all_grad_scalers(module):
    for name, child in module.named_modules():
        if isinstance(child, GradScaler):
            child.step()


def collect_all_grad_scaler_stats(module):
    grad_norm_dict = {}
    for name, child in module.named_modules():
        if isinstance(child, GradScaler):
            grad_norm_dict['Tracked Gradients/' + name + f'.grad_in'] = child.exp_grad_in
            grad_norm_dict['Tracked Gradients/' + name + f'.grad_out'] = child.exp_grad_out
            grad_norm_dict['Tracked Gradients/' + name + f'.grad_ratio'] = child.exp_ratio
            grad_norm_dict['Tracked Gradients/' + name + f'.grad_scale'] = child.get_scale()

    return grad_norm_dict


def dirichlet_energy(x, edge_index, batch=None):
    with torch.no_grad():
        aggr = SumAggregation()
        src, dst = edge_index
        deg = degree(src, num_nodes=x.shape[0])

        x = x / torch.sqrt(deg + 1.0).view(-1, 1)
        energy = torch.norm(x[src] - x[dst], dim=1, p=2) ** 2.0

        if batch is not None:
            energy = aggr(energy.view(-1, 1), batch[dst], dim_size=x.shape[0])
        else:
            energy = energy.sum()

        energy *= 0.5

    return energy
