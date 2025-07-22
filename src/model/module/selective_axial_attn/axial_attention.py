import torch
import logging
from torch import nn
from einops import rearrange
from operator import itemgetter
from .sel_utils import GumbelTopKSelector

log = logging.getLogger(__name__)


# helper functions
def exists(val):
    return val is not None


def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))


def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)


# calculates the permutation to bring the input tensor to something attend-able
# also calculates the inverse permutation to bring the tensor back to its original shape
def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)
        
    return permutations


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x)
            x = x + g(x)
        return x
    

class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation
        
    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation) # .contiguous()

        shape = axial.shape
        *_, i, j, d = shape
        # merge all but axial dimension
        axial = axial.reshape(-1, j, d)
        # attention
        axial = self.fn(axial, merged_axis=i, **kwargs)

        # restore to original shape and permutation
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial


class AxialPositionalEmbedding(nn.Module):
    """axial pos emb"""
    def __init__(self, dim, shape, emb_dim_index = 1):
        super().__init__()
        parameters = []
        total_dimensions = len(shape) + 2
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i != emb_dim_index]

        self.num_axes = len(shape)

        for i, (axial_dim, axial_dim_index) in enumerate(zip(shape, ax_dim_indexes)):
            shape = [1] * total_dimensions
            shape[emb_dim_index] = dim
            shape[axial_dim_index] = axial_dim
            parameter = nn.Parameter(torch.randn(*shape))
            setattr(self, f'param_{i}', parameter)

    def forward(self, x):
        for i in range(self.num_axes):
            x = x + getattr(self, f'param_{i}')
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads = None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias = False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv = None, **kwargs):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out


class SelectiveSelfAttention(torch.nn.Module):
    def __init__(self, attn, selector, row_selector=None):
        super().__init__()
        self.attn = attn
        self.selector = selector
        self.row_selector = row_selector
    
    def forward(self, x, merged_axis=None):
        if self.row_selector is None:
            x_sel_in, idx_in, probs_out = self.selector(x)  # '(b k) k d'
            if probs_out is not None and self.training:
                # probs_out with shape (b, i)
                x = x - probs_out[..., None].detach() + probs_out[..., None]
            x_sel_in = self.attn(x_sel_in)
            x = x.scatter(dim=1, index=idx_in, src=x_sel_in)
            return x
        
        # out-axis selection
        if merged_axis is None:
            log.warning('`merged_axis` is not given, assumed to be same as unmerged one.')
            merged_axis = x.shape[-2]
        
        _, j, d = x.shape
        k, row_k = self.selector.k, self.row_selector.k
        x = rearrange(x, '(b i) j d -> b i j d', i=merged_axis)
        
        idx_out, probs_out = self.row_selector(x.mean(dim=-2), index_only=True)  # index=(b, k, d)
        # offer grad to unchosen points
        if probs_out is not None:
            # probs_out with shape (b, i)
            x = x - probs_out[..., None, None].detach() + probs_out[..., None, None]
        
        idx_out = idx_out[..., None, :].expand(-1, -1, j, -1)
        x_sel_out = x.gather(dim=1, index=idx_out)  # (b, k, j, d)
        x_sel_out_back = rearrange(x_sel_out, 'b k j d -> (b k) j d')
        
        # in-axis selection
        x_sel_in, idx_in, probs_in = self.selector(x_sel_out_back)  # '(b k) k d'
        
        if probs_in is not None:
            x_sel_out_back = x_sel_out_back - probs_in[..., None].detach() + probs_in[..., None]

        # compute attn
        x_sel_in = self.attn(x_sel_in)
        
        # restore shape
        x_sel_out_back = x_sel_out_back.scatter(dim=1, index=idx_in, src=x_sel_in)  # '(b k) j d'
        x_sel_out_back = rearrange(x_sel_out_back, '(b k) j d -> b k j d', k=row_k)
        x = x.scatter(dim=1, index=idx_out, src=x_sel_out_back)  # 'b i j d'
        x = rearrange(x, 'b i j d -> (b i) j d')
        
        return x


class AxialAttention(nn.Module):
    def __init__(
        self, 
        dim, 
        num_dimensions = 2, 
        heads = 8, 
        dim_heads = None, 
        dim_index = -1, 
        sum_axial_out = True,
        use_selector=False,
        selector_config=None,
        row_selector_config=None,
    ):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)

        attentions = []
        if not use_selector or not (selector_config or row_selector_config):
            for permutation in calculate_permutations(num_dimensions, dim_index):
                attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))
        else:
            for permutation in calculate_permutations(num_dimensions, dim_index):
                attentions.append(PermuteToFrom(
                    permutation,
                    SelectiveSelfAttention(
                        attn=SelfAttention(dim, heads, dim_heads), 
                        selector=GumbelTopKSelector(feature_dim=dim, **selector_config) if selector_config else None,
                        row_selector=GumbelTopKSelector(feature_dim=dim, **row_selector_config) if row_selector_config else None,
                    )
                ))
                
        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions)) / 2

        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
            return out
        return out