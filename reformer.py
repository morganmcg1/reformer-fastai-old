# credits to @lucidrains https://github.com/lucidrains
# raw version to be added LSH attention and more...

from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states
from functools import wraps

from basic_transformer import *

# helper functions

# helper classes

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim = self.dim)

class ChunkedFeedForward(nn.Module):
    def __init__(self, d, ff_d=None, chunks=1, dropout=0., along_dim=-1):
        super().__init__()
        ff_d = default(ff_d, 4*d)
        self.net = nn.Sequential(
            nn.Linear(d, ff_d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_d, d),
            nn.Dropout(dropout)
            )
        self.chunks = chunks
        self.dim = along_dim
    def forward(self, x):
        if self.chunks == 1:
            return self.net(x)
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.net(c) for c in chunks], dim = self.dim)


# following example for saving and setting rng here https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html
class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng = False, set_rng = False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)

# heavily inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
# once multi-GPU is confirmed working, refactor and send PR back to source
class ReversibleBlock(nn.Module):
    def __init__(self, f, g, depth=None, send_signal = False):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

        self.depth = depth
        self.send_signal = send_signal

    def forward(self, x, f_args = {}, g_args = {}):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1, y2 = None, None

        if self.send_signal:
            f_args['_reverse'] = g_args['_reverse'] = False
            f_args['_depth'] = g_args['_depth'] = self.depth

        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)

        return torch.cat([y1, y2], dim=2)

    def backward_pass(self, y, dy, f_args = {}, g_args = {}):
        y1, y2 = torch.chunk(y, 2, dim=2)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy

        if self.send_signal:
            f_args['_reverse'] = g_args['_reverse'] = True
            f_args['_depth'] = g_args['_depth'] = self.depth

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)
        
        return x, dx

class IrreversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g
    def forward(self, x, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1 = x1 + self.f(x2, **f_args)
        y2 = x2 + self.g(y1, **g_args)
        return torch.cat([y1, y2], dim=2)


class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        ctx.kwargs = kwargs
        for block in blocks:
            x = block(x, **kwargs)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x
    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        kwargs = ctx.kwargs
        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None

class ReversibleSequence(nn.Module):
    def __init__(self, blocks, layer_dropout = 0., reverse_thres = 0, send_signal = False):
        super().__init__()
        self.layer_dropout = layer_dropout
        self.reverse_thres = reverse_thres # uses revblocks if seq_len else irrev_blocks

        self.blocks = nn.ModuleList([ReversibleBlock(f, g, depth, send_signal) for depth, (f, g) in enumerate(blocks)])
        self.irrev_blocks = nn.ModuleList([IrreversibleBlock(f=f, g=g) for f, g in blocks])

    def forward(self, x, arg_route = (True, True), **kwargs):
        reverse = x.shape[1] > self.reverse_thres
        blocks = self.blocks if reverse else self.irrev_blocks

        if self.training and self.layer_dropout > 0:
            to_drop = torch.empty(len(self.blocks)).uniform_(0, 1) < self.layer_dropout
            blocks = [block for block, drop in zip(self.blocks, to_drop) if not drop]
            blocks = self.blocks[:1] if len(blocks) == 0 else blocks

        f_args, g_args = map(lambda route: kwargs if route else {}, arg_route)
        block_kwargs = {'f_args': f_args, 'g_args': g_args}

        if not reverse:
            for block in blocks:
                x = block(x, **block_kwargs)
            return x

        return _ReversibleFunction.apply(x, blocks, block_kwargs)

# mess for now; will clean up after LSHAttention args finalized
class ReformerEncoder(nn.Module):
    def __init__(self, 
                 dim, 
                 depth, 
                 max_seq_len, 
                 heads = 8, 
                 dim_head = None, 
                 bucket_size = 64, 
                 n_hashes = 8, 
                 ff_chunks = 100, 
                 attn_chunks = None, # ??
                 causal = False, 
                 weight_tie = False, # ??
                 attn_dropoup = 0.,
                 post_attn_dropout = 0.,
                 lsh_dropout = 0., 
                 ff_dropout = 0.,  
                 ff_d = None, 
                 layer_dropout = 0., 
                 lsh_attend_across_buckets = True, 
                 lsh_allow_duplicate_attention = True, 
                 random_rotations_per_head = False,                  
                 use_full_attn = False, 
                 full_attn_thres = 0, 
                 reverse_thres = 0,  
                 one_value_head = False, 
                 n_local_attn_heads = 0,
                 prenorm=True):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.bucket_size = bucket_size
        # self.full_attn_thres = full_attn_thres
        
        # use regular attention for now
        get_attn = lambda: Attention(dim, heads, causal=causal, dropout=attn_dropoup)
        # get_attn = lambda: LSHSelfAttention(dim, heads, bucket_size, n_hashes, causal = causal, dim_head = dim_head, dropout = lsh_dropout, post_attn_dropout = post_attn_dropout, attn_chunks = attn_chunks, allow_duplicate_attention = lsh_allow_duplicate_attention, attend_across_buckets = lsh_attend_across_buckets, random_rotations_per_head = random_rotations_per_head, num_mem_kv = num_mem_kv, use_full_attn = use_full_attn, full_attn_thres = full_attn_thres, one_value_head = one_value_head, n_local_attn_heads = n_local_attn_heads)
        # get_ff = lambda: Chunk(ff_chunks, FeedForward(dim, d_ff=ff_d, dropout=ff_dropout), along_dim = -2)
        get_ff = lambda: ChunkedFeedForward(dim, ff_d, chunks=ff_chunks, dropout=ff_dropout, along_dim=1)

        blocks = []
        #TODO: find where ReZero proposed
        #residual_fn_wrapper = ReZero if use_rezero else partial(PreNorm, norm_type, dim)
        norm_wrapper = PreNorm if prenorm else PostNorm
        
        for ind in range(depth):
            layer_num = ind + 1
            
            attn = get_attn()
            ff = get_ff()

            f = norm_wrapper(dim, attn)
            g = norm_wrapper(dim, ff)

            blocks.append(nn.ModuleList([f, g]))
        # send_signal is not implemented for now
        self.layers = ReversibleSequence(nn.ModuleList(blocks), layer_dropout=layer_dropout, reverse_thres=reverse_thres, send_signal=False)

    def forward(self, x, **kwargs):
        x = torch.cat([x, x], dim = -1)
        arg_route = (True, False)
        # pdb.set_trace()
        x = self.layers(x, arg_route = arg_route, **kwargs)
        return torch.stack(x.chunk(2, dim=-1)).mean(dim=0)


class ReformerLM(nn.Module):#, TransformerLM):
    def __init__(self,
                 vocab_sz,
                 dim, 
                 depth = 6,
                 tie_weights = True,
                 max_seq_len = 512, 
                 heads = 8, 
                 dim_head = None, 
                 bucket_size = 64, 
                 n_hashes = 8, 
                 ff_chunks = 100, 
                 attn_chunks = None, # ??
                 causal = True, 
                 weight_tie = False, # ??
                 attn_dropoup = 0.,
                 post_attn_dropout = 0.,
                 lsh_dropout = 0., 
                 ff_dropout = 0.,  
                 ff_d = None, 
                 layer_dropout = 0., 
                 lsh_attend_across_buckets = True, 
                 lsh_allow_duplicate_attention = True, 
                 random_rotations_per_head = False,                  
                 use_full_attn = False, 
                 full_attn_thres = 0, 
                 reverse_thres = 0,  
                 one_value_head = False, 
                 n_local_attn_heads = 0,
                 prenorm=True):
        super().__init__()
        self.emb = TransformerEmbedding(vocab_sz, dim, max_seq_len=max_seq_len)
        #temp line to mark we need to pass more args to encoder
        kwargs = {}
        self.encoder = ReformerEncoder(dim, depth, max_seq_len, causal=causal,
                                        **kwargs)
        self.proj = nn.Linear(d_model, vocab_sz)
        if tie_weights: self.proj.weight = self.emb.emb.weight
    def forward(self, x, mask=None):
        x = self.emb(x)
        x = self.encoder(x, mask=mask)
        return self.proj(x)