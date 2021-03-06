# credits to @lucidrains https://github.com/lucidrains
# raw version to be added LSH attention and more...

from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states
from functools import wraps

from basic_transformer import *

# helper functions

# helper classes

class Chunk(nn.Module):
    def __init__(self, chunks, fn, dim = -1):
        super().__init__()
        self.dim = dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim = self.dim)

class ChunkedFeedForward(nn.Module):
    def __init__(self, d, d_ff=None, chunks=1, dropout=0., dim=-1):
        super().__init__()
        d_ff = default(d_ff, 4*d)
        self.net = nn.Sequential(
            nn.Linear(d, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d),
            nn.Dropout(dropout)
            )
        self.chunks = chunks
        self.dim = dim
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
                 d_model, 
                 depth, 
                 heads = 8, 
                 max_seq_len = 512,              
                 d_head = None, 
                 bucket_size = 64, 
                 n_hashes = 8, 
                 ff_chunks = 100, 
                 attn_chunks = None, # ??
                 causal = False, 
                 weight_tie = False, # ??
                 attn_dropout = 0.,
                 post_attn_dropout = 0.,
                 lsh_dropout = 0., 
                 ff_dropout = 0.,  
                 d_ff = None, 
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
        self.d_model = d_model
        self.depth = depth

        self.bucket_size = bucket_size
        # self.full_attn_thres = full_attn_thres
        
        # use regular attention for now
        get_attn = lambda: Attention(d_model, heads, causal=causal, dropout=attn_dropout)
        # get_attn = lambda: LSHSelfAttention(d_model, heads, bucket_size, n_hashes, causal = causal, d_head = d_head, dropout = lsh_dropout, post_attn_dropout = post_attn_dropout, attn_chunks = attn_chunks, allow_duplicate_attention = lsh_allow_duplicate_attention, attend_across_buckets = lsh_attend_across_buckets, random_rotations_per_head = random_rotations_per_head, num_mem_kv = num_mem_kv, use_full_attn = use_full_attn, full_attn_thres = full_attn_thres, one_value_head = one_value_head, n_local_attn_heads = n_local_attn_heads)
        # get_ff = lambda: Chunk(ff_chunks, FeedForward(d_model, d_ff=d_ff, dropout=ff_dropout), dim = -2)
        get_ff = lambda: ChunkedFeedForward(d_model, d_ff, chunks=ff_chunks, dropout=ff_dropout, dim=1)

        blocks = []
        #residual_fn_wrapper = ReZero if use_rezero else partial(PreNorm, norm_type, d_model)
        norm_wrapper = PreNorm if prenorm else PostNorm
        
        for ind in range(depth):
            layer_num = ind + 1
            
            attn = get_attn()
            ff = get_ff()

            f = norm_wrapper(d_model, attn)
            g = norm_wrapper(d_model, ff)

            blocks.append(nn.ModuleList([f, g]))
        # send_signal is not implemented for now
        self.layers = ReversibleSequence(nn.ModuleList(blocks), layer_dropout=layer_dropout, reverse_thres=reverse_thres, send_signal=False)

    def forward(self, x, **kwargs):
        x = torch.cat([x, x], dim = -1)
        arg_route = (True, False)
        # pdb.set_trace()
        x = self.layers(x, arg_route = arg_route, **kwargs)
        return torch.stack(x.chunk(2, dim=-1)).mean(dim=0)


class ReformerDecoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 depth = 6, 
                 heads = 8,  
                 max_seq_len = 512,
                 d_head = None, 
                 bucket_size = 64, 
                 n_hashes = 8, 
                 ff_chunks = 100, 
                 attn_chunks = None, # ??
                 causal = False, 
                 weight_tie = False, # weight sharing option do we need to keep this?
                 attn_dropout = 0.,
                 post_attn_dropout = 0.,
                 ff_dropout = 0.,  
                 d_ff = None, 
                 layer_dropout = 0.,
                 prenorm=True,
                 reverse_thres = 0,
                 ):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        
        # use regular attention for now
        get_attn = lambda: DecoderAttention(d_model, heads, causal=causal, dropout=attn_dropout)
        get_ff = lambda: ChunkedFeedForward(d_model, d_ff, chunks=ff_chunks, dropout=ff_dropout, dim=1)
        norm_wrapper = PreNorm if prenorm else PostNorm
        blocks = []
        for ind in range(depth):
            layer_num = ind + 1
            
            f = norm_wrapper(d_model, get_attn())
            g = norm_wrapper(d_model, get_ff())

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
    """
    Reformer for language modelling
    Parameters:
        * vocab_sz: int
        * d_model: int - inner dimension of the model
        * n_layers: int (default: 6) 
        * heads: int (default: 8)
        * d_ff: int - inner dimension of the pointwise FeedForward net, if None defaults to 4*d_model
        * attn_dropout: float - attention dropout
        * ff_dropout: float - feed-forward dropout
        * emb_dropout: float - embedding dropout
        * causal: bool (default: True) - if True does causal masking automatically
        * max_seq_len: int (default: 512)
        * tie_weights: bool - if True target embedding weights are used for computation output projection
        * prenorm: bool - wether to use PreNorm or PostNorm
        * attn_bias: bool - wether to allow biases in attention projection layers
        * pad_idx: int - padding token id, required for autogeneration of padding mask
        * pos_enc: str from {'absolute', 'fixed', 'axial'} - type of positional encoding to use
        * axial_shape: tuple - required if 'axial' positional encoding are used, should be factors of 
                max_seq_len
        * axial_emb_dims: tuple - [optional] axial embedding components, should sum to d_model
    Inputs:
        * x - input ids, shape [bs, sl]
        * mask - optional boolean mask, shape [bs, sl]
    Returns:
        * logits - target token logits, shape [bs, sl, vocab_sz]
    """
    def __init__(self,
                 vocab_sz,
                 d_model, 
                 depth = 6,
                 tie_weights = True,
                 max_seq_len = 512, 
                 heads = 8, 
                 d_head = None, 
                 bucket_size = 64, 
                 n_hashes = 8, 
                 ff_chunks = 100, 
                 attn_chunks = None, # ??
                 causal = True, 
                 weight_tie = False, # ??
                 attn_dropout = 0.,
                 post_attn_dropout = 0.,
                 lsh_dropout = 0., 
                 ff_dropout = 0.,  
                 d_ff = None, 
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
        self.emb = TransformerEmbedding(vocab_sz, d_model, max_seq_len=max_seq_len)
        #temp line to mark we need to pass more args to encoder
        kwargs = {}
        self.encoder = ReformerEncoder(d_model, depth, max_seq_len=max_seq_len, causal=causal, reverse_thres=reverse_thres,
                                        **kwargs)
        self.proj = nn.Linear(d_model, vocab_sz)
        if tie_weights: self.proj.weight = self.emb.emb.weight
    def forward(self, x, mask=None):
        x = self.emb(x)
        x = self.encoder(x, mask=mask)
        return self.proj(x)



class ReformerEncDec(nn.Module):
    """
    Basic Transformer Encoder-Decoder model
    Parameters:
        * enc_vocab_sz: int - source vocab size
        * dec_vocab_sz: int - target vocab size
        * d_model: int - inner dimension of the model
        * n_enc_layers: int (default: 6) 
        * n_dec_layers: int (default: 6)
        * heads: int (default: 8)
        * d_ff: int - inner dimension of the FeedForward net, if None defaults to 4*d_model
        * attn_dropout: float - attention dropout
        * ff_dropout: float - feed-forward dropout
        * emb_dropout: float - embedding dropout
        * max_seq_len: int (default: 512)
        * prenorm: bool - whether to use PreNorm or PostNorm
        * attn_bias: bool - whether to allow biases in attention projection layers
        * pad_idx: int - padding token id, if pad_idx is provided, and no mask/context_mask are passed to 
                forward method will be used to generate padding masks
        * tie_weights: bool - if True target embedding weights are used for computation output projection
        * shared_emb: bool - if True encoder and decoder will use shared embedding layer
        * pos_enc: str from {'absolute', 'fixed', 'axial'} - type of positional encoding to use
        * axial_shape: tuple - required if 'axial' positional encoding are used, should be factors of 
                max_seq_len
        * axial_emb_dims: tuple - [optional] axial embedding components, should sum to d_model
    Inputs:
        * src - source input ids, shape [bs, src_sl]
        * tgt - target input ids, shape [bs, tgt_sl]
        * src_mask - optional boolean source mask, shape [bs, src_sl]
        * tgt_mask - optional boolean target mask, shape [bs, tgt_sl]
    Returns:
        * logits - target token logits, shape [bs, tgt_sl, tgt_vocab_sz]
    """
    def __init__(self,
                 enc_vocab_sz, 
                 dec_vocab_sz, 
                 d_model, 
                 depth=6, 
                 heads=8, 
                 max_seq_len=512, 
                 pad_idx=None, 
                 tie_weights=True,                  
                 emb_dropout=0.1,
                 attn_dropout=0.1, 
                 ff_dropout=0.1,
                 pos_enc='absolute', 
                 d_ff=None, 
                 prenorm=False, 
                 axial_shape=None, 
                 axial_emb_dims=None,
                 comb_attn=False,
                 reverse_thres=0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.depth = depth
        self.pad_idx = pad_idx
        self.enc_emb = TransformerEmbedding(enc_vocab_sz, d_model, max_seq_len, dropout=emb_dropout,
                                            axial_shape=axial_shape, axial_emb_dims=axial_emb_dims)
        self.dec_emb = TransformerEmbedding(dec_vocab_sz, d_model, max_seq_len, dropout=emb_dropout,
                                            axial_shape=axial_shape, axial_emb_dims=axial_emb_dims)
        self.encoder = ReformerEncoder(d_model, depth, heads, d_ff=d_ff, attn_dropout=attn_dropout, ff_dropout=ff_dropout, prenorm=prenorm, reverse_thres=reverse_thres)
        self.decoder = ReformerDecoder(d_model, depth, heads, d_ff=d_ff, attn_dropout=attn_dropout, ff_dropout=ff_dropout, prenorm=prenorm, reverse_thres=reverse_thres)
        self.proj = nn.Linear(d_model, dec_vocab_sz)
        if tie_weights: self.proj.weight = self.dec_emb.emb.weight

    def forward(self, src, tgt, src_mask = None, tgt_mask = None):
        src_mask = default(src_mask, self.get_padding_mask(src))
        tgt_mask = default(tgt_mask, self.get_padding_mask(tgt))
        enc = self.encoder(self.enc_emb(src), mask = src_mask)
        out = self.decoder(self.dec_emb(tgt), context=enc, mask=tgt_mask, context_mask=src_mask)
        return self.proj(out)
    def get_padding_mask(self, x):
        if self.pad_idx is None: return None
        return (x != self.pad_idx)
    #TODO add beam search and refactor
    @torch.no_grad()
    def generate(self, src,
                src_mask=None,
                max_len=50,
                temperature=1.,
                method = 'top_k',
                top_k = 20,
                top_p = 0.9,
                early_stopping=False,
                bos_idx=2, # TODO change to match future usecases
                eos_idx=None):
        self.to(src.device) #TODO test for potential problems
        self.eval()
        thresh = top_k if method=='top_k' else top_p
        sampler = _sampler[method]
        src = expand_dim1(src)
        bs = src.size(0)
        inp = src.new_full((bs, 1), bos_idx) #start with bos tokens
        pdb.set_trace()
        src_mask = default(src_mask, self.get_padding_mask(src))
        enc = self.encoder(self.enc_emb(src), mask = src_mask)
        out = inp
        for _ in range(max_len):
            x = out[:, -self.max_seq_len:]
            dec = self.decoder(self.dec_emb(out), context=enc)
            logits = self.proj(dec)[:, -1, :]
            if method == 'greedy':
                sample = sampler(logits)
            else:
                filtered_logits = sampler(logits)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

            if (early_stopping and 
                ((sample == eos_idx).all() or 
                (sample == self.pad_idx).all())):
                break
        #TODO mb output cleanup
        return out
    
    def store_attention(self, layer_ids=None, store_encoder=False, store_decoder=True):
        #defaults to storing attention for all layers
        layer_ids = default(layer_ids, list(range(self.depth)))
        for module in self.children():
            if issubclass(type(module), TransformerEncoder) and store_encoder:
                for i, l in enumerate(module.layers):
                    if i in layer_ids:
                        for m in l.modules():
                            if issubclass(type(m), (Attention)):
                                m.store_attention = True
            elif issubclass(type(module), TransformerDecoder) and store_decoder:
                for i, l in enumerate(module.layers):
                    if i in layer_ids:
                        for m in l.modules():
                            if issubclass(type(m), (Attention)):
                                m.store_attention = True
    #TODO mb separate encoder and decoder attention
    def get_attention_matrix(self, get_encoder=False, get_decoder=True):
        res = []
        if get_encoder:
            for m in self.encoder.modules():
                if issubclass(type(m), (Attention)):
                    attention = getattr(m, 'attention', None)
                    if attention is not None:
                        res.append(attention)
                    # reset stored attention
                    m.attention = None
                    m.store_attention = False
        if get_decoder:
            for m in self.decoder.modules():
                if issubclass(type(m), (Attention)):
                    attention = getattr(m, 'attention', None)
                    if attention is not None:
                        res.append(attention)
                    # reset stored attention
                    m.attention = None
                    m.store_attention = False
        return res