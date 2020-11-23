# credits to @lucidrains https://github.com/lucidrains

import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial, reduce
from inspect import isfunction
from operator import mul

from einops import rearrange, repeat
try:
    from axial_positional_embedding import AxialPositionalEmbedding, AxialPositionalEmbeddingImage
except ImportError as e:
    print(e)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def expand_dim1(x):
    if len(x.shape) == 1:
        return x[None, :]
    else: return x

# generative helpers
# credit https://github.com/huggingface/transformers/blob/a0c62d249303a68f5336e3f9a96ecf9241d7abbe/src/transformers/generation_logits_process.py
def top_p_filter(logits, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    # if min_tokens_to_keep > 1:
    #         # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
    #         sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    scores[indices_to_remove] = float('-inf')
    return scores

def top_k_filter(logits, top_k=20):
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    return logits

_sampler = {
    'top_k':top_k_filter,
    'top_p':top_p_filter,
    'gready':lambda x: x.argmax(-1)
}

# axial position helpers (subjected to review)
def get_axial_dims(dim, n):
    res = (dim//n, )*(n-1)
    res += (dim-sum(res), )
    return res

"""## Helpers and FeedForward"""

# helper classes 
# based on https://github.com/lucidrains/all-normalization-transformer/blob/master/all_normalization_transformer/all_normalization_transformer.py

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
# Added *args, **kwargs here to pass context and masks
class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.fn(x, *args, **kwargs)
        return self.norm(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, d_ff=None, dropout=0.):
        super().__init__()
        d_ff = default(d_ff, 4*dim)
        self.net = nn.Sequential(
            nn.Linear(dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


"""## Attention"""

MASK_VAL = -1e4 # instead of float('-inf') to make fp16 work

class Attention(nn.Module):
    def __init__(self, 
                 dim, 
                 heads = 8, 
                 causal = False,
                 mask = None,
                 dropout=0.1):
        super().__init__()
        self.causal = causal
        self.store_attention = False
        self.mask = mask #??
        self.heads = heads
        self.scale = dim ** -0.5
        
        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_kv = nn.Linear(dim, dim * 2, bias = False)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, context = None, mask = None, context_mask = None, store_attention=False):
        b, n, _, h, device = *x.shape, self.heads, x.device
        kv_input = default(context, x)

        q = self.to_q(x) # replaced q_ with q (don't need to store it fore basic tfmr)
        kv = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, *kv))
        # boolean input_mask is False at positions not to attend to
        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: torch.ones((b, n), device = device).bool())
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(k_mask, lambda: torch.ones((b, k.shape[-2]), device = device).bool())
            q_mask = rearrange(q_mask, 'b i -> b () i ()')
            k_mask = rearrange(k_mask, 'b j -> b () () j')
            input_mask = q_mask * k_mask
        # classic dot-product attention
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # might need to tune MASK_VAL for fp16 to work
        if exists(input_mask):
            dots.masked_fill_(~input_mask, MASK_VAL)
            del input_mask

        if self.causal:
            i, j = dots.shape[-2:]
            mask = torch.ones((i, j), device = device).triu_(j - i + 1).bool()
            dots.masked_fill_(mask, MASK_VAL)
            del mask

        attn = F.softmax(dots, -1)
        if self.store_attention: # and not self.training
            self.attention = attn.detach().cpu()
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        #out = self.dropout(out) # option for more dropout here
        return out


"""## Transformer blocks

### Encoder
"""

class TransformerEncoderBlock(nn.Module):
    """
    Bacis transformer encoder block. Consists of multi-head attention and positional feedforward layers
    """
    def __init__(self, dim, heads = 8, causal = False, mask = None, 
                 attn_dropout=0.1, ff_dropout=0.1, d_ff=None, prenorm=False):
        super().__init__()
        self.attn_dropout = attn_dropout # mb separate argument attn_post_dropout
        norm_wrapper = PreNorm if prenorm else PostNorm
        self.attn = Residual(norm_wrapper(dim, Attention(dim, heads=heads, causal=causal, dropout=attn_dropout)))
        self.ff = Residual(norm_wrapper(dim, FeedForward(dim, d_ff=d_ff, dropout=ff_dropout)))
        
    def forward(self, x, mask=None): #? more args
        out = self.attn(x, mask=mask)
        out = F.dropout(out, p=self.attn_dropout)
        out = self.ff(out)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth=6, heads=8, causal=False, d_ff=None, attn_dropout=0.1, ff_dropout=0.1, prenorm=False):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerEncoderBlock(dim, heads, causal=causal, d_ff=d_ff, 
                                    attn_dropout=attn_dropout, ff_dropout=ff_dropout, prenorm=prenorm))
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

"""Decoder block has attention and cross attention

### Decoder
"""

class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim, heads = 8, mask = None, d_ff=None,
                 attn_dropout=0.1, ff_dropout=0.1, prenorm=False):
        super().__init__()
        self.attn_dropout = attn_dropout # mb separate argument attn_post_dropout
        norm_wrapper = PreNorm if prenorm else PostNorm
        self.attn = Residual(norm_wrapper(dim, Attention(dim, heads=heads, causal=True, dropout=attn_dropout)))
        self.cross = Residual(norm_wrapper(dim, Attention(dim, heads=heads, causal=False, dropout=attn_dropout)))
        self.ff = Residual(norm_wrapper(dim, FeedForward(dim, d_ff=d_ff, dropout=ff_dropout)))
        
    def forward(self, x, context, mask=None, context_mask=None):
        out = self.attn(x, mask=mask)
        out = F.dropout(out, p=self.attn_dropout)
        out = self.cross(out, context, mask=mask, context_mask=context_mask)
        out = F.dropout(out, p=self.attn_dropout)
        out = self.ff(out)
        return out

class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth=6, heads=8, d_ff=None, attn_dropout=0.1, ff_dropout=0.1, prenorm=False):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerDecoderBlock(dim, heads, d_ff=d_ff, attn_dropout=attn_dropout, ff_dropout=ff_dropout, prenorm=False))
    def forward(self, x, context, mask=None, context_mask=None):
        for layer in self.layers:
            x = layer(x, context, mask, context_mask)
        return x

"""### Models"""
# from https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reformer_pytorch.py#L609

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)

class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]
#TODO add axial positional encodings

class TransformerEmbedding(nn.Module):
    """
    Combines token embedings with positional encodings
    pos_enc: str from {'absolute', 'fixed', 'axial'}
    """
    def __init__(self, emb_sz, dim, max_seq_len=512, dropout=0., pos_enc='absolute', 
                 axial_shape=None, axial_emb_dims=None):
        super().__init__()
        self.scale = dim**0.5
        self.emb = nn.Embedding(emb_sz, dim)
        if pos_enc == 'absolute':
            self.pos_enc = AbsolutePositionalEmbedding(dim, max_seq_len)
        elif pos_enc == 'fixed':
            self.pos_enc = FixedPositionalEmbedding(dim)
        elif pos_enc == 'axial':
            assert axial_shape is not None
            assert reduce(mul, axial_shape) == max_seq_len
            axial_emb_dims = default(axial_emb_dims, get_axial_dims(dim, len(axial_shape)))
            self.pos_enc = AxialPositionalEmbedding(dim, axial_shape, axial_emb_dims)
        self.dropout = nn.Dropout(dropout)
        self._init()
    def forward(self, x):
        _, n = x.shape
        x = self.emb(x)
        x *= self.scale
        x += self.pos_enc(x)
        return self.dropout(x)
    def _init(self):
        nn.init.normal_(self.emb.weight, std = 0.02)
        if hasattr(self.pos_enc, 'weight'):
            nn.init.normal_(self.pos_enc.weight, std = 0.02)

#TODO test weight tying
# Note on weight tying: it's done like here in fastai AWD_LSTM model
# Lucidrains does it with custom MatrixMultiply module https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reformer_pytorch.py#L106
class TransformerEncDec(nn.Module):
    """
    Basic Transformer Encoder-Decoder model
    Parameters:
        * enc_vocab_sz: int - source vocab size 
        * dec_vocab_sz: int - target vocab size
        * dim: int - inner dimension of the model
        * depth: int (default: 6) 
        * heads: int (default: 8)
        * max_seq_len: int (default: 512)
        * pad_idx: int - padding token id, if pad_idx is provided, and no mask/context_mask are passed to 
                forward method will be used to generate padding masks
        * tie_weights: bool - if True target embedding weights are used for computation output projection
        * pos_enc: str from {'absolute', 'fixed', 'axial'} - type of positional encoding to use
    Inputs:
        * src - source input ids, shape [bs, src_sl]
        * tgt - target input ids, shape [bs, tgt_sl]
        * src_mask - optional boolean source mask, shape [bs, src_sl]
        * tgt_mask - optional boolean target mask, shape [bs, tgt_sl]
    Returns:
        * logits - target token logits, shape [bs, tgt_sl, tgt_vocab_sz]
    """
    def __init__(self, enc_vocab_sz, dec_vocab_sz, dim, depth=6, heads=8, 
                 max_seq_len=512, pad_idx=None, tie_weights=True, 
                 attn_dropout=0.1, ff_dropout=0.1, emb_dropout=0.1,
                 pos_enc='absolute', d_ff=None, prenorm=False, 
                 axial_shape=None, axial_emb_dims=None):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.depth = depth
        self.pad_idx = pad_idx
        self.enc_emb = TransformerEmbedding(enc_vocab_sz, dim, max_seq_len, dropout=emb_dropout,
                                            axial_shape=axial_shape, axial_emb_dims=axial_emb_dims)
        self.dec_emb = TransformerEmbedding(dec_vocab_sz, dim, max_seq_len, dropout=emb_dropout,
                                            axial_shape=axial_shape, axial_emb_dims=axial_emb_dims)
        self.encoder = TransformerEncoder(dim, depth, heads, d_ff=d_ff, attn_dropout=attn_dropout, ff_dropout=ff_dropout, prenorm=prenorm)
        self.decoder = TransformerDecoder(dim, depth, heads, d_ff=d_ff, attn_dropout=attn_dropout, ff_dropout=ff_dropout, prenorm=prenorm)
        self.proj = nn.Linear(dim, dec_vocab_sz)
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

class TransformerLM(nn.Module):
    """
    Basic Transformer for language modelling
    Parameters:
        * vocab_sz: int
        * dim: int - inner dimension of the model
        * depth: int (default: 6) 
        * heads: int (default: 8)
        * causal: bool (default: True) - if True does causal masking automatically
        * max_seq_len: int (default: 512)
        * tie_weights: bool - if True target embedding weights are used for computation output projection
        * pos_enc: str from {'absolute', 'fixed', 'axial'} - type of positional encoding to use
    Inputs:
        * x - input ids, shape [bs, sl]
        * mask - optional boolean mask, shape [bs, sl]
    Returns:
        * logits - target token logits, shape [bs, sl, vocab_sz]
    """
    def __init__(self, vocab_sz, dim, depth=6, heads=8, causal=True,
                 max_seq_len=512, tie_weights=True, d_ff=None,
                 attn_dropout=0.1, ff_dropout=0.1, emb_dropout=0.1,
                 pos_enc='absolute', pad_idx=None, prenorm=False,
                 axial_shape=None, axial_emb_dims=None):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.depth = depth
        self.pad_idx = pad_idx
        self.emb = TransformerEmbedding(vocab_sz, dim, max_seq_len, dropout=emb_dropout, 
                                        axial_shape=axial_shape, axial_emb_dims=axial_emb_dims)
        self.tfmr = TransformerEncoder(dim, depth, heads, causal=causal, d_ff=d_ff, 
                                       attn_dropout=attn_dropout, ff_dropout=ff_dropout,
                                       prenorm=prenorm)
        self.proj = nn.Linear(dim, vocab_sz)
        if tie_weights: self.proj.weight = self.emb.emb.weight
        
    def forward(self, x, mask=None):
        x = self.emb(x)
        x = self.tfmr(x, mask=mask)
        return self.proj(x)
    #TODO maybe refactor
    @torch.no_grad()
    def generate(self, inp,
                max_len=50,
                temperature=1.,
                method = 'top_k',
                top_k = 20,
                top_p = 0.9,
                early_stopping=False, #need eos_idx to work
                eos_idx=None):
        self.to(inp.device) #TODO test for potential problems
        self.eval()
        thresh = top_k if method=='top_k' else top_p
        sampler = _sampler[method]
        inp = expand_dim1(inp)
        b, t = inp.shape
        out = inp
        for _ in range(max_len):
            x = out[:, -self.max_seq_len:]

            logits = self(x)[:, -1, :]
            if method == 'greedy':
                sample = sampler(logits)
            else:
                filtered_logits = sampler(logits)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

            if early_stopping and (sample == eos_idx).all():
                break
        # out = out[:, t:]
        return out
    
    def store_attention(self, layer_ids=None):
        #defaults to storing attention for all layers
        layer_ids = default(layer_ids, list(range(self.depth)))
        for module in self.children():
            if issubclass(type(module), (TransformerEncoder, TransformerDecoder)):
                for i, l in enumerate(module.layers):
                    if i in layer_ids:
                        for m in l.modules():
                            if issubclass(type(m), (Attention)):
                                m.store_attention = True
    def get_attention_matrix(self):
        res = []
        for m in self.modules():
            if issubclass(type(m), (Attention)):
                attention = getattr(m, 'attention', None)
                if attention is not None:
                    res.append(attention)
                # reset stored attention
                m.attention = None
                m.store_attention = False
        return res