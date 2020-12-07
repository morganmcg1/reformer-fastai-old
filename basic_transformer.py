# credits to @lucidrains https://github.com/lucidrains

import torch
from torch import nn, einsum
import torch.nn.functional as F
from fastai.basics import *

from functools import partial, reduce
from inspect import isfunction
from operator import mul
from copy import deepcopy

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
    logits[indices_to_remove] = float('-inf')
    return logits

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
def get_axial_dims(d_emb, n):
    res = (d_emb//n, )*(n-1)
    res += (d_emb-sum(res), )
    return res

"""## Helpers and FeedForward"""

# helper classes
# based on https://github.com/lucidrains/all-normalization-transformer/blob/master/all_normalization_transformer/all_normalization_transformer.py

class Residual(Module):
    def __init__(self, fn): store_attr()
    def forward(self, x, *args, **kwargs):
        return x + self.fn(x, *args, **kwargs)

# Added *args, **kwargs here to pass context and masks
class PostNorm(Module):
    def __init__(self, d_model, fn):
        store_attr('fn')
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, *args, **kwargs):
        x = self.fn(x, *args, **kwargs)
        return self.norm(x)

    
class PreNorm(Module):
    def __init__(self, d_model, fn):
        store_attr('fn')
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

    
class FeedForward(Module):
    def __init__(self, d_model, d_ff=None, dropout=0.):
        d_ff = default(d_ff, 4 * d_model)
        layers = [nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)
        self._init()
        
    def forward(self, x):
        return self.net(x)
    
    def _init(self):
        [nn.init.xavier_uniform_(p) for p in self.parameters() if p.dim() > 1]


"""## Attention"""

MASK_VAL = -5e4 # instead of float('-inf') to make fp16 work

class Attention(Module):
    def __init__(self, 
                 d_model, 
                 heads = 8, 
                 causal = False,
                 mask = None,
                 dropout=0.1, 
                 bias=True,
                 store_attention=False):
        store_attr('causal, mask, heads, store_attention')
        
        self.scale=(d_model//heads) ** -0.5
        self.to_q = nn.Linear(d_model, d_model, bias=bias)
        self.to_kv = nn.Linear(d_model, d_model * 2, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(d_model, d_model)
        
        self._init()

    def forward(self, x, context = None, mask = None, context_mask = None, store_attention=False):
        b, n, _, h, device = *x.shape, self.heads, x.device
        
        kv_input = default(context, x)
        q = self.to_q(x)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        
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
        dots = torch.einsum('bhid,bhjd->bhij', q*self.scale, k)
        
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
        if self.store_attention: self.attention = attn.detach().cpu()
        
        attn = self.dropout(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return  self.to_out(out)  #out = self.dropout(out) # option for more dropout here
    #TODO
    def _compute_attention(q, k, v, mask):
        pass
    def _init(self):
        [nn.init.xavier_uniform_(w) for w in [self.to_q.weight, self.to_kv.weight, self.to_out.weight]]
        if getattr(self.to_q, 'bias', None) is not None: nn.init.constant_(self.to_q.bias, 0)
        if getattr(self.to_kv, 'bias', None) is not None: nn.init.constant_(self.to_kv.bias, 0)
        nn.init.constant_(self.to_out.bias, 0)

# decoder attention class combining self and cross attention 
# may be replaced with generalized attention in future
class DecoderAttention(nn.Module):
    def __init__(self, 
                 d_model, 
                 heads = 8, 
                 causal = False,
                 mask = None,
                 dropout=0.1, 
                 bias=True):
        super().__init__()
        self.causal = causal
        self.store_attention = False
        self.mask = mask #??
        self.heads = heads
        self.scale = (d_model//heads) ** -0.5
        
        self.to_q = nn.Linear(d_model, d_model, bias = bias)
        self.to_kv = nn.Linear(d_model, d_model * 2, bias = bias)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Linear(d_model, d_model)

        self._init()

    def forward(self, x, context = None, mask = None, context_mask = None, store_attention=False):
        b, n, d, h, device = *x.shape, self.heads, x.device
        context = default(context, torch.empty(b, 0, d, dtype=x.dtype, device=device))
        kv_input = torch.cat([x, context], dim=-2)
        
        q = self.to_q(x)
        kv = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, *kv))

        # boolean input_mask is False at positions not to attend to
        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: torch.ones((b, n), device = device).bool())
            self_mask = q_mask[:, None, :, None] * q_mask[:, None, None, :]
            if context.size(-2) != 0:
                k_mask = default(context_mask, lambda: torch.ones((b, context.shape[-2]), device = device).bool())
                cross_mask = q_mask[:, None, :, None] * k_mask[:, None, None, :]
            else: cross_mask = torch.empty(0, dtype=self_mask.dtype, device=device)
            input_mask = torch.cat([self_mask, cross_mask], dim=-1)
        
        # classic scaled dot-product attention
        dots = torch.einsum('bhid,bhjd->bhij', q * self.scale, k)
        
        # might need to tune MASK_VAL for fp16 to work
        if exists(input_mask):
            dots.masked_fill_(~input_mask, MASK_VAL)
            del input_mask

        if self.causal:
            i, j = torch.triu_indices(n, n, 1)
            dots[:,:,i,j] = MASK_VAL

        attn = F.softmax(dots, -1)
        if self.store_attention: # and not self.training
            self.attention = attn.detach().cpu()
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

    def _init(self):
        [nn.init.xavier_uniform_(w) for w in [self.to_q.weight, self.to_kv.weight, self.to_out.weight]]
        if getattr(self.to_q, 'bias', None) is not None: nn.init.constant_(self.to_q.bias, 0)
        if getattr(self.to_kv, 'bias', None) is not None: nn.init.constant_(self.to_kv.bias, 0)
        nn.init.constant_(self.to_out.bias, 0)


"""## Transformer blocks

### Encoder
"""

class TransformerEncoderBlock(Module):
    """
    Bacis transformer encoder block. Consists of multi-head attention and positional feedforward layers
    """
    def __init__(self, 
                 d_model, 
                 heads = 8, 
                 d_ff = None, 
                 attn_dropout = 0.1,
                 ff_dropout = 0.1,
                 causal = False, 
                 mask = None, 
                 attn_bias = True, 
                 prenorm=False):
        store_attr('attn_dropout') # mb separate argument attn_post_dropout
        if prenorm:
            self.attn = Residual(PreNorm(d_model, Attention(d_model, heads=heads, causal=causal, dropout=attn_dropout, bias=attn_bias)))
            self.ff = Residual(PreNorm(d_model, FeedForward(d_model, d_ff=d_ff, dropout=ff_dropout)))
        else:
            self.attn = PostNorm(d_model, Residual(Attention(d_model, heads=heads, causal=causal, dropout=attn_dropout, bias=attn_bias)))
            self.ff = PostNorm(d_model, Residual(FeedForward(d_model, d_ff=d_ff, dropout=ff_dropout)))
        self.dropout = nn.Dropout(attn_dropout)
        
    def forward(self, x, mask=None): #? more args
        out = self.attn(x, mask=mask)
        out = self.dropout(out)
        return self.ff(out)

    
class TransformerEncoder(Module):
    def __init__(self, 
                 d_model, 
                 n_layers=6, 
                 heads=8, 
                 d_ff=None,
                 ff_dropout=0.1, 
                 attn_dropout=0.1,
                 attn_bias=True,
                 causal=False, 
                 prenorm=False, 
                 final_norm=None):
        store_attr('d_model')
        self.layers = nn.ModuleList([])    
        for _ in range(n_layers):
            self.layers.append(TransformerEncoderBlock(d_model, heads, causal=causal, d_ff=d_ff, 
                                    attn_dropout=attn_dropout, ff_dropout=ff_dropout, prenorm=prenorm, attn_bias=attn_bias))
        self.norm = None if final_norm is None else final_norm(d_model)
        
    def forward(self, x, mask=None):
        for layer in self.layers: x = layer(x, mask=mask)
        if self.norm is not None: x = self.norm(x)
        return x

"""Decoder block has attention and cross attention

### Decoder
"""

class TransformerDecoderBlock(Module):
    def __init__(self, 
                 d_model, 
                 heads = 8, 
                 d_ff = None,
                 attn_dropout = 0.1, 
                 ff_dropout=0.1,
                 mask = None ,
                 attn_bias = True,
                 prenorm=False):
        store_attr('attn_dropout')     # mb separate argument attn_post_dropout
        if prenorm:
            self.attn = Residual(PreNorm(d_model, Attention(d_model, heads=heads, causal=True, dropout=attn_dropout, bias=attn_bias)))
            self.cross = Residual(PreNorm(d_model, Attention(d_model, heads=heads, causal=False, dropout=attn_dropout, bias=attn_bias)))
            self.ff = Residual(PreNorm(d_model, FeedForward(d_model, d_ff=d_ff, dropout=ff_dropout)))
        else:
            self.attn = PostNorm(d_model, Residual(Attention(d_model, heads=heads, causal=True, dropout=attn_dropout, bias=attn_bias)))
            self.cross = PostNorm(d_model, Residual(Attention(d_model, heads=heads, causal=False, dropout=attn_dropout, bias=attn_bias)))
            self.ff = PostNorm(d_model, Residual(FeedForward(d_model, d_ff=d_ff, dropout=ff_dropout)))
        self.dropout = nn.Dropout(attn_dropout)
        
    def forward(self, x, context, mask=None, context_mask=None):
        out = self.attn(x, mask=mask)
        out = self.dropout(out)
        out = self.cross(out, context, mask=mask, context_mask=context_mask)
        out = self.dropout(out)
        return self.ff(out)

    
class TransformerDecoderBlockV2(nn.Module):
    def __init__(self, d_model, heads = 8, mask = None, d_ff=None,
                 attn_dropout=0.1, ff_dropout=0.1, attn_bias=True,
                 prenorm=False):
        super().__init__()
        self.attn_dropout = attn_dropout # mb separate argument attn_post_dropout
        if prenorm:
            self.attn = Residual(PreNorm(d_model, DecoderAttention(d_model, heads=heads, causal=True, dropout=attn_dropout, bias=attn_bias)))
            self.ff = Residual(PreNorm(d_model, FeedForward(d_model, d_ff=d_ff, dropout=ff_dropout)))
        else:
            self.attn = PostNorm(d_model, Residual(DecoderAttention(d_model, heads=heads, causal=True, dropout=attn_dropout, bias=attn_bias)))
            self.ff = PostNorm(d_model, Residual(FeedForward(d_model, d_ff=d_ff, dropout=ff_dropout)))
        
    def forward(self, x, context, mask=None, context_mask=None):
        out = self.attn(x, context, mask=mask, context_mask=context_mask)
        out = F.dropout(out, p=self.attn_dropout)
        out = self.ff(out)
        return out

    
class TransformerDecoder(Module):
    def __init__(self, 
                 d_model, 
                 n_layers=6, 
                 heads=8, 
                 d_ff=None, 
                 attn_dropout=0.1, 
                 ff_dropout=0.1, 
                 prenorm=False, 
                 comb_attn=False, 
                 attn_bias=True, 
                 final_norm=None):
        store_attr('d_model')
        block = TransformerDecoderBlockV2 if comb_attn else TransformerDecoderBlock            #TODO(Arto) refactor
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(block(d_model, heads, d_ff=d_ff, attn_dropout=attn_dropout, ff_dropout=ff_dropout, prenorm=prenorm, attn_bias=attn_bias))
        self.norm = None if final_norm is None else final_norm(d_model)
        
    def forward(self, x, context, mask=None, context_mask=None):
        for layer in self.layers: x = layer(x, context, mask, context_mask)
        if self.norm is not None: x = self.norm(x)
        return x

"""### Models"""
# from https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reformer_pytorch.py#L609

class AbsolutePositionalEmbedding(Module):
    def __init__(self, d_emb, max_seq_len):
        self.emb = nn.Embedding(max_seq_len, d_emb)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)

    
class FixedPositionalEmbedding(Module):
    def __init__(self, d_emb):
        inv_freq = 1. / (10000 ** (torch.arange(0, d_emb, 2).float() / d_emb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i, j -> i j", t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]
    
#TODO add axial positional encodings


class TransformerEmbedding(Module):
    """
    Combines token embedings with positional encodings
    pos_enc: str from {'absolute', 'fixed', 'axial'}
    """
    def __init__(self, 
                 emb_sz, 
                 d_emb, 
                 max_seq_len=512, 
                 dropout=0., 
                 pos_enc='absolute', 
                 axial_shape=None, 
                 axial_emb_dims=None):
        store_attr('d_emb')
        self.scale = d_emb ** 0.5
        self.std = 0.02    # fairseq: d_emb ** -0.5, fastai: 0.01
        self.emb = nn.Embedding(emb_sz, d_emb)
        self.dropout = nn.Dropout(dropout)
        
        if pos_enc == 'absolute': self.pos_enc = AbsolutePositionalEmbedding(d_emb, max_seq_len)
        elif pos_enc == 'fixed': self.pos_enc = FixedPositionalEmbedding(d_emb)
        elif pos_enc == 'axial':
            assert axial_shape is not None
            assert reduce(mul, axial_shape) == max_seq_len
            axial_emb_dims = default(axial_emb_dims, get_axial_dims(d_emb, len(axial_shape)))
            self.pos_enc = AxialPositionalEmbedding(d_emb, axial_shape, axial_emb_dims)
        self._init()
        
    def forward(self, x):
        x = self.emb(x)  #* self.scale
        x *= self.scale 
        x += self.pos_enc(x)
        return self.dropout(x)
    
    def _init(self):
        nn.init.trunc_normal_(self.emb.weight, std = self.std)
        if hasattr(self.pos_enc, 'weight'): nn.init.trunc_normal_(self.pos_enc.weight, std = self.std)

#TODO test weight tying
# Note on weight tying: it's done like here in fastai AWD_LSTM model
# Lucidrains does it with custom MatrixMultiply module https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reformer_pytorch.py#L106
#TODO: update docstrings
class TransformerEncDec(Module):
    """
    Basic Transformer Encoder-Decoder model
    Parameters:
        * enc_vocab_sz: int - source vocab size 
        * dec_vocab_sz: int - target vocab size
        * d_model: int - inner dimension of the model
        * n_enc_layers: int (default: 6) 
        * n_dec_layers: int (default: 6) 
        * heads: int (default: 8)
        * d_ff: int - inner dimension of the pointwise FeedForward net, if None defaults to 4*d_model
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
                 n_enc_layers=6, 
                 n_dec_layers=6, 
                 heads=8, 
                 d_ff=None,
                 pad_idx=None, 
                 tie_weights=True,
                 shared_emb = False,
                 attn_dropout=0.1, 
                 ff_dropout=0.1, 
                 emb_dropout=0.1,
                 prenorm=False, 
                 attn_bias=True,
                 comb_attn=False, 
                 pos_enc='absolute', 
                 max_seq_len=512, 
                 axial_shape=None, 
                 axial_emb_dims=None):
        store_attr('max_seq_len, n_enc_layers, n_dec_layers, pad_idx')
        self.enc_emb = TransformerEmbedding(enc_vocab_sz, d_model, max_seq_len, dropout=emb_dropout, pos_enc=pos_enc,
                                            axial_shape=axial_shape, axial_emb_dims=axial_emb_dims)
        if shared_emb:
            assert (enc_vocab_sz == dec_vocab_sz), "Encoder and decoder vocab size doesn't match"
            self.dec_emb = self.emc_emb
        else:
            self.dec_emb = TransformerEmbedding(dec_vocab_sz, d_model, max_seq_len, dropout=emb_dropout, pos_enc=pos_enc,
                                                axial_shape=axial_shape, axial_emb_dims=axial_emb_dims)
        
        self.encoder = TransformerEncoder(d_model, n_enc_layers, heads, d_ff=d_ff, attn_dropout=attn_dropout, ff_dropout=ff_dropout, 
                                          prenorm=prenorm, attn_bias=attn_bias, final_norm=nn.LayerNorm, causal=False)
        self.decoder = TransformerDecoder(d_model, n_dec_layers, heads, d_ff=d_ff, attn_dropout=attn_dropout, ff_dropout=ff_dropout, 
                                          prenorm=prenorm, comb_attn=comb_attn, attn_bias=attn_bias, final_norm=nn.LayerNorm)
        self.proj = nn.Linear(d_model, dec_vocab_sz)
        if tie_weights: self.proj.weight = self.dec_emb.emb.weight

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_mask = default(src_mask, self.get_padding_mask(src))
        tgt_mask = default(tgt_mask, self.get_padding_mask(tgt))
        enc = self.encoder(self.enc_emb(src), mask=src_mask)
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
                filtered_logits = sampler(logits, thresh)
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
        layer_ids = default(layer_ids, list(range(self.n_enc_layers)))
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

class TransformerLM(Module):
    """
    Basic Transformer for language modelling
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
                 n_layers=6,
                 heads=8,
                 d_ff=None,
                 attn_dropout=0.1,
                 ff_dropout=0.1,
                 emb_dropout=0.1,
                 tie_weights=True,
                 causal=True,
                 pos_enc='absolute',
                 max_seq_len=512,
                 axial_shape=None,
                 axial_emb_dims=None,
                 pad_idx=None,
                 prenorm=False,
                 attn_bias=True):
        store_attr('max_seq_len, n_layers, pad_idx')
        self.emb = TransformerEmbedding(vocab_sz, d_model, max_seq_len, dropout=emb_dropout, pos_enc=pos_enc,
                                        axial_shape=axial_shape, axial_emb_dims=axial_emb_dims)
        self.tfmr = TransformerEncoder(d_model, n_layers, heads, causal=causal, d_ff=d_ff, 
                                       attn_dropout=attn_dropout, ff_dropout=ff_dropout,
                                       prenorm=prenorm, attn_bias=attn_bias, final_norm=nn.LayerNorm)
        self.proj = nn.Linear(d_model, vocab_sz)
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
        layer_ids = default(layer_ids, list(range(self.n_layers)))
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
