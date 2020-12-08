from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states
from functools import wraps

from basic_transformer import *


# Allow each chunk to attend within itself, and also one chunk back. Chunk
# boundaries might occur in the middle of a sequence of items from the
# same bucket, so this increases the chances of attending to relevant items.

# Solution for 'RuntimeError: Expected object of device type cuda but got device type cpu for argument #2 'mat2' in call to _th_bmm_out' from here
# https://discuss.pytorch.org/t/runtimeerror-expected-object-of-device-type-cuda-but-got-device-type-cpu-for-argument-2-mat1-in-call-to-th-addmm/75690
# explicity indicate device while using torch.arange and torch.randn

def look_one_back(x):
    x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
    return torch.cat([x, x_extra], dim=2)

def sort_key_val(t1, t2, dim=-1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)

def default(val, default_val):
    return default_val if val is None else val

def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tensor.reshape(-1, last_dim)
    summed_tensors = [c.sum(dim=-1) for c in tensor.chunk(chunks, dim=0)]
    return torch.cat(summed_tensors, dim=0).reshape(orig_size)

def cache_method_decorator(cache_attr, cache_namespace, reexecute = False):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, key_namespace=None, fetch=False, set_cache=True, **kwargs):
            namespace_str = str(default(key_namespace, ''))
            _cache = getattr(self, cache_attr)
            _keyname = f'{cache_namespace}:{namespace_str}'

            if fetch:
                val = _cache[_keyname]
                if reexecute:
                    fn(self, *args, **kwargs)
            else:
                val = fn(self, *args, **kwargs)
                if set_cache:
                    setattr(self, cache_attr, {**_cache, **{_keyname: val}})
            return val
        return wrapper
    return inner_fn


def process_inputs_chunk(fn, chunks=1, dim=0):
    def inner_fn(*args, **kwargs):
        keys, values, len_args = kwargs.keys(), kwargs.values(), len(args)
        chunked_args = list(zip(*map(lambda x: x.chunk(chunks, dim=dim), list(args) + list(values))))
        all_args = map(lambda x: (x[:len_args], dict(zip(keys, x[len_args:]))), chunked_args)
        outputs = [fn(*c_args, **c_kwargs) for c_args, c_kwargs in all_args]
        return tuple(map(lambda x: torch.cat(x, dim=dim), zip(*outputs)))
    return inner_fn

TOKEN_SELF_ATTN_VALUE = -1e4 # carefully set for half precision to work

class LSHAttention(Module):
    def __init__( self,
                  dropout = 0.,                       # attention matrix dropout
                  bucket_size = 64,                   # at least 64 suggested in trax
                  n_hashes = 8,                       # papers sugests 8
                  causal = False,
                  allow_duplicate_attention = False,  # as in the paper
                  attend_across_buckets = False,      # as in the paper
                  drop_for_hash_rate = 0.0,           # unsure of default, not mentioned in paper
                  return_attn = False,
                  **kwargs):
        
        if dropout >= 1.0 or drop_for_hash_rate >=1.0:
            raise ValueError('Dropout rates must be lower than 1.')
        
        store_attr(but=['dropout', 'drop_for_hash_rate'])  # fastcore - store attibutes
        self.dropout = nn.Dropout(dropout)
        self.dropout_for_hash = nn.Dropout(drop_for_hash_rate)
        self._cache = {} # cache buckets for reversible network, required to make Reformer work at depth

    @cache_method_decorator('_cache', 'buckets', reexecute=True)
    def hash_vectors(self, n_buckets, vecs):
        # 0. We need an even number of buckets: 
        assert n_buckets % 2 == 0

        # 1. account for the input shapes. vecs = [bs, sl, dim]
        batch_size, seqlen, dim = vecs.shape
        device = vecs.device
        #print(device)
        rotations_shape = (dim, self.n_hashes, n_buckets // 2)

        # 2. Calculate hash bucket id via random rotations, concatenation and argmax 
        # note: we copy rotations accross batch dimension (see exploration notebook for details). 
		# Imran: added device
        random_rotations = repeat(torch.randn(rotations_shape,device=device), 
                                  'd nh nb -> bs d nh nb', bs=batch_size)           
        dropped_vecs = self.dropout_for_hash(vecs)
                       
        rotated_vecs = torch.einsum('bsd,bdhn->bhsn', 
                                    dropped_vecs,       # [bs, sl, dim]
                                    random_rotations)   # [bs, dim, n_hashes, n_buckets//2]
                                                        # rotated vecs: [bs, n_hashes, sl, n_buckets//2]

        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1) # [bs, n_hashes, sl, n_buckets]
        buckets = torch.argmax(rotated_vecs, dim=-1)                    # [bs, n_hashes, sl] 

        # 3. Next we add offsets so that bucket numbers from different hashing rounds don't overlap.
        # We also reshape the buckets so that each hash round is concatenated along the -1 dim
		# Imran: added device
        offsets = torch.arange(self.n_hashes,device=device)                              # list of [0,1,2,..n_hashes-1]
        offsets = rearrange(offsets * n_buckets, 'nh -> 1 nh 1')        # [1, n_hashes, 1]
        buckets = rearrange(buckets+offsets, 'bs nh sl -> bs (nh sl)')  # [bs, (n_hashes*sl)]
        return buckets

    def forward(self, qk, v, input_mask = None, **kwargs):
        batch_size, seqlen, dim, device = *qk.shape, qk.device
        #print(qk.device)

        # caching
        is_reverse = kwargs.pop('_reverse', False)
        depth = kwargs.pop('_depth', None)
        
        # We will have an even number of buckets, and our attention chunks needs to fit completely within a seqlen
        assert seqlen % (self.bucket_size * 2) == 0, f'Sequence length ({seqlen}) needs to be divisible by target bucket size  x 2 - {self.bucket_size * 2}'
        
        # get the hash buckets for our qk input vectors
        n_buckets = seqlen // self.bucket_size
        buckets = self.hash_vectors(n_buckets, qk, key_namespace=depth, fetch=is_reverse, set_cache=self.training)

        # We use the same vector as both a query and a key.
        assert int(buckets.shape[1]) == self.n_hashes * seqlen
        
        # Create an index that reflexts both bucket id and sequence id. This let's us sort qk according 
        # to both simultaneously. Repeated across the batch dimension.
        ticker = repeat(torch.arange((self.n_hashes * seqlen),device=device), 'l -> bs l', bs=batch_size)
        buckets_and_t = seqlen * buckets + (ticker % seqlen) 
        buckets_and_t = buckets_and_t.detach()                # [bs, seqlen*n_hashes]

        # Hash-based sort ("s" at the start of variable names means "sorted")
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)  # [bs, seqlen*n_hashes]
        _, undo_sort = sticker.sort(dim=-1)                                    # indexes to undo sortings
        del ticker

        sbuckets_and_t = sbuckets_and_t.detach()   # no need to store gradiens for indexes
        sticker = sticker.detach()
        undo_sort = undo_sort.detach()

        st = (sticker % seqlen)             # index of [0..seqlen-1] for each hash round
        sqk = batched_index_select(qk, st)  # get the sorted qk, [bs, seqlen*n_hashes, dim]
        sv = batched_index_select(v, st)    # get the sorted v, [bs, seqlen*n_hashes, dim] 

        # Reshape to include a n_chunks axis.
        n_chunks = self.n_hashes * n_buckets
        bq_t = bkv_t = rearrange(st, 'bs (n s) -> bs n s', n=n_chunks) # [bs, n_chunks, chunk_size]
        bqk = rearrange(sqk, 'bs (n s) d -> bs n s d', n=n_chunks)     # [bs, n_chunks, chunk_size, dim]
        bv = rearrange(sv, 'bs (n s) d -> bs n s d', n=n_chunks)       # [bs, n_chunks, chunk_size, dim]

        # Hashing operates on unit-length vectors. Unnormalized query vectors are
        # fine because they effectively provide a learnable temperature for the
        # attention softmax, but normalizing keys is needed so that similarity for
        # the purposes of attention correctly corresponds to hash locality.
        bq = bqk
        bk = F.normalize(bqk, p=2, dim=-1).type_as(bq)

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        # Note: no look_back for queries

        bk = look_one_back(bk)        # [bs, n_chunks, chunk_size*2, dim]
        bv = look_one_back(bv)        # [bs, n_chunks, chunk_size*2, dim]
        bkv_t = look_one_back(bkv_t)

        # Dot-product attention.
        dots = torch.einsum('bnsd,bnzd->bnsz', 
                    bq,                  # [bs, n_chunks, chunk_size, dim]
                    bk                   # [bs, n_chunks, chunk_size*2, dim]
                   ) * (dim ** -0.5)     # dots: [bs, n_chunks, chunk_size, chunk_size*2]
        masked_value = max_neg_value(dots)

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            input_mask = F.pad(input_mask, (0, seqlen - input_mask.shape[1]), value=True)
            mq = input_mask.gather(1, st).reshape((batch_size, n_chunks, -1))
            mkv = look_one_back(mq)
            mask = mq[:, :, :, None] * mkv[:, :, None, :]
            dots.masked_fill_(~mask, masked_value)
            del mask

        # Causal masking
        if self.causal:
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :]
            dots.masked_fill_(mask, masked_value)
            del mask

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)
        del self_mask

        # Mask out attention to other hash buckets.
        if not self.attend_across_buckets:
            bq_buckets = bkv_buckets = torch.reshape(sbuckets_and_t // seqlen, (batch_size, n_chunks, -1))
            bkv_buckets = look_one_back(bkv_buckets)
            bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
            dots.masked_fill_(bucket_mask, masked_value)
            del bucket_mask

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # There are two possible strategies here. (1) The default is to count how
        # many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition.
        
        if not self.allow_duplicate_attention:
            locs1 = undo_sort // bq_t.shape[-1]
            locs2 = (locs1 + 1) % n_chunks
            if not self.attend_across_buckets:
                locs1 = buckets * n_chunks + locs1
                locs2 = buckets * n_chunks + locs2
            locs = torch.cat([
                torch.reshape(locs1, (batch_size, self.n_hashes, seqlen)),
                torch.reshape(locs2, (batch_size, self.n_hashes, seqlen)),
            ], 1).permute((0, 2, 1))

            slocs = batched_index_select(locs, st)
            b_locs = torch.reshape(slocs, (batch_size, n_chunks, -1, 2 * self.n_hashes))

            b_locs1 = b_locs[:, :, :, None, :self.n_hashes]

            bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, self.n_hashes))
            bq_locs = torch.reshape(bq_locs, b_locs.shape)
            bkv_locs = look_one_back(b_locs)

            dup_counts = (bq_locs[:, :, :, None, :] == bkv_locs[:, :, None, :, :])
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = chunked_sum(dup_counts, chunks=(self.n_hashes * batch_size))
            dup_counts = dup_counts.detach()
            assert dup_counts.shape == dots.shape
            dots = dots - torch.log(dup_counts + 1e-9)
            del dup_counts

        # Softmax.
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = torch.exp(dots - dots_logsumexp).type_as(dots)
        dropped_dots = self.dropout(dots)
        
        # calculate self-attention (attn * values)
        bo = torch.einsum('bnsz,bnzd->bnsd', 
                          dropped_dots,      # [bs, n_chunks, chunk_size, chunk_size*2]
                          bv)                # [bs, n_chunks, chunk_size*2, dim]    
                                             # bo: [bs, n_chunks, chunk_size, dim]
        
        # unchunk, unsort and reshape self-attention
        so = rearrange(bo, 'b n s d -> b (n s) d')                     # [bs, seqlen*n_hashes, dim]
        o = batched_index_select(so, undo_sort)                        # [bs, seqlen*n_hashes, dim]
        o = rearrange(o, 'b (nh sl) d -> b nh sl d', nh=self.n_hashes) # [bs, n_hashes, seqlen, dim]
        
        # unchunk, unsort and reshape logits
        slogits = rearrange(dots_logsumexp, 'bs n s 1 -> bs (n s)')              # [bs, seqlen*n_hashes]
        logits = slogits.gather(1, undo_sort)                                    # [bs, seqlen*n_hashes]
        logits = rearrange(logits, 'bs (nr sl) -> bs nr sl 1', nr=self.n_hashes) # [bs, n_hashes, seqlen, 1]
        
        # average probabilites across hash rounds (dim 1) and get weighted attention
        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True)) # [bs, n_rounds, seqlen, 1]
        out = torch.sum(o * probs, dim=1)                                        # [bs, seqlen, dim]

        # return unsorted attention weights - empty otherwise
        attn = torch.empty(0, device=device)
        if self.return_attn:
            attn_unsort = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            attn_unsort = attn_unsort.view(batch_size * self.n_hashes, -1).long()
            unsorted_dots = torch.zeros(batch_size * self.n_hashes, seqlen * seqlen, device=device)
            unsorted_dots.scatter_add_(1, attn_unsort, dots.view_as(attn_unsort))
            del attn_unsort
            unsorted_dots = unsorted_dots.reshape(batch_size, self.n_hashes, seqlen, seqlen)
            attn = torch.sum(unsorted_dots * probs, dim=1)

        # return output, attention matrix, and bucket distribution
        return out, attn, buckets

class LSHSelfAttention(Module):
    def __init__(self, 
                 dim,                                 # Note: dim refers to model dim/similar to embedding dim for input
                 n_heads = 8, 
                 bucket_size = 64,                    # reccomended default from paper/lucid
                 n_hashes = 8,                        # reccomended default from paper/lucid
                 causal = False, 
                 dim_head = None, 
                 attend_across_buckets = False,      
                 allow_duplicate_attention = False,   # Penalize multiple qk-v pairs in same attention chunk or not
                 return_attn = False,                 # Not implemented yet
                 dropout = 0., 
                 post_attn_dropout = 0.,              # a final dropout on output (not standard)
                 **kwargs):
         
        assert dim_head or (dim % n_heads) == 0, 'dimensions must be divisible by number of heads'
        
        dim_head = default(dim_head, dim // n_heads)  # dim single head
        dim_heads = dim_head * n_heads                # dim all heads
        self.n_heads = n_heads
        
        self.toqk = nn.Linear(dim, dim_heads, bias = False)
        self.tov = nn.Linear(dim, dim_heads, bias = False)
        self.to_out = nn.Linear(dim_heads, dim)
        
        self.lsh_attn = LSHAttention(bucket_size=bucket_size, n_hashes=n_hashes, causal=causal, 
                                     attend_across_buckets = attend_across_buckets,  
                                     allow_duplicate_attention = allow_duplicate_attention, 
                                     return_attn = return_attn, dropout = dropout, **kwargs)
        self.post_attn_dropout = nn.Dropout(post_attn_dropout)

    def forward(self, x, keys = None, input_mask = None, input_attn_mask = None, context_mask = None, **kwargs):
        device, dtype = x.device, x.dtype
        bs, sl, emb_dim = x.shape

        keys = default(keys, torch.empty(bs, 0, emb_dim, dtype=dtype, device=device))
        c = keys.shape[1]

        # project qk and v
        x = torch.cat((x, keys), dim=1)  # [bs, sl+keys.shape[1], dim]
        qk = self.toqk(x)                # [bs, sl, dim_heads (dim_head * heads)]
        v = self.tov(x)                  # [bs, sl, dim_heads]
        
        # split off head dimension for qk and v. Resulting shapes are: [nh, bs, sl, dim_head]
        qk, v = map(lambda t: rearrange(t, 'bs sl (nh dh) -> nh bs sl dh', nh=self.n_heads), (qk, v))
        
        # masks have shape [bs, sl] and are maybe concatenated [bs, sl*2]
        mask = None
        if input_mask is not None or context_mask is not None:
            default_mask = torch.tensor([True], device=device)
            i_mask = default(input_mask, default_mask.expand(bs, sl))
            c_mask = default(context_mask, default_mask.expand(bs, c))
            mask = torch.cat((i_mask, c_mask), dim=1)
        
        # run lsh per head (iterate through 0th dim i.e. the n_head dim), concatenate and rearrange
        # Note: masks are reused per head
        lsh_results = L([self.lsh_attn(qk_h, v_h, mask) for qk_h, v_h in zip(qk, v)])  
        out = lsh_results.itemgot(0)                                   # split tuple (output, attn, buckets)
        out = torch.cat([head for head in out], dim=0)                 # concatenate [n_heads*bs, sl, dh]
        out = rearrange(out, '(nh bs) sl dh -> bs sl (nh dh)', bs=bs)  # [bs, sl, dim_heads] (dim_heads = head_dim * n_heads)
        
        # pass through final feed forward and maybe dropout
        out = self.to_out(out)                                            # [bs, sl, dim]
        return self.post_attn_dropout(out)


class Attention(Module):
    def __init__(self, 
                 d_model, 
                 n_heads = 8, 
                 causal = False,
                 mask = None,
                 dropout=0.1, 
                 bias=True,
                 store_attention=False,
                 **kwargs):
        store_attr('causal, mask, n_heads, store_attention')
        
        self.scale=(d_model//n_heads) ** -0.5
        self.to_q = nn.Linear(d_model, d_model, bias=bias)
        self.to_kv = nn.Linear(d_model, d_model * 2, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(d_model, d_model)
        
        self._init()

    def forward(self, x, context = None, mask = None, context_mask = None, store_attention=False):
        b, n, _, h, device = *x.shape, self.n_heads, x.device
        
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


class ReformerEncoderBlock(Module):
    """
    Bacis transformer encoder block. Consists of multi-head attention and positional feedforward layers
    """
    def __init__(self, 
                 d_model,
                 heads = 8,
                 full_attn = False,
                 d_ff = None,
                 attn_dropout = 0.1,
                 ff_dropout = 0.1,
                 causal = False, 
                 mask = None, 
                 attn_bias = True, 
                 prenorm=False, 
                 bucket_size = 64,                    # reccomended default from paper/lucid
                 n_hashes = 8,                        # reccomended default from paper/lucid
                 attend_across_buckets = False,      
                 allow_duplicate_attention = False,   # Penalize multiple qk-v pairs in same attention chunk or not
                 **kwargs):
        store_attr('attn_dropout') # mb separate argument attn_post_dropout
        attn_module = Attention if full_attn else LSHSelfAttention
        if prenorm:
            self.attn = Residual(PreNorm(d_model, attn_module(d_model, n_heads=heads, causal=causal, dropout=attn_dropout, bias=attn_bias, 
                                                              bucket_size=bucket_size, n_hashes=n_hashes, 
                                                              attend_across_buckets=attend_across_buckets,
                                                              allow_duplicate_attention=allow_duplicate_attention)))
            self.ff = Residual(PreNorm(d_model, FeedForward(d_model, d_ff=d_ff, dropout=ff_dropout)))
        else:
            self.attn = PostNorm(d_model, Residual(attn_module(d_model, n_heads=heads, causal=causal, dropout=attn_dropout, bias=attn_bias, 
                                                              bucket_size=bucket_size, n_hashes=n_hashes, 
                                                              attend_across_buckets=attend_across_buckets,
                                                              allow_duplicate_attention=allow_duplicate_attention)))
            self.ff = PostNorm(d_model, Residual(FeedForward(d_model, d_ff=d_ff, dropout=ff_dropout)))
        
    def forward(self, x, mask=None): #? more args
        out = self.attn(x, mask=mask)
        return self.ff(out)


class ReformerEncoder(Module):
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
                 final_norm=None,
                 full_attn=False,
                 bucket_size = 64,                    # reccomended default from paper/lucid
                 n_hashes = 8,                        # reccomended default from paper/lucid
                 attend_across_buckets = False,      
                 allow_duplicate_attention = False,   # Penalize multiple qk-v pairs in same attention chunk or not
                 **kwargs):
        store_attr('d_model')
        self.layers = nn.ModuleList([])    
        for _ in range(n_layers):
            self.layers.append(ReformerEncoderBlock(d_model, heads, causal=causal, d_ff=d_ff, full_attn=full_attn, 
                                    attn_dropout=attn_dropout, ff_dropout=ff_dropout, prenorm=prenorm, attn_bias=attn_bias,
                                    bucket_size=bucket_size, n_hashes=n_hashes, attend_across_buckets=attend_across_buckets,
                                    allow_duplicate_attention=allow_duplicate_attention))
        self.norm = None if final_norm is None else final_norm(d_model)
        
    def forward(self, x, mask=None):
        for layer in self.layers: x = layer(x, mask=mask)
        if self.norm is not None: x = self.norm(x)
        return x


class ReformerLM(Module):
    """
    Reformer for language modelling using LSH
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
                 attn_bias=True,
                 full_attn=False,
                 bucket_size = 64,                    # reccomended default from paper/lucid
                 n_hashes = 8,                        # reccomended default from paper/lucid
                 attend_across_buckets = False,      
                 allow_duplicate_attention = False):
        store_attr('max_seq_len, n_layers, pad_idx')
        self.emb = TransformerEmbedding(vocab_sz, d_model, max_seq_len, dropout=emb_dropout, pos_enc=pos_enc,
                                        axial_shape=axial_shape, axial_emb_dims=axial_emb_dims)
        self.tfmr = ReformerEncoder(d_model, n_layers, heads, causal=causal, d_ff=d_ff, 
                                       attn_dropout=attn_dropout, ff_dropout=ff_dropout,
                                       prenorm=prenorm, attn_bias=attn_bias, final_norm=nn.LayerNorm,
                                       full_attn=full_attn)
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
