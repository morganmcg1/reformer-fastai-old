"""
    Helper functions, including profiling
"""

import torch
from torch import nn
import torch.autograd.profiler as profiler
from fastai.text.all import *
from fastai.basics import *

def do_cuda_timing(f, inp, context=None, n_loops=100):
    '''
        Get timings of cuda modules. Note `self_cpu_time_total` is returned, but
        from experiments this appears to be similar/same to the total CUDA time
        
        f :  function to profile, typically an nn.Module
        inp : required input to f
        context : optional additional input into f, used for Decoder-style modules
    '''
    f.cuda()
    inp = inp.cuda()
    if context is not None: context = context.cuda()
    with profiler.profile(record_shapes=False, use_cuda=True) as prof:
        with profiler.record_function("model_inference"):
            with torch.no_grad():
                for _ in range(n_loops):
                    if context is None: f(inp)
                    else: f(inp, context)
                    torch.cuda.synchronize()
                    
    res = round((prof.key_averages().self_cpu_time_total / 1000) / n_loops, 3)
    print(f'{res}ms')
    return res


def model_performance(n_loops=5, model='arto', dls=None, n_epochs=1, lr=5e-4):
    """
        DEMO CODE ONLY!
        Run training loop to measure timings. Note that the models internally
        should be changed depending on the model you would like to use. 
        You should also adjust the metrics you are monitoring
    """
    acc_ls, ppl_ls =[], []
    for i in range(n_loops):
        # ADD YOUR MODEL(S) INIT HERE
#         if model == 'arto': m = artoTransformerLM(vocab_sz, 512)
#         elif model == 'pt': m = ptTransformerLM(vocab_sz, 512)
#         else: print('model name not correct')
        
        learn = Learner(dls, m,
                    loss_func=CrossEntropyLossFlat(),
                    metrics=[accuracy, Perplexity()]).to_native_fp16()

        learn.fit_one_cycle(n_epochs, lr, wd=0.05)
        
        acc_ls.append(learn.recorder.final_record[2])
        ppl_ls.append(learn.recorder.final_record[3])
    print(f'Avg Accuracy: {round(sum(acc_ls)/len(acc_ls),3)}, std: {np.std(acc_ls)}')
    print(f'Avg Perplexity: {round(sum(ppl_ls)/len(ppl_ls),3)}, std: {np.std(ppl_ls)}')
    print()
    return learn, acc_ls, ppl_ls


def total_params(m):
    """
    Give the number of parameters of a module and if it's trainable or not
    - Taken from Taken from fastai.callback.hook
    """
    params = sum([p.numel() for p in m.parameters()])
    trains = [p.requires_grad for p in m.parameters()]
    return params, (False if len(trains)==0 else trains[0])