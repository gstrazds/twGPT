import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):  # over last dimension of logits (by default)
    # logits.shape == (batch_size,vocab_size)  #(b,v)
    v, ix = torch.topk(logits, k)
    # v is a tensor like logits, but reduced to size k along last dimension -- v.shape=(b,k)
    # ix is vector of indices into logits, corresponding to the values in v
    #   (such that for each i in 0..k-1, logits[...,ix[i]] == v[...,i])
    out = logits.clone()  # avoid trashing the original tensor
    # v[:,[-1]] is a tensor w/shape=(b,1) -- let's call it min_of_topk
    #   whose elements are the min of the k largest values for each batch entry
    out[out < v[:, [-1]]] = -float('Inf')  # replace logits smaller than the min_of_topk value
    return out

@torch.no_grad()
def sample(model, block_size, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    # block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        logits = logits[:, -1, :]  # use the logits from the last seq pos
        ix = tokid_from_logits(logits, temperature=temperature, sample=sample, top_k=top_k)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x

@torch.no_grad()
def tokid_from_logits(logits, temperature=1.0, sample=False, top_k=None):
    logits / temperature
    # print(f"sample: logits.size = {logits.size()}")
    # optionally crop logits to only the top k options
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    # apply softmax to convert to probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution or take the most likely
    if sample:
        ix = torch.multinomial(probs, num_samples=1)
    else:
        _, ix = torch.topk(probs, k=1, dim=-1)
    return ix

