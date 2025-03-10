import torch as th 
import numpy as np 


def flow_entropy(flow, condition, n_samples=1000):
    """
     get MC estimate of the entropy of a flow
    """
    ## is it sensible that i'm getting negative values here?
    with th.no_grad():
        samples = flow.sample((n_samples,), condition.unsqueeze(0))
        return -th.mean(flow.log_prob(samples, condition.unsqueeze(0)))