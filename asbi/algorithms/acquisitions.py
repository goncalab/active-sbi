import torch as th 
from copy import deepcopy 
from asbi.algorithms.EnsembleFlow import EnsembleFlow

def bald_acq_func(ensemble, theta_pool, k=1):
    """
     Bayesian Active Learning by Disagreement (BALD)
    """
    flows = [deepcopy(inference._neural_net) for inference in ensemble]
    ensemble = EnsembleFlow(flows)
    scores = []
    for theta in theta_pool:
        score = ensemble.compute_bald_score(theta)
        scores.append(score)
    # select theta values with the highest score
    sorted_scores, sorted_indices = th.sort(th.stack(scores, dim=0), descending=True)
    return theta_pool[sorted_indices[:k]], sorted_scores[:k]
