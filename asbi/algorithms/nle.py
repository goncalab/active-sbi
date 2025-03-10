import sbi 
import torch as th
from sbi.inference import NLE
from sbi.inference import EnsemblePosterior
from asbi.algorithms.acquisitions import bald_acq_func

def run_NLE(simulator, prior, n_sims, density_estimator="maf"):
    """
     Runs neural likelihood estimation 
     for now, we generate the data from the simulator inside the function 
    """
    inference = NLE(prior, density_estimator=density_estimator)
    theta = prior((n_sims,))
    x = simulator(theta)
    _ = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior()

    return posterior

def run_ensemble_NLE(simulator, prior, n_sims, n_ensemble_members=3, density_estimator="maf"):
    """
     Runs neural likelihood estimation 
     for now, we generate the data from the simulator inside the function 
    """
    ensemble = [NLE(prior, density_estimator=density_estimator) for _ in range(n_ensemble_members)]
    theta = prior((n_sims,))
    x = simulator(theta)
    for inference in ensemble:
        _ = inference.append_simulations(theta, x).train()

    posteriors = [inference.build_posterior() for inference in ensemble]
    ensemble_posterior = EnsemblePosterior(posteriors)

    return ensemble_posterior

def run_bald_NLE(simulator, 
                 prior, 
                 n_sims_init, 
                 n_sims_active,
                 theta_pool_size,
                 n_ensemble_members, 
                 density_estimator="maf"):
    
    # intialize ensemble
    ensemble = [NLE(prior, density_estimator=density_estimator) for _ in range(n_ensemble_members)]

    theta_init = prior((n_sims_init,))
    x_init = simulator(theta_init)

    # train ensemble on inital data
    for inference in ensemble:
        _ = inference.append_simulations(theta_init, x_init).train() 
        print(' --- training complete --- ')
    
    # the rest of the simulations will be used for active learning
    for i in range(n_sims_active):
        theta_pool = prior((theta_pool_size,))
        theta_star, _ = bald_acq_func(ensemble, theta_pool, k=1)
        x_star = simulator(theta_star)
        for inference in ensemble:
            _ = inference.append_simulations(theta_star, x_star).train()
            print(' --- training complete --- ')

    print('building ensemble posterior...') 
    posteriors = [inference.build_posterior() for inference in ensemble]
    ensemble_posterior = EnsemblePosterior(posteriors)
    
    print('done!')
    return ensemble_posterior
