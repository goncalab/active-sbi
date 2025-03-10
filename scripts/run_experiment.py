import os
import sys
import yaml
import argparse
import numpy as np
from tqdm import tqdm, trange
import torch as th 
from typing import Dict, Any, List, LiteralString
from asbi.tasks import get_task
from asbi.algorithms.nle import run_NLE, run_ensemble_NLE, run_bald_NLE
from sbibm.metrics import c2st
import matplotlib.pyplot as plt
import datetime
import pickle as pk
from plotting import plot_results

class Runner:
    def __init__(self, config: Dict[str, Any], exp_name: str) -> None:
        self.config = config
        # get task, simulator, and prior
        self.task = get_task(self.config['task'])
        self.simulator = self.task.get_simulator()
        self.prior = self.task.get_prior_dist()
        # set up directories for outputs
        base_dir = f"../results/{exp_name}"
        self.output_dir = f"{base_dir}/{datetime.datetime.now()}"
        self.results_dir = f"{self.output_dir}/results"
        self.plots_dir = f"{self.output_dir}/plots"
        if not os.path.exists(self.results_dir):
            print('creating outputs directories...')
            for dir in [base_dir, self.output_dir, self.results_dir, self.plots_dir]:
                if not os.path.exists(dir):
                    os.mkdir(dir)

    def run(self) -> None:
        """
         Loads configuration and initiate experiment
        """
        # load information from configuration 
        n_sims_array = self.config['n_sims']
        methods_array = self.config['methods']
        n_repeats = self.config['n_repeats']
        n_evals = self.config['n_evals']

        # run pipeline 
        results = self.run_multiple_experiments(n_sims_array, methods_array, n_repeats, n_evals)

        # pickle results
        with open(f"{self.results_dir}/results.pkl", 'wb') as f:
            print('saving results...')
            pk.dump(results, f)

        # plot results
        print("plotting results...")
        plot_results(results, n_sims_array, methods_array, self.plots_dir)
    
        return 
    
    def run_multiple_experiments(self, n_sims_array: List , methods_array: List, n_repeats: int, n_eval: int) -> th.Tensor:
        """
         run multiple experiments
        """
        print('running multiple experiments...')
        results = th.empty((n_repeats, len(n_sims_array), len(methods_array)))
        for i in trange(n_repeats):
            print(f"\n=== Repeat: {i+1} ===")
            for  j, n_sims in enumerate(tqdm(n_sims_array)):
                for k, method in enumerate(methods_array):
                   results[i, j, k], _ = self.run_one_experiment(n_sims, method, n_eval)

        return results
    

    def run_one_experiment(self, n_sims: int, method: LiteralString, n_eval: int):
        """
         run 1 experiment with 1 method. Eval on 10 true obs
         returns c2st 
        """
        if n_eval > 10:
            print('n_eval too large \nEvaluating on 10 obs...')
            n_eval = 10

        if method == 'NLE':
            posterior = run_NLE(self.simulator, self.prior, n_sims)

        elif method == 'EnsembleNLE':
            try:
                n_ensemble_members = self.config['n_ensemble_members']
            except KeyError:
                print('n_ensemble_members not found in config. Using default value of 3')
                n_ensemble_members = 3
                
            posterior = run_ensemble_NLE(self.simulator, self.prior, n_sims, n_ensemble_members=n_ensemble_members)

        elif method == 'BALD_NLE':
            try:
                n_sims_init = self.config['n_sims_init']
                n_sims_active = self.config['n_sims_active']
                theta_pool_size = self.config['theta_pool_size']
                n_ensemble_members = self.config['n_ensemble_members']
            except KeyError:
                print('missing parameters for BALD_NLE in config')
                sys.exit(1)
            posterior = run_bald_NLE(self.simulator, self.prior, n_sims_init, n_sims_active, theta_pool_size, n_ensemble_members)

        else:
            print(f'method: {method} not found')
            sys.exit(1)

        print('running evaluation...')
        c2st_accuracy = th.empty(n_eval)
        for i in range(1, n_eval + 1):
            reference_samples = self.task.get_reference_posterior_samples(num_observation=i)
            obs = self.task.get_observation(num_observation=i)
            posterior_samples = posterior.sample((len(reference_samples),), x=obs)
            c2st_accuracy[i-1] = c2st(reference_samples, posterior_samples)

        return c2st_accuracy.mean(), c2st_accuracy.std()
    


def load_config(config_path: str) -> Dict[str, Any]:

    """
     Load and parse the YAML configuration file
    """
    try:
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error reading config file: {e}")
        sys.exit(1)


def main():
    # set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    
    # load configuration
    print('loading config...')
    config = load_config(f"../configs/{args.config}.yaml")
    print(f'config: \n{config}')
    
    # initialize and run the program
    print('initializing runner...')
    program = Runner(config, args.config)
    program.run()

if __name__ == '__main__':
    main()