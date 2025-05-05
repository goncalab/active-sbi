import os
import sys
import argparse
import datetime
import pickle as pk
from pprint import pprint
from typing import Any, Dict, List, LiteralString

import pandas as pd
import torch as th
from tqdm import tqdm, trange

from asbi.tasks import get_task
from asbi.algorithms.nle import run_NLE, run_ensemble_NLE, run_bald_NLE
from asbi.experiments.utils import load_config, get_device
from sbibm.metrics import c2st
from asbi.experiments.plot import plot_results

class Runner: 
    def __init__(self, config, output_path) -> None:
        self.config = config
        # get task, simulator, and prior
        self.task = get_task(self.config['task'])
        self.simulator = self.task.get_simulator()
        self.prior = self.task.get_prior_dist()
        
        # set device for computation
        self.device = get_device()
        
        # set up directories for outputs
        self.output_dir = f"{output_path}/{datetime.datetime.now()}"
        self.results_dir = f"{self.output_dir}/results"
        self.plots_dir = f"{self.output_dir}/plots"
        
        # create directories if they do not exist
        if not os.path.exists(self.results_dir):
            print('creating outputs directories...')
            for dir in [output_path, self.output_dir, self.results_dir, self.plots_dir]:
                if not os.path.exists(dir):
                    os.mkdir(dir)

    """
    def run(self) -> None:

        # load information from configuration 
        n_sims_array = self.config['n_sims']
        methods_array = self.config['methods']
        n_repeats = self.config['n_repeats']
        n_evals = self.config['n_evals']

        # run pipeline 
        results = self.run_multiple_experiments(n_sims_array, methods_array, n_repeats, n_evals)

ß        # pickle results
        with open(f"{self.results_dir}/results.pkl", 'wb') as f:
            print('saving results...')
            pk.dump(results, f)

        # plot results
        print("plotting results...")
        plot_results(results, n_sims_array, methods_array, self.plots_dir)
    
        return 

    def run_multiple_experiments(self, n_sims_array: List , methods_array: List, n_repeats: int, n_eval: int) -> th.Tensor:
        print('running multiple experiments...')
        results = th.empty((n_repeats, len(n_sims_array), len(methods_array)))
        for i in trange(n_repeats):
            print(f"\n=== Repeat: {i+1} ===")
            for  j, n_sims in enumerate(tqdm(n_sims_array)):
                for k, method in enumerate(methods_array):
                   results[i, j, k], _ = self.run_one_experiment(n_sims, method, n_eval)
        return results
    """
    def run(self) -> None:
        """
         Loads configuration and initiate experiment
        """
        # load information from configuration 
        n_sims_array = self.config['n_sims']
        methods_array = self.config['methods']
        n_repeats = self.config['n_repeats']
        n_evals = self.config['n_evals']

        # run pipeline and get results dataframe
        results_df = self.run_multiple_experiments(n_sims_array, methods_array, n_repeats, n_evals)

        # save final results
        results_file = f"{self.results_dir}/results.csv"
        print(f'Saving final results to {results_file}')
        results_df.to_csv(results_file, index=False)
        
        # Also save as pickle for compatibility
        with open(f"{self.results_dir}/results.pkl", 'wb') as f:
            print('Saving results as pickle...')
            pk.dump(results_df, f)

        # plot results
        print("Plotting results...")
        plot_results(results_df, self.plots_dir)
    
        return 
    
    def run_multiple_experiments(self, n_sims_array: List , methods_array: List, n_repeats: int, n_eval: int) -> pd.DataFrame:
        """
        Run multiple experiments and save results to a dataframe
        
        Returns:
            pd.DataFrame: Results with columns [repeat, n_sims, method, c2st_mean, c2st_std]
        """
        print('Running multiple experiments...')
        
        # Create empty dataframe
        results_df = pd.DataFrame(columns=['repeat', 'n_sims', 'method', 'c2st_mean', 'c2st_std'])
        
        # Create checkpoint file
        checkpoint_file = f"{self.results_dir}/results_checkpoint.csv"
        
        for i in trange(n_repeats):
            print(f"\n=== Repeat: {i+1}/{n_repeats} ===")
            for j, n_sims in enumerate(tqdm(n_sims_array, desc="Sim sizes")):
                for k, method in enumerate(methods_array):
                    print(f"Running {method} with {n_sims} simulations")
                    c2st_mean, c2st_std = self.run_one_experiment(n_sims, method, n_eval)
                    
                    # Append row to dataframe
                    new_row = pd.DataFrame({
                        'repeat': [i+1],
                        'n_sims': [n_sims],
                        'method': [method],
                        'c2st_mean': [c2st_mean.item()],
                        'c2st_std': [c2st_std.item()]
                    })
                    results_df = pd.concat([results_df, new_row], ignore_index=True)
                    
                    # Save checkpoint after each method
                    results_df.to_csv(checkpoint_file, index=False)
                    
                # Plot intermediate results after each n_sims value
                if j > 0:  # Only plot if we have data for at least two n_sims values
                    plot_results(results_df, self.plots_dir, intermediate=True)

        return results_df    

    def run_one_experiment(self, n_sims: int, method: LiteralString, n_eval: int):
        """
         run 1 experiment with 1 method. Eval on 10 true obs
         returns c2st 
        """
        if n_eval > 10:
            print('n_eval too large \nEvaluating on 10 obs...')
            n_eval = 10

        if method == 'NLE':
            posterior = run_NLE(self.simulator, self.prior, n_sims, device=self.device)

        elif method == 'EnsembleNLE':
            try:
                n_ensemble_members = self.config['n_ensemble_members']

            except KeyError:
                print('n_ensemble_members not found in config. Using default value of 3')
                n_ensemble_members = 3
                
            posterior = run_ensemble_NLE(self.simulator, self.prior, n_sims, n_ensemble_members=n_ensemble_members, device=self.device)

        elif method == 'BALD_NLE':
            try:
                pct_active = self.config['pct_active']
                n_sims_active = int(n_sims * pct_active)
                n_sims_init = n_sims - n_sims_active
                theta_pool_size = self.config['theta_pool_size']
                n_ensemble_members = self.config['n_ensemble_members']

            except KeyError:
                print('missing parameters for BALD_NLE in config')
                sys.exit(1)

            print("running BALD NLE...")
            print(f"n_sims_init: {n_sims_init}, n_sims_active: {n_sims_active}")
            posterior = run_bald_NLE(self.simulator, self.prior, n_sims_init, n_sims_active, theta_pool_size, n_ensemble_members, device=self.device)

        else:
            print(f'method: {method} not found')
            sys.exit(1)

        print('running evaluation...')
        c2st_accuracy = th.empty(n_eval)

        # get c2st for all observation in evaluation set
        for i in range(1, n_eval + 1):
            reference_samples = self.task.get_reference_posterior_samples(num_observation=i)
            obs = self.task.get_observation(num_observation=i)
            posterior_samples = posterior.sample((len(reference_samples),), x=obs)
            c2st_accuracy[i-1] = c2st(reference_samples, posterior_samples)

        # return to mean and std of the c2st across all obs
        return c2st_accuracy.mean(), c2st_accuracy.std()
    

def main():
    # set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    
    # load configuration
    print('loading config...')
    config = load_config(f"../configs/{args.config}.yaml")
    print(f'config:')
    pprint(config)
    
    # initialize and run the program
    print('initializing runner...')
    program = Runner(config, args.config)
    program.run()

if __name__ == '__main__':
    main()
