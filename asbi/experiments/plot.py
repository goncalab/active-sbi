import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def plot_results(results_df, plots_dir, show=False, intermediate=False):
    """
    Create more informative plots from the results dataframe
    
    Args:
        results_df: DataFrame with columns [repeat, n_sims, method, c2st_mean, c2st_std]
        plots_dir: Directory to save plots
        show: Whether to show plots interactively
        intermediate: Whether this is an intermediate plot during experiment
    """
    if len(results_df) == 0:
        print("Empty results dataframe, skipping plotting")
        return
        
    # Set style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Calculate aggregate statistics across repeats
    agg_df = results_df.groupby(['n_sims', 'method']).agg({
        'c2st_mean': ['mean', 'std', 'count']
    }).reset_index()
    agg_df.columns = ['n_sims', 'method', 'c2st_mean', 'c2st_std', 'count']
    
    # Create color palette
    methods = results_df['method'].unique()
    palette = sns.color_palette('tab10', n_colors=len(methods))
    
    # Plot main results with error bands (standard error)
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    for i, method in enumerate(methods):
        method_data = agg_df[agg_df['method'] == method].sort_values('n_sims')
        x = method_data['n_sims']
        y = method_data['c2st_mean']
        
        # Standard error = std / sqrt(n)
        yerr = method_data['c2st_std'] / np.sqrt(method_data['count'])
        
        plt.plot(x, y, '-o', label=method, color=palette[i])
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.3, color=palette[i])
    
    plt.xscale('log')
    plt.xlabel('Number of Simulations', fontsize=12)
    plt.ylabel('C2ST Score (lower is better)', fontsize=12)
    plt.title('Performance Comparison of Different Methods', fontsize=14)
    plt.legend(title='Method', frameon=True)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Save figure
    file_suffix = "_intermediate" if intermediate else ""
    main_plot_path = f"{plots_dir}/c2st_comparison{file_suffix}.png"
    plt.savefig(main_plot_path, dpi=300)
    
    # Create a boxplot to show distribution of results
    if not intermediate and len(results_df['repeat'].unique()) > 1:
        plt.figure(figsize=(12, 7))
        sns.boxplot(
            data=results_df, 
            x='n_sims', 
            y='c2st_mean', 
            hue='method',
            palette=palette
        )
        plt.xscale('log')
        plt.title('Distribution of C2ST Scores Across Repeats', fontsize=14)
        plt.xlabel('Number of Simulations', fontsize=12)
        plt.ylabel('C2ST Score', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/c2st_distribution.png", dpi=300)
    
    if show:
        plt.show()
    else:
        plt.close('all')
        
    print(f"Plots saved to {plots_dir}")


"""
def plot_results(results, n_sims_array, methods_array, plots_dir, show=False):
    print(f"plots_dir: {plots_dir}")
    plt.ion()  # enable interactive mode
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xscale('log')
    for i, method in enumerate(methods_array):
        # plot in log scale
        mean_values = results[:,:,i].mean(axis=0)
        std_values = results[:,:,i].std(axis=0)
        ax.errorbar(n_sims_array, mean_values, yerr=std_values, label=method)

    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Value')
    ax.legend()
    fig.savefig(f"{plots_dir}/c2st.png")
    if show:
        plt.show()
    return
"""
