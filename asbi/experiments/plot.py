import matplotlib.pyplot as plt

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
