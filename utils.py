import numpy
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def compute_utility(total_reward, threshold, u_type, u_order):
    if u_type == 1:
        if total_reward - threshold >= 0:
            return 1
        else:
            return 0
    elif u_type == 2:
        return 1 - threshold**(- 1/u_order) * (numpy.maximum(0, threshold - total_reward))**(1/u_order)
    else:
        return (1 + numpy.exp(-u_order * (1 - threshold))) / (1 + numpy.exp(-u_order * (total_reward - threshold)))


def plot_data(y_data, xlabel, ylabel, filename, x_data=None, ylim=None, linewidth=4, fill_bounds=None):
    """
    Generic plotting function to handle repetitive plotting tasks.
    """
    plt.figure(figsize=(8, 6))
    x_data = x_data if x_data is not None else range(len(y_data))
    plt.plot(x_data, y_data, linewidth=linewidth)
    if fill_bounds:
        lower_bound, upper_bound = fill_bounds
        plt.fill_between(x_data, lower_bound, upper_bound, color='blue', alpha=0.2)
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    if ylim:
        plt.ylim(ylim)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.grid(True)
    plt.savefig(filename, format="pdf")
    plt.close()


def compute_bounds(perf_ref, perf_lrn):
    """
    Computes regret and confidence bounds.
    """
    avg_creg = numpy.mean(numpy.cumsum(numpy.sum(perf_ref - perf_lrn, axis=2), axis=1), axis=0)
    std_creg = numpy.std(numpy.cumsum(numpy.sum(perf_ref - perf_lrn, axis=2), axis=1), axis=0)
    avg_reg = [avg_creg[k] / (k + 1) for k in range(len(avg_creg))]
    return avg_reg, avg_creg, (avg_creg - std_creg, avg_creg + std_creg)


def process_and_plot(prob_err, indx_err, perf_ref, perf_lrn, suffix, path, key_value):
    """
    Processes data and generates all required plots for a given suffix.
    """
    trn_err = numpy.mean(prob_err, axis=(0, 2))
    wis_err = numpy.mean(indx_err, axis=(0, 2))
    reg, creg, bounds = compute_bounds(perf_ref, perf_lrn)

    plot_data(trn_err, 'Episodes', 'Max Probability Error', f'{path}per_{suffix}_{key_value}.pdf')
    plot_data(wis_err, 'Episodes', 'Max WI Error', f'{path}wer_{suffix}_{key_value}.pdf')
    plot_data(creg, 'Episodes', 'Cumulative Regret', f'{path}cumreg_{suffix}_{key_value}.pdf')
    plot_data(creg, 'Episodes', 'Cumulative Regret', f'{path}cumregbounds_{suffix}_{key_value}.pdf', fill_bounds=bounds)
    plot_data(reg, 'Episodes', 'Regret', f'{path}reg_{suffix}_{key_value}.pdf')


def inverse_utility(U, threshold, u_type, u_order):
    if u_type == 1:
        return threshold  # CE is the quantile at expected U
    elif u_type == 2:
        return threshold - (threshold**u_order * (1 - U)**u_order)**(1/u_order)
    else:
        return threshold + (1/u_order) * numpy.log((1 + numpy.exp(-u_order * (1 - threshold))) / U - 1)
   