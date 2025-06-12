import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

def inverse_utility(utility, threshold, u_type, u_order):
    if u_type == 1:
        return threshold  # CE is the quantile at expected U
    elif u_type == 2:
        return threshold - (threshold**u_order * (1 - utility)**u_order)**(1/u_order)
    else:
        return threshold + (1/u_order) * numpy.log((1 + numpy.exp(-u_order * (1 - threshold))) / utility - 1)
  
def plot_reg(y_data, xlabel, ylabel, filename, x_data=None, ylim=None, linewidth=4, fill_bounds=None):
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

def plot_data(ys_data_dict, xlabel, ylabel, filename, x_data=None, ylim=None, linewidth=4, fill_bounds_dict=None):
    """
    Generic plotting function to handle repetitive plotting tasks.
    """
    plt.figure(figsize=(8, 6))
    
    # Define a list of colors to use for different lines using a colormap
    colors = cm.get_cmap('tab10', len(ys_data_dict.keys()))
    
    for i, (label, y_data) in enumerate(ys_data_dict.items()):
        x_data_to_plot = x_data if x_data is not None else range(len(y_data))
        plt.plot(x_data_to_plot, y_data, linewidth=linewidth, label=label, color=colors(i))
        
        if fill_bounds_dict and label in fill_bounds_dict:
            lower_bound, upper_bound = fill_bounds_dict[label]
            plt.fill_between(x_data_to_plot, lower_bound, upper_bound, color=colors(i), alpha=0.2)
            
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    
    if ylim:
        plt.ylim(ylim)
        
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.grid(True)
    plt.legend(fontsize=12)
    
    plt.savefig(filename, format="pdf")
    plt.close()

def compute_bounds(perf_ref, perf_lrn):
    """
    Computes regret and confidence bounds.
    """
    avg_creg = numpy.mean(numpy.cumsum(numpy.sum(perf_ref - perf_lrn, axis=2), axis=1), axis=0)
    std_creg = numpy.std(numpy.cumsum(numpy.sum(perf_ref - perf_lrn, axis=2), axis=1), axis=0)
    min_creg = numpy.min(numpy.cumsum(numpy.sum(perf_ref - perf_lrn, axis=2), axis=1), axis=0)
    max_creg = numpy.max(numpy.cumsum(numpy.sum(perf_ref - perf_lrn, axis=2), axis=1), axis=0)
    avg_reg = [avg_creg[k] / (k + 1) for k in range(len(avg_creg))]
    return avg_reg, avg_creg, (avg_creg - std_creg, avg_creg + std_creg), (avg_creg - 3*std_creg, avg_creg + 3*std_creg), (min_creg, max_creg)

def process_and_plot(prob_err, indx_err, perf_ref, perf_lrn, suffix, path, key_value):
    """
    Processes data and generates all required plots for a given suffix.
    """
    trn_err = numpy.mean(prob_err, axis=(0, 2))
    wis_err = numpy.mean(indx_err, axis=(0, 2))
    reg, creg, bounds, boundstri, minmax = compute_bounds(perf_ref, perf_lrn)

    plot_reg(trn_err, 'Episodes', 'Max Probability Error', f'{path}per_{suffix}_{key_value}.pdf')
    plot_reg(wis_err, 'Episodes', 'Max WI Error', f'{path}wer_{suffix}_{key_value}.pdf')
    plot_reg(creg, 'Episodes', 'Cumulative Regret', f'{path}cumreg_{suffix}_{key_value}.pdf')
    plot_reg(creg, 'Episodes', 'Cumulative Regret', f'{path}cumregbounds_{suffix}_{key_value}.pdf', fill_bounds=bounds)
    plot_reg(creg, 'Episodes', 'Cumulative Regret', f'{path}cumregboundstri_{suffix}_{key_value}.pdf', fill_bounds=boundstri)
    plot_reg(creg, 'Episodes', 'Cumulative Regret', f'{path}cumregminmax_{suffix}_{key_value}.pdf', fill_bounds=minmax)
    plot_reg(reg, 'Episodes', 'Regret', f'{path}reg_{suffix}_{key_value}.pdf')


def compute_regbounds_inf(perf_ref, perf_lrn, perf_bas=None):
    """
    Computes regret and confidence bounds.
    """
    res_lrn = {}
    res_lrn["avg_creg"] = numpy.mean(perf_ref - perf_lrn, axis=0)
    res_lrn["std_creg"] = numpy.std(perf_ref - perf_lrn, axis=0)
    res_lrn["min_creg"] = numpy.min(perf_ref - perf_lrn, axis=0)
    res_lrn["max_creg"] = numpy.max(perf_ref - perf_lrn, axis=0)
    res_lrn["avg_reg"] = [res_lrn["avg_creg"][k] / (k + 1) for k in range(len(res_lrn["avg_creg"]))]
    output_lrn = (
        res_lrn["avg_reg"], res_lrn["avg_creg"], 
        (res_lrn["avg_creg"] - res_lrn["std_creg"], res_lrn["avg_creg"] + res_lrn["std_creg"]), 
        (res_lrn["min_creg"], res_lrn["max_creg"])
    )

    output_bas = None
    if perf_bas is not None:
        res_bas = {}
        res_bas["avg_creg"] = numpy.mean(perf_ref - perf_bas, axis=0)
        res_bas["std_creg"] = numpy.std(perf_ref - perf_bas, axis=0)
        res_bas["min_creg"] = numpy.min(perf_ref - perf_bas, axis=0)
        res_bas["max_creg"] = numpy.max(perf_ref - perf_bas, axis=0)
        res_bas["avg_reg"] = [res_bas["avg_creg"][k] / (k + 1) for k in range(len(res_bas["avg_creg"]))]
        output_bas = (
            res_bas["avg_reg"], res_bas["avg_creg"], 
            (res_bas["avg_creg"] - res_bas["std_creg"], res_bas["avg_creg"] + res_bas["std_creg"]), 
            (res_bas["min_creg"], res_bas["max_creg"])
        )
    
    return output_lrn, output_bas

def process_and_plot_inf(prob_err, indx_err, perf_ref, perf_lrn, suffix, path, key_value, perf_bas=None, perf_stt=None):
    """
    Processes data and generates all required plots for a given suffix.
    """
    arg_for_perf_stt = {}
    arg_for_bnds_stt = {}
    
    # trn_err = numpy.mean(prob_err, axis=(0, 2))
    # wis_err = numpy.mean(indx_err, axis=(0, 2))
    arg_for_perf_ref = numpy.tile(numpy.sum(perf_ref, axis=0)[:, numpy.newaxis], (1, perf_lrn.shape[1]))
    arg_for_perf_lrn = numpy.sum(perf_lrn, axis=2)

    arg_for_perf_stt['RAP'] = numpy.mean(arg_for_perf_ref, axis=0)
    arg_for_bnds_stt['RAP'] = (arg_for_perf_stt['RAP']-numpy.std(arg_for_perf_ref, axis=0), arg_for_perf_stt['RAP']+numpy.std(arg_for_perf_ref, axis=0))
    arg_for_perf_stt['RAPTS'] = numpy.mean(arg_for_perf_lrn, axis=0)
    arg_for_bnds_stt['RAPTS'] = (arg_for_perf_stt['RAPTS']-numpy.std(arg_for_perf_lrn, axis=0), arg_for_perf_stt['RAPTS']+numpy.std(arg_for_perf_lrn, axis=0))

    arg_for_perf_bas = None
    if perf_bas is not None:
        arg_for_perf_bas = numpy.sum(perf_bas, axis=2)
        arg_for_perf_stt['WIPTS'] = numpy.mean(arg_for_perf_bas, axis=0)
        arg_for_bnds_stt['WIPTS'] = (arg_for_perf_stt['WIPTS']-numpy.std(arg_for_perf_bas, axis=0), arg_for_perf_stt['WIPTS']+numpy.std(arg_for_perf_bas, axis=0))

    # output_lrn, output_bas = compute_regbounds_inf(arg_for_perf_ref, arg_for_perf_lrn, arg_for_perf_bas)
    # reg, creg, bounds, minmax = output_lrn

    # plot_reg(trn_err, 'Episodes', 'Max Probability Error', f'{path}per_{suffix}_{key_value}.pdf')
    # plot_reg(wis_err, 'Episodes', 'Max WI Error', f'{path}wer_{suffix}_{key_value}.pdf')
    # plot_reg(creg, 'Episodes', 'Cumulative Regret', f'{path}cumreg_{suffix}_{key_value}.pdf')
    # plot_reg(creg, 'Episodes', 'Cumulative Regret', f'{path}cumregbounds_{suffix}_{key_value}.pdf', fill_bounds=bounds)
    # plot_reg(creg, 'Episodes', 'Cumulative Regret', f'{path}cumregminmax_{suffix}_{key_value}.pdf', fill_bounds=minmax)
    # plot_reg(reg, 'Episodes', 'Regret', f'{path}reg_{suffix}_{key_value}.pdf')
    # if output_bas is not None:
    #     reg, creg, bounds, minmax = output_bas
    #     plot_reg(creg, 'Episodes', 'Cumulative Regret', f'{path}basecumreg_{suffix}_{key_value}.pdf')
    #     plot_reg(creg, 'Episodes', 'Cumulative Regret', f'{path}cumregbounds_{suffix}_{key_value}.pdf', fill_bounds=bounds)
    #     plot_reg(creg, 'Episodes', 'Cumulative Regret', f'{path}cumregminmax_{suffix}_{key_value}.pdf', fill_bounds=minmax)
    #     plot_reg(reg, 'Episodes', 'Regret', f'{path}reg_{suffix}_{key_value}.pdf')

    # if perf_stt is not None:
    #     for key, perf_res in perf_stt.items():
    #         mean_of_summed_perf_res = numpy.mean(numpy.sum(perf_res, axis=0))
    #         arg_for_perf_stt[key] = mean_of_summed_perf_res * numpy.ones(perf_lrn.shape[1])
    plot_data(arg_for_perf_stt, 'Episodes', 'Objective', f'{path}perf_{suffix}_{key_value}.pdf', fill_bounds_dict=arg_for_bnds_stt)
 