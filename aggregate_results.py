from utils import *
import numpy as np
import constants as keys
import json


# Function used for obtaining a unified result dictionary for the experiments of a specific dataset and dropout rate
def aggregate_dropout_lists(dataset, dropout_rates):

    lr, bs, _ = keys.HYPERPARAMETER_CROSS_DICT[dataset]

    # Defining the lists for mean NLL, AUCC and RMSE, both for embedded (emb) and injected (inj)
    nll_emb_test_original, nll_inj_test_original = [], []
    nll_emb_test_scaled, nll_inj_test_scaled = [], []
    nll_emb_test_optimal, nll_inj_test_optimal = [], []
    aucc_emb_test_original, aucc_inj_test_original = [], []
    aucc_emb_test_scaled, aucc_inj_test_scaled = [], []
    rmse_emb, rmse_inj = [], []

    # Defining the lists for standard deviation of NLL, AUCC and RMSE, both for embedded (emb) and injected (inj)
    std_nll_emb_test_original, std_nll_inj_test_original = [], []
    std_nll_emb_test_scaled, std_nll_inj_test_scaled = [], []
    std_nll_emb_test_optimal, std_nll_inj_test_optimal = [], []
    std_aucc_emb_test_original, std_aucc_inj_test_original = [], []
    std_aucc_emb_test_scaled, std_aucc_inj_test_scaled = [], []
    std_rmse_emb, std_rmse_inj = [], []

    # Iterating over all the dropout rates
    for dr in dropout_rates:

        # Obtaining the folder path for the experiments with (embedded) and without (injected) dropout
        with_dropout_path = os.path.join(obtain_training_folder(dataset, dr, lr, bs), 'cross_validation',
                                         'test_dr{}'.format(str(dr).split('.')[1]))
        without_dropout_path = os.path.join(obtain_training_folder(dataset, 0.0, lr, bs), 'cross_validation',
                                            'test_dr{}'.format(str(dr).split('.')[1]))

        # Loading the result dictionaries for embedded and injected dropout
        a_file = open(os.path.join(with_dropout_path, 'results.json'), "r")
        output = a_file.read()
        with_dropout_results_dict = json.loads(output)
        a_file = open(os.path.join(without_dropout_path, 'results.json'), "r")
        output = a_file.read()
        without_dropout_results_dict = json.loads(output)

        # Loading the NLL and AUCC sub-dictionaries
        nll_emb = with_dropout_results_dict['nll']
        nll_inj = without_dropout_results_dict['nll']
        aucc_emb = with_dropout_results_dict['aucc']
        aucc_inj = without_dropout_results_dict['aucc']

        # Loading the Negative Log-Likelihood
        nll_emb_test_original.append(nll_emb['nll_test_unscaled'])
        nll_inj_test_original.append(nll_inj['nll_test_unscaled'])
        nll_emb_test_scaled.append(nll_emb['nll_test_scaled'])
        nll_inj_test_scaled.append(nll_inj['nll_test_scaled'])
        nll_emb_test_optimal.append(nll_emb['nll_test_perfect'])
        nll_inj_test_optimal.append(nll_inj['nll_test_perfect'])

        # Expected Calibration Error
        aucc_emb_test_original.append(aucc_emb['AUCC_test_unscaled'])
        aucc_inj_test_original.append(aucc_inj['AUCC_test_unscaled'])
        aucc_emb_test_scaled.append(aucc_emb['AUCC_test_scaled'])
        aucc_inj_test_scaled.append(aucc_inj['AUCC_test_scaled'])

        # Mean Square Error
        rmse_emb.append(with_dropout_results_dict['mses']['RMSE_test'])
        rmse_inj.append(without_dropout_results_dict['mses']['RMSE_test'])

        # --- Standard Deviations ---
        # Negative Log Likelihood
        std_nll_emb_test_original.append(with_dropout_results_dict['stds']['std_nll_test_unscaled'])
        std_nll_inj_test_original.append(without_dropout_results_dict['stds']['std_nll_test_unscaled'])
        std_nll_emb_test_scaled.append(with_dropout_results_dict['stds']['std_nll_test_scaled'])
        std_nll_inj_test_scaled.append(without_dropout_results_dict['stds']['std_nll_test_scaled'])
        std_nll_emb_test_optimal.append(with_dropout_results_dict['stds']['std_nll_test_perfect'])
        std_nll_inj_test_optimal.append(without_dropout_results_dict['stds']['std_nll_test_perfect'])

        # Expected Calibration Error
        std_aucc_emb_test_original.append(with_dropout_results_dict['stds']['std_aucc_test_unscaled'])
        std_aucc_inj_test_original.append(without_dropout_results_dict['stds']['std_aucc_test_unscaled'])
        std_aucc_emb_test_scaled.append(with_dropout_results_dict['stds']['std_aucc_test_scaled'])
        std_aucc_inj_test_scaled.append(without_dropout_results_dict['stds']['std_aucc_test_scaled'])

        # Mean Square Error
        std_rmse_emb.append(with_dropout_results_dict['stds']['std_rmse_test'])
        std_rmse_inj.append(without_dropout_results_dict['stds']['std_rmse_test'])

    # Composing the final results dictionary
    out_dict = {
        'nll_emb_test_original': nll_emb_test_original, 'nll_emb_test_scaled': nll_emb_test_scaled,
        'nll_emb_test_optimal': nll_emb_test_optimal, 'nll_inj_test_original': nll_inj_test_original,
        'nll_inj_test_scaled': nll_inj_test_scaled, 'nll_inj_test_optimal': nll_inj_test_optimal,
        'aucc_emb_test_original': aucc_emb_test_original, 'aucc_inj_test_original': aucc_inj_test_original,
        'aucc_emb_test_scaled': aucc_emb_test_scaled, 'aucc_inj_test_scaled': aucc_inj_test_scaled,
        'rmse_emb': rmse_emb, 'rmse_inj': rmse_inj,
        'std_nll_emb_test_original': std_nll_emb_test_original, 'std_nll_inj_test_original': std_nll_inj_test_original,
        'std_nll_emb_test_scaled': std_nll_emb_test_scaled, 'std_nll_inj_test_scaled': std_nll_inj_test_scaled,
        'std_nll_emb_test_optimal': std_nll_emb_test_optimal, 'std_nll_inj_test_optimal': std_nll_inj_test_optimal,
        'std_aucc_emb_test_original': std_aucc_emb_test_original, 'std_aucc_inj_test_original': std_aucc_inj_test_original,
        'std_aucc_emb_test_scaled': std_aucc_emb_test_scaled, 'std_aucc_inj_test_scaled': std_aucc_inj_test_scaled,
        'std_rmse_emb': std_rmse_emb, 'std_rmse_inj': std_rmse_inj
    }

    return out_dict


# Function used for plotting over a single sub-part of the general grid
def get_multiple_plot(fig, pos, xss, yss, title, axis_labels, colors, markers, legend=None, stdss=None, linestyles=None):

    ax = fig.add_subplot(pos[0], pos[1], pos[2])

    # Iterating over all the lines; if stdss are note None we fill in between [mean - std, mean + std]
    for i in range(len(xss)):
        if stdss is None:
            plt.plot(xss[i], yss[i], linestyle='--', color=colors[i], marker=markers[i])
        else:
            xs, ys, stds = np.array(xss[i]), np.array(yss[i]), np.array(stdss[i])
            plt.fill_between(xs, ys + stds, ys - stds, color=colors[i], alpha=0.3)
            plt.plot(xs, ys, linestyle=linestyles[i], color=colors[i])

    # Setting up meta-data
    plt.title(title)
    plt.xticks([0.0, 0.5])
    plt.xlabel(axis_labels['x'])
    plt.ylabel(axis_labels['y'])

    # Plotting the legend
    if legend is not None:
        plt.legend(legend)

    return ax


# Function used for generating the results on the eight UCI datasets
def generate_uci_dropout_table(plot_type=2):
    dropout_rates = keys.DROPOUT_RATES

    # Setting up the figure size and file name
    size = 1
    if plot_type == 0:
        fig = plt.figure(figsize=(10 * size, 8 * size), dpi=240)
        file_name = 'uci_results_rmse_nll'
    elif plot_type == 1:
        fig = plt.figure(figsize=(10 * size, 3 * size), dpi=240)
        file_name = 'uci_results_nll_scaled'
    elif plot_type == 2:
        fig = plt.figure(figsize=(10 * size, 3 * size), dpi=240)
        file_name = 'uci_results_aucc'

    # Iterating over the UCI datasets
    for k, ds in enumerate(keys.UCI_DATASETS_SPLITTED):
        for i, curr_dataset in enumerate(ds):

            # Getting the result list
            out_dict = aggregate_dropout_lists(curr_dataset, dropout_rates)

            # Getting mean of RMSE, NLL and AUCC
            rmse_emb, rmse_inj = out_dict['rmse_emb'], out_dict['rmse_inj']
            nll_emb_test_original = out_dict['nll_emb_test_original']
            nll_inj_test_original = out_dict['nll_inj_test_original']
            nll_emb_test_optimal = out_dict['nll_emb_test_optimal']
            nll_inj_test_optimal = out_dict['nll_inj_test_optimal']
            nll_emb_test_scaled = out_dict['nll_emb_test_scaled']
            nll_inj_test_scaled = out_dict['nll_inj_test_scaled']
            aucc_emb_test_original = out_dict['aucc_emb_test_original']
            aucc_inj_test_original = out_dict['aucc_inj_test_original']
            aucc_emb_test_scaled = out_dict['aucc_emb_test_scaled']
            aucc_inj_test_scaled = out_dict['aucc_inj_test_scaled']

            # Getting standard deviation of NLL, AUCC and RMSE
            std_rmse_emb = out_dict['std_rmse_emb']
            std_rmse_inj = out_dict['std_rmse_inj']
            std_nll_emb_test_original = out_dict['std_nll_emb_test_original']
            std_nll_inj_test_original = out_dict['std_nll_inj_test_original']
            std_nll_emb_test_scaled = out_dict['std_nll_emb_test_scaled']
            std_nll_inj_test_scaled = out_dict['std_nll_inj_test_scaled']
            std_nll_emb_test_optimal = out_dict['std_nll_emb_test_optimal']
            std_nll_inj_test_optimal = out_dict['std_nll_inj_test_optimal']
            std_aucc_emb_test_original = out_dict['std_aucc_emb_test_original']
            std_aucc_inj_test_original = out_dict['std_aucc_inj_test_original']
            std_aucc_emb_test_scaled = out_dict['std_aucc_emb_test_scaled']
            std_aucc_inj_test_scaled = out_dict['std_aucc_inj_test_scaled']

            # Plotting RMSE and original NLL, divided in two intervals (one original and one reduced)
            if plot_type == 0:
                # k manage top/bottom plot (there are 12 plot on top and 12 plots on bottom)
                idx1 = (i + 1) + k * 12     # iterates over the four plots of the first sub-row
                idx2 = idx1 + 4             # iterates over the four plots of the second sub-row
                idx3 = idx1 + 8             # iterates over the four plots of the third sub-row

                # Setting RMSE and NLL only on the right portion of the plots
                rmse_ylabel = 'RMSE' if i == 0 else ''                      # Label for RMSE
                nll_ylabel = 'NLL\n ([0.001, 0.5])' if i == 0 else ''       # Label for the NLL interval [0.001, 0.5]
                nll_extr_ylabel = 'NLL\n ([0.01, 0.5])' if i == 0 else ''   # Label for the NLL interval [0.01, 0.5]

                # Plotting the RMSE, embedded vs injected
                get_multiple_plot(fig, [7, 4, idx1], [dropout_rates, dropout_rates],
                                  [rmse_emb, rmse_inj],
                                  axis_labels={'x': '', 'y': rmse_ylabel}, markers=['s', '^'], title=curr_dataset,
                                  colors=[keys.PALETTE['emb'], keys.PALETTE['inj']],
                                  stdss=[std_rmse_emb, std_rmse_inj], linestyles=['-', '-'])

                # Plotting the NLL on the entire interval [0.001, 0.5] (all dropout values)
                get_multiple_plot(fig, [7, 4, idx2], [dropout_rates, dropout_rates],
                                  [nll_emb_test_original, nll_inj_test_original],
                                  axis_labels={'x': '', 'y': nll_ylabel}, markers=['s', '^'], title='',
                                  colors=[keys.PALETTE['emb'], keys.PALETTE['inj']],
                                  stdss=[std_nll_emb_test_original, std_nll_inj_test_original],
                                  linestyles=['-', '-'])

                # Plotting the NLL on the reduced interval [0.01, 0.5] (i.e. starting from the 5-th dropout rate)
                threshold = 5
                get_multiple_plot(fig, [7, 4, idx3], [dropout_rates[threshold:], dropout_rates[threshold:]],
                                  [nll_emb_test_original[threshold:], nll_inj_test_original[threshold:]],
                                  axis_labels={'x': '', 'y': nll_extr_ylabel}, markers=['s', '^'], title='',
                                  colors=[keys.PALETTE['emb'], keys.PALETTE['inj']],
                                  stdss=[std_nll_emb_test_original[threshold:], std_nll_inj_test_original[threshold:]],
                                  linestyles=['-', '-'])

            # Plotting the NLL comparing injected and embedded dropout
            if plot_type == 1:
                idx1 = (i + 1) + k * 4
                ylabel = 'NLL' if i == 0 else ''
                get_multiple_plot(fig, [2, 4, idx1], [dropout_rates[3:], dropout_rates[3:],
                                                      dropout_rates, dropout_rates],
                                  [nll_emb_test_scaled[3:], nll_inj_test_scaled[3:],
                                   nll_emb_test_optimal, nll_inj_test_optimal],
                                  axis_labels={'x': '', 'y': ylabel}, markers=['x', 'x', 's', 's'], title=curr_dataset,
                                  colors=[keys.PALETTE['emb_opt'], keys.PALETTE['inj_opt'],
                                          keys.PALETTE['emb_opt'], keys.PALETTE['inj_opt']],
                                  stdss=[std_nll_emb_test_scaled[3:], std_nll_inj_test_scaled[3:],
                                         std_nll_emb_test_optimal, std_nll_inj_test_optimal],
                                  linestyles=[':', ':', '-', '-'])

            # Plotting the AUCC
            if plot_type == 2:
                idx1 = (i + 1) + k * 4
                ylabel = 'AUCC' if i == 0 else ''
                get_multiple_plot(fig, [2, 4, idx1], [dropout_rates, dropout_rates, dropout_rates, dropout_rates],
                                  [aucc_emb_test_original, aucc_emb_test_scaled,
                                   aucc_inj_test_original, aucc_inj_test_scaled],
                                  axis_labels={'x': '', 'y': ylabel}, markers=['x', 'x', 's', 's'], title=curr_dataset,
                                  colors=[keys.PALETTE['emb'], keys.PALETTE['emb'],
                                          keys.PALETTE['inj'], keys.PALETTE['inj']],
                                  stdss=[std_aucc_emb_test_original, std_aucc_emb_test_scaled,
                                         std_aucc_inj_test_original, std_aucc_inj_test_scaled],
                                  linestyles=['-.', '-', '-.', '-'])

    # Setting tight layout and saving the plot
    fig.tight_layout()
    plt.savefig(file_name)


generate_uci_dropout_table(plot_type=0)
generate_uci_dropout_table(plot_type=1)
generate_uci_dropout_table(plot_type=2)

