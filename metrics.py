import numpy as np
import matplotlib.pyplot as plt
import constants as keys


# Implementation of the negative log likelihood approximation for regression problems
def neg_log_likelihood(pred, gt, unc):
    nll_vector = 0.5 * np.log(unc) + 0.5 * ((pred - gt) ** 2 / unc)
    return np.nanmean(nll_vector)


# Function for obtaining the expected and the observed errors (calibration curve)
def observe_calibration_error(mean, var, gt):

    obs = []
    for i, z in enumerate(keys.ZS):
        # Obtaining the standard deviation from the variance
        mean, std = mean, var ** 0.5

        # Verifying how many gt are inside the confidence interval
        score = np.logical_and(mean - std * z <= gt, gt <= mean + std * z)

        # Getting and comparing expected and observed frequencies
        obs.append(np.sum(score) / score.shape[0])

    obs.append(1)
    obs, exp = np.array(obs), np.arange(keys.P_SPLITS + 1) / keys.P_SPLITS
    return obs, exp


# Implementation of the Area Under the Calibration Curve (AUCC) metric
def area_under_calibration_curve(mean, var, gt, return_integral=False):
    # Obtaining the area under the curve
    obs, exp = observe_calibration_error(mean, var, gt)
    aucc = np.sum(np.abs(exp - obs)) / keys.P_SPLITS
    signed_integral = np.sum(exp - obs) / keys.P_SPLITS

    # Returning the AUCC and eventually the value of the integral epx(alpha) - obs(alpha)
    if return_integral:
        return aucc, signed_integral
    else:
        return aucc


# Combining calibration curves of multiple elements
def plot_calibration_curve(means, vars, gt, path_to_save=None, aggregate=False, aggr_titles=None, title=None):
    iterations = len(vars) if aggregate else 1
    col = (keys.PALETTE['org'], keys.PALETTE['scl_nll'], keys.PALETTE['scl_cal'])

    # Plotting the expected calibration bisector
    plt.plot(np.arange(keys.P_SPLITS+1) / keys.P_SPLITS, np.arange(keys.P_SPLITS+1) / keys.P_SPLITS, 'k--', alpha=0.7)

    for k in range(iterations):

        # Getting the 'currents'
        curr_vars = vars[k] if aggregate else vars
        curr_means = means[k] if aggregate else means
        curr_gt = gt[k] if aggregate else gt

        # Obtaining the area under the curve
        obs, exp = observe_calibration_error(curr_means, curr_vars, curr_gt)

        # Plotting the current calibration curve
        plt.plot(obs, exp, color=col[k], linewidth=3)

    # Setting plot meta data
    title = 'Calibration Curve' if title is None else title
    plt.title(title)
    plt.xlabel('Expected Frequency')
    plt.ylabel('Observed Error')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Adding the legend
    if aggregate:
        legend = ['perfect curve']
        for k in range(iterations):
            legend.append(aggr_titles[k])
    else:
        legend = ['perfect curve', 'model curve']
    plt.legend(legend)

    # Showing or saving the image
    if path_to_save is None:
        plt.show()
    else:
        plt.savefig(path_to_save)
        plt.clf()


# Function for obtaining the optimal scale factor on a given set (prediction, ground truth and uncertainty)
def optimal_scaler(pred, gt, unc):
    return np.mean((pred - gt) ** 2 / unc)


# Function for obtaining the optimal relaxed scale factor on a given set
def optimal_calibration(pred, gt, unc):
    stab_ancor = 1e-4
    c = 1
    s_min, s_max = 0, 1
    _, score = area_under_calibration_curve(pred, c * unc, gt, return_integral=True)

    # Finding Bounds
    while score > stab_ancor:
        s_min, s_max = s_max, s_max * 2
        c = s_max
        _, score = area_under_calibration_curve(pred, c * unc, gt, return_integral=True)

    # Update bounds and calculate the score for the first iteration
    c = (s_max + s_min) / 2
    _, score = area_under_calibration_curve(pred, c * unc, gt, return_integral=True)

    res_anchor, repeat = 0, 0
    while (not (-stab_ancor < score < stab_ancor)) and repeat < 25:

        # If smaller, enlarge; if bigger, shrink
        if score > 0:
            s_min = (s_max + s_min) / 2
        elif score < 0:
            s_max = (s_max + s_min) / 2

        # Update bounds and recalculate the score
        c = (s_max + s_min) / 2
        _, score = area_under_calibration_curve(pred, c * unc, gt, return_integral=True)

        # Setting an anchor for avoiding being stuck
        if res_anchor == score:
            repeat += 1
        else:
            repeat = 0
        res_anchor = score

    return c
