# Code for "Reserve Speculative Attacks", by
# Manuel Amador, Javier Bianchi, Luigi Bocola, and Fabrizio Perri
# published at
# The Journal of Economics Dynamics and Control,
# November 2016, Volume 72, Pages 125-137
#
# Python 3.5
#
# Main file

import modelclass as gs
import time
from matplotlib import pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    # Money demand parameters
    rbar = None  # Using log-log money demand
    psi = 415  # elasticity parameter

    zlb = - 100  # No ZLB

    # Foreign interest rate process
    ihigh = gs.annual_to_monthly(1.5 / 100)  # high interest rate
    ilow = gs.annual_to_monthly(0.0 / 100)  # low interest rate
    interest_transition = {
        0: (1 - 1.7/100, 1.7/100),
        1: (1.0/100, 1 - 1.0/100)
    }  # transition matrix for the interest rate

    lam = 0.4 / 100  # probability of abandonment (lambda)
    ebar = 0.7  # exchange rate if abandonment occurs

    epeg = 1.0  # level of exchange rate peg
    gamma = 3.5 / 100  # probability of money demand increasing by g
    g_val = 0.505  # growth rate of money demand ( g )

    # initial level of money demand
    b0 = psi * ((1 + ihigh) * (lam * ebar / epeg + (1 - lam)) - 1)

    N = 24  # number of states

    par = {
        'psi': psi,
        'rbar': rbar,
        'zlb': zlb,
        'b0': b0,
        'N': N,
        'nw': 0.2,   # net worth of the central bank
        'pibar': 1.6,  # constraint on central bank losses
        'epeg': epeg,
        'ebar': ebar,
        'gamma': gamma,
        'lam': lam,
        'g_val': g_val,
        'c_states': (0, 1),  # (low perm, low transitory, high transitory)
        'c_transition': interest_transition,
        'c_istar_values': (ilow, ihigh)
    }

    sns.set_style("whitegrid")  # optional seabird settings
    t1 = time.clock()
    model = gs.Model(**par)
    print("time : {} seconds".format(time.clock() - t1))
    gs.do_plots(model, save_to_file=False, file_name='benchmark_figure.pdf')

    robustness = {}
    labels = {}
    markers = ['^--', 'o-', 's--']

    robustness['ebar'] = []
    labels['ebar'] = []
    for ebar_val, m in zip([0.65, 0.7], markers):
        par2 = par.copy()
        par2['ebar'] = ebar_val
        robustness['ebar'].append(gs.Model(**par2))
        lab = {}
        lab['label'] = '$1 - \\bar S = {:.2}$'.format(1 - ebar_val)
        lab['marker'] = m
        labels['ebar'].append(lab)

    robustness['lambda'] = []
    labels['lambda'] = []
    for lambda_values, m in zip([0.6, 0.4], markers):
        par2 = par.copy()
        par2['lam'] = lambda_values / 100
        robustness['lambda'].append(gs.Model(**par2))
        lab = {}
        lab['label'] = '$\\lambda = ' + str(lambda_values) + '$'
        lab['marker'] = m
        labels['lambda'].append(lab)

    robustness['pibar'] = []
    labels['pibar'] = []
    for pibar_val, m in zip([1, 1.6], markers):
        par2 = par.copy()
        par2['pibar'] = pibar_val
        robustness['pibar'].append(gs.Model(**par2))
        lab = {}
        lab['label'] = '$\\bar \\Pi = ' + str(pibar_val) + '$'
        lab['marker'] = m
        labels['pibar'].append(lab)

    robustness['psi'] = []
    labels['psi'] = []
    for psi, m in zip(
                    [212, 415],
                    markers):
        par2 = par.copy()
        par2['psi'] = psi
        par2['b0'] = (
            psi * ((1 + par2['c_istar_values'][1]) *
                   (par2['lam'] * par2['ebar'] / par2['epeg'] +
                   (1 - par2['lam'])) -
                   1))
        robustness['psi'].append(gs.Model(**par2))
        lab = {}
        lab['label'] = '$\\psi = {:.2}$'.format(psi / 12 / 100 - 0.01)
        lab['marker'] = m
        labels['psi'].append(lab)

    fs = 15
    b_range = list(range(4))
    plt.figure(figsize=(10, 8))
    for i, key in enumerate(['ebar', 'lambda', 'pibar', 'psi']):
        plt.subplot(221 + i)
        model_list = robustness[key]
        temp = labels[key]
        e_min = model_list[0].epeg
        c = 1  # plot only the high interest rate
        for m, lab in zip(model_list, temp):
            e_list = [m.e_rate['top'][(1, b, c)] for b in b_range]
            plt.plot(e_list, lab['marker'], label=lab['label'])
            e_min = min(e_min, min(e_list))
        plt.ylim([.85, 1.01])
        plt.title([
            'A. Size of Appreciation Shock',
            'B. Probability of Appreciation Shock',
            'C. Tightness of Loss Constraint',
            'D. Elasticity of Money Demand'][i], fontsize=fs)
        plt.xticks(b_range, [b + 1 for b in b_range])
        if i > 1:
            plt.xlabel('b')
        if i in (0, 2):
            plt.ylabel('Exchange Rate')
        plt.legend(loc=3, handlelength=4)

    plt.subplots_adjust(wspace=0.3)
    # plt.savefig('robustness_E.pdf')

    plt.figure(figsize=(10, 8))
    for i, key in enumerate(['ebar', 'lambda', 'pibar', 'psi']):
        plt.subplot(221 + i)
        model_list = robustness[key]
        temp = labels[key]
        c = 1  # plot only the high interest rate
        for m, lab in zip(model_list, temp):
            plt.plot([m.reserves[(1, b, c)]
                      for b in b_range], lab['marker'], label=lab['label'])
        plt.legend(loc=4, handlelength=4)
        plt.title([
            'A. Size of Appreciation Shock',
            'B. Probability of Appreciation Shock',
            'C. Tightness of Loss Constraint',
            'D. Elasticity of Money Demand'][i], fontsize=fs)
        if i > 1:
            plt.xlabel('b')
        if i in (0, 2):
            plt.ylabel('Reserves')
        plt.xticks(b_range, [b + 1 for b in b_range])
        plt.ylim([0, 12])

    plt.subplots_adjust(wspace=0.25)
    # plt.savefig('robustness_F.pdf')


robustness['psi'][0].e_rate['top']

plt.show()
